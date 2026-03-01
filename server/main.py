from __future__ import annotations

import argparse
import asyncio
import os
import uuid
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any

import structlog
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File
from fastapi.responses import Response, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from common.constants import API_PREFIX
from common.schemas import (
    WorkerRegisterRequest, WorkerRegisterResponse,
    HeartbeatRequest,
    TrainingStartRequest, TrainingStatus, TrainingConfig,
    DataAssignment, SubmitUpdateRequest, SubmitUpdateResponse,
    TrainingResults
)
from common.model import CifarCNN, state_dict_to_cpu
from server.config import ServerConfig
from server.logging_setup import configure_logging
from server.auth import AuthConfig, create_worker_token, worker_auth_dependency
from server.db import init_db, Worker, TrainingRun, RoundMetric
from server.data_manager import create_cifar10_partitions, get_indices_file_path
from server.coordinator import Coordinator, UpdateRecord, deserialize_state_dict
from server.eval import evaluate_cifar10

from sqlalchemy import select, update
import datetime as dt

log = structlog.get_logger()

def create_app(cfg: ServerConfig) -> FastAPI:
    configure_logging(cfg.log_level)
    app = FastAPI(title="LydianAI PoC Server", version="0.1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    auth_cfg = AuthConfig(jwt_secret=cfg.jwt_secret)
    worker_auth = worker_auth_dependency(auth_cfg)

    coordinator = Coordinator(artifacts_dir=cfg.artifacts_dir)

    @app.on_event("startup")
    async def _startup():
        engine, Session = await init_db(cfg.db_url)
        app.state.db_engine = engine
        app.state.db = Session
        app.state.coordinator = coordinator
        Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)
        log.info("server_started", host=cfg.host, port=cfg.port, db_url=cfg.db_url, artifacts_dir=cfg.artifacts_dir)

    @app.on_event("shutdown")
    async def _shutdown():
        engine = getattr(app.state, "db_engine", None)
        if engine is not None:
            await engine.dispose()

    def get_session():
        return app.state.db()

    # --- Health
    @app.get("/health")
    async def health():
        return {"status": "ok"}

    # --- Workers
    @app.post(f"{API_PREFIX}/workers/register", response_model=WorkerRegisterResponse)
    async def register_worker(req: WorkerRegisterRequest):
        worker_id = str(uuid.uuid4())
        token = create_worker_token(auth_cfg, worker_id)

        async with get_session() as s:
            w = Worker(
                id=worker_id,
                name=req.name,
                tailscale_ip=req.tailscale_ip,
                gpu_count=req.gpu_count,
                gpus={"gpus": [g.model_dump() for g in req.gpus]},
                is_active=True,
                last_seen=dt.datetime.utcnow(),
            )
            s.add(w)
            await s.commit()

        log.info("worker_registered", worker_id=worker_id, name=req.name, tailscale_ip=req.tailscale_ip, gpu_count=req.gpu_count)
        return WorkerRegisterResponse(worker_id=worker_id, access_token=token)

    @app.post(f"{API_PREFIX}/workers/heartbeat")
    async def worker_heartbeat(req: HeartbeatRequest, worker_id: str = Depends(worker_auth)):
        if req.worker_id != worker_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="worker_id mismatch")

        async with get_session() as s:
            await s.execute(
                update(Worker).where(Worker.id == worker_id).values(last_seen=dt.datetime.utcnow(), is_active=True)
            )
            await s.commit()

        return {"ok": True}

    # --- Training lifecycle
    @app.get(f"{API_PREFIX}/training/status", response_model=TrainingStatus)
    async def training_status():
        st = await coordinator.get_state()
        if not st:
            return TrainingStatus(state="idle", current_round=0, num_rounds=0, active_workers=0, message="no run")
        # count active workers as those seen in last 2 heartbeats (rough: 90s)
        cutoff = dt.datetime.utcnow() - dt.timedelta(seconds=90)
        async with get_session() as s:
            res = await s.execute(select(Worker).where(Worker.is_active == True))  # noqa
            workers = res.scalars().all()
            active = 0
            for w in workers:
                if w.last_seen and w.last_seen >= cutoff:
                    active += 1
        return TrainingStatus(
            run_id=st.run_id,
            state=st.state,
            current_round=st.current_round,
            num_rounds=st.num_rounds,
            active_workers=active,
            last_aggregate_time=(dt.datetime.utcfromtimestamp(st.last_aggregate_time).isoformat()+"Z") if st.last_aggregate_time else None,
            message=st.message,
        )

    @app.post(f"{API_PREFIX}/training/start", response_model=TrainingStatus)
    async def training_start(req: TrainingStartRequest):
        run_id = str(uuid.uuid4())

        # Snapshot active workers + their GPU counts
        async with get_session() as s:
            res = await s.execute(select(Worker))
            workers = res.scalars().all()
        if not workers:
            raise HTTPException(status_code=400, detail="No workers registered")

        worker_gpu_counts = {w.id: int(w.gpu_count) for w in workers}

        # Create partitions (CIFAR-10 train indices per worker)
        create_cifar10_partitions(cfg.artifacts_dir, run_id, worker_gpu_counts)

        # Start coordinator run
        await coordinator.start(run_id, req.config)

        # Record run in DB
        async with get_session() as s:
            tr = TrainingRun(
                id=run_id,
                state="running",
                config=req.config.model_dump(),
                current_round=0,
                message="running",
                updated_at=dt.datetime.utcnow(),
            )
            s.add(tr)
            await s.commit()

        return TrainingStatus(run_id=run_id, state="running", current_round=0, num_rounds=req.config.num_rounds, active_workers=len(workers), message="started")

    @app.get(f"{API_PREFIX}/training/config", response_model=TrainingConfig)
    async def get_training_config(worker_id: str = Depends(worker_auth)):
        st = await coordinator.get_state()
        if not st:
            raise HTTPException(status_code=404, detail="No active run")
        return st.config

    @app.get(f"{API_PREFIX}/training/run")
    async def get_current_run(worker_id: str = Depends(worker_auth)):
        st = await coordinator.get_state()
        if not st:
            raise HTTPException(status_code=404, detail="No active run")
        return {"run_id": st.run_id, "state": st.state, "current_round": st.current_round, "num_rounds": st.num_rounds, "message": st.message}

    @app.get(f"{API_PREFIX}/training/model")
    async def get_global_model(worker_id: str = Depends(worker_auth)):
        blob = await coordinator.get_global_model_blob()
        return Response(content=blob, media_type="application/octet-stream")

    @app.get(f"{API_PREFIX}/training/data-assignment", response_model=DataAssignment)
    async def get_data_assignment(worker_id: str = Depends(worker_auth)):
        st = await coordinator.get_state()
        if not st:
            raise HTTPException(status_code=404, detail="No active run")
        indices_path = get_indices_file_path(cfg.artifacts_dir, st.run_id, worker_id)
        if not os.path.exists(indices_path):
            raise HTTPException(status_code=404, detail="No indices file for worker")
        # Count samples
        idxs = json.loads(Path(indices_path).read_text(encoding="utf-8"))
        return DataAssignment(
            run_id=st.run_id,
            worker_id=worker_id,
            num_samples=len(idxs),
            indices_path=f"{API_PREFIX}/artifacts/{st.run_id}/cifar10_train_indices__{worker_id}.json",
        )

    @app.get(f"{API_PREFIX}/artifacts/{{run_id}}/{{filename}}")
    async def get_artifact(run_id: str, filename: str, worker_id: str = Depends(worker_auth)):
        p = Path(cfg.artifacts_dir) / run_id / filename
        if not p.exists():
            raise HTTPException(status_code=404, detail="Artifact not found")
        return FileResponse(str(p))

    @app.post(f"{API_PREFIX}/training/submit-update", response_model=SubmitUpdateResponse)
    async def submit_update(meta: SubmitUpdateRequest, update_file: UploadFile = File(...), worker_id: str = Depends(worker_auth)):
        if meta.worker_id != worker_id:
            raise HTTPException(status_code=403, detail="worker_id mismatch")

        st = await coordinator.get_state()
        if not st or st.run_id != meta.run_id:
            raise HTTPException(status_code=404, detail="Unknown run_id")
        blob = await update_file.read()

        upd = UpdateRecord(
            worker_id=worker_id,
            round_idx=meta.round_idx,
            num_samples=meta.num_samples,
            train_loss=meta.train_loss,
            train_accuracy=meta.train_accuracy,
            wall_time_s=meta.wall_time_s,
            state_blob=blob,
        )
        ok, msg = await coordinator.submit_update(upd)

        # Try aggregate (using workers considered active by last_seen)
        cutoff = dt.datetime.utcnow() - dt.timedelta(seconds=90)
        async with get_session() as s:
            res = await s.execute(select(Worker))
            workers = res.scalars().all()
        active_ids = [w.id for w in workers if (w.last_seen and w.last_seen >= cutoff)]

        agg = await coordinator.maybe_aggregate(active_worker_ids=active_ids)
        if agg is not None:
            # Persist round metric
            async with get_session() as s:
                s.add(RoundMetric(run_id=st.run_id, round_idx=agg["round_idx"], payload=agg))
                await s.execute(
                    update(TrainingRun).where(TrainingRun.id == st.run_id).values(
                        current_round=st.current_round,
                        state=st.state,
                        updated_at=dt.datetime.utcnow(),
                        message=st.message,
                    )
                )
                await s.commit()

            # If finished, do evaluation (CPU by default)
            if st.state == "completed" and st.final_test is None:
                try:
                    loss, acc = evaluate_cifar10(st.global_state, batch_size=256, device="cpu")
                    await coordinator.attach_final_test({"test_loss": loss, "test_accuracy": acc})
                    async with get_session() as s:
                        await s.execute(
                            update(TrainingRun).where(TrainingRun.id == st.run_id).values(
                                state="completed",
                                updated_at=dt.datetime.utcnow(),
                                message="training completed",
                            )
                        )
                        await s.commit()
                    log.info("final_evaluation_done", run_id=st.run_id, test_loss=loss, test_accuracy=acc)
                except Exception as e:
                    log.error("final_evaluation_failed", run_id=st.run_id, error=str(e))

        return SubmitUpdateResponse(accepted=ok, message=msg)

    @app.get(f"{API_PREFIX}/training/results", response_model=TrainingResults)
    async def training_results():
        st = await coordinator.get_state()
        if not st:
            raise HTTPException(status_code=404, detail="No run")
        out = TrainingResults(run_id=st.run_id, state=st.state, history=st.history)
        if st.final_test:
            out.final_test_loss = st.final_test.get("test_loss")
            out.final_test_accuracy = st.final_test.get("test_accuracy")
        return out

    return app

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    cfg = ServerConfig()
    host = args.host or cfg.host
    port = args.port or cfg.port

    app = create_app(cfg)

    import uvicorn
    uvicorn.run(app, host=host, port=port, log_level="info")

if __name__ == "__main__":
    main()

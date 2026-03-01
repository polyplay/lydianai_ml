from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, List

import structlog
from fastapi import FastAPI, Depends, HTTPException, status, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, FileResponse

from sqlalchemy import select, update
import datetime as dt

from common.constants import API_PREFIX
from common.schemas import (
    WorkerRegisterRequest, WorkerRegisterResponse,
    HeartbeatRequest,
    TrainingStartRequest, TrainingStatus, TrainingConfig,
    DataAssignment, SubmitUpdateResponse,
    TrainingResults, HealthResponse, WorkerListEntry,
    SubmitUpdateRequest,
)
from server.config import ServerConfig
from server.logging_setup import configure_logging
from server.auth import AuthConfig, create_worker_token, worker_auth_dependency
from server.db import init_db, Worker, TrainingRun, RoundMetric
from server.data_manager import create_cifar10_partitions, get_indices_file_path
from server.coordinator import Coordinator, UpdateRecord
from server.eval import evaluate_cifar10

log = structlog.get_logger()

_start_time: float = 0.0


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def create_app(cfg: ServerConfig) -> FastAPI:
    configure_logging(cfg.log_level)

    app = FastAPI(title="LydianAI Server", version="0.1.0")

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

    def get_session():
        return app.state.db()

    @asynccontextmanager
    async def lifespan(app_: FastAPI):
        global _start_time
        _start_time = time.time()

        engine, Session = await init_db(cfg.db_url)
        app_.state.db_engine = engine
        app_.state.db = Session
        Path(cfg.artifacts_dir).mkdir(parents=True, exist_ok=True)

        log.info(
            "server_started",
            host=cfg.host,
            port=cfg.port,
            db_url=cfg.db_url,
            artifacts_dir=cfg.artifacts_dir,
        )

        watchdog_task = asyncio.create_task(_aggregation_watchdog(coordinator, app_))
        try:
            yield
        finally:
            watchdog_task.cancel()
            try:
                await watchdog_task
            except Exception:
                pass
            engine = getattr(app_.state, "db_engine", None)
            if engine is not None:
                await engine.dispose()

    # attach lifespan after defining it
    app.router.lifespan_context = lifespan

    # ─── Health ─────────────────────────────────────────────

    @app.get("/health", response_model=HealthResponse)
    @app.get(f"{API_PREFIX}/health", response_model=HealthResponse)
    async def health():
        """Health check with active worker count, uptime, and training status."""
        st = await coordinator.get_state()
        active = 0
        try:
            cutoff = _utcnow() - dt.timedelta(seconds=90)
            async with get_session() as s:
                res = await s.execute(select(Worker).where(Worker.is_active == True))  # noqa: E712
                workers = res.scalars().all()
                for w in workers:
                    if w.last_seen:
                        last_seen = w.last_seen
                        if last_seen.tzinfo is None:
                            last_seen = last_seen.replace(tzinfo=dt.timezone.utc)
                        if last_seen >= cutoff:
                            active += 1
        except Exception:
            pass

        return HealthResponse(
            status="healthy",
            uptime_sec=round(time.time() - _start_time, 1),
            active_workers=active,
            active_training=st.run_id if st and st.state == "running" else None,
        )

    # ─── Workers ─────────────────────────────────────────────

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
                last_seen=_utcnow(),
            )
            s.add(w)
            await s.commit()

        log.info(
            "worker_registered",
            worker_id=worker_id,
            name=req.name,
            tailscale_ip=req.tailscale_ip,
            gpu_count=req.gpu_count,
        )
        return WorkerRegisterResponse(worker_id=worker_id, access_token=token)

    @app.post(f"{API_PREFIX}/workers/heartbeat")
    async def worker_heartbeat(req: HeartbeatRequest, worker_id: str = Depends(worker_auth)):
        if req.worker_id != worker_id:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="worker_id mismatch")

        async with get_session() as s:
            await s.execute(
                update(Worker)
                .where(Worker.id == worker_id)
                .values(last_seen=_utcnow(), is_active=True)
            )
            await s.commit()

        return {"ok": True}

    @app.get(f"{API_PREFIX}/workers")
    async def list_workers():
        """List all registered workers and their status."""
        async with get_session() as s:
            res = await s.execute(select(Worker))
            workers = res.scalars().all()

        now = _utcnow()
        entries = []
        for w in workers:
            last_seen = w.last_seen
            if last_seen and last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=dt.timezone.utc)
            age = (now - last_seen).total_seconds() if last_seen else 9999.0
            is_active = bool(w.is_active and age < 90)

            entries.append(
                WorkerListEntry(
                    worker_id=w.id,
                    name=w.name,
                    tailscale_ip=w.tailscale_ip,
                    gpu_count=w.gpu_count,
                    gpus=w.gpus,
                    is_active=is_active,
                    last_seen=last_seen.isoformat() if last_seen else None,
                ).model_dump()
            )

        return {"workers": entries, "total": len(entries)}

    # ─── Training lifecycle ─────────────────────────────────

    @app.get(f"{API_PREFIX}/training/status", response_model=TrainingStatus)
    async def training_status():
        st = await coordinator.get_state()
        if not st:
            return TrainingStatus(state="idle", current_round=0, num_rounds=0, active_workers=0, message="no run")

        cutoff = _utcnow() - dt.timedelta(seconds=90)
        async with get_session() as s:
            res = await s.execute(select(Worker).where(Worker.is_active == True))  # noqa: E712
            workers = res.scalars().all()
            active = 0
            for w in workers:
                if not w.last_seen:
                    continue
                last_seen = w.last_seen
                if last_seen.tzinfo is None:
                    last_seen = last_seen.replace(tzinfo=dt.timezone.utc)
                if last_seen >= cutoff:
                    active += 1

        return TrainingStatus(
            run_id=st.run_id,
            state=st.state,
            current_round=st.current_round,
            num_rounds=st.num_rounds,
            active_workers=active,
            last_aggregate_time=(
                dt.datetime.fromtimestamp(st.last_aggregate_time, tz=dt.timezone.utc).isoformat()
                if st.last_aggregate_time
                else None
            ),
            message=st.message,
        )

    @app.post(f"{API_PREFIX}/training/start", response_model=TrainingStatus)
    async def training_start(req: TrainingStartRequest):
        run_id = str(uuid.uuid4())

        async with get_session() as s:
            res = await s.execute(select(Worker))
            workers = res.scalars().all()

        if not workers:
            raise HTTPException(status_code=400, detail="No workers registered")

        worker_gpu_counts = {w.id: int(w.gpu_count) for w in workers}

        create_cifar10_partitions(cfg.artifacts_dir, run_id, worker_gpu_counts)
        await coordinator.start(run_id, req.config)

        async with get_session() as s:
            tr = TrainingRun(
                id=run_id,
                state="running",
                config=req.config.model_dump(),
                current_round=0,
                message="running",
                updated_at=_utcnow(),
            )
            s.add(tr)
            await s.commit()

        return TrainingStatus(
            run_id=run_id,
            state="running",
            current_round=0,
            num_rounds=req.config.num_rounds,
            active_workers=len(workers),
            message="started",
        )

    @app.get(f"{API_PREFIX}/training/config", response_model=TrainingConfig)
    async def get_training_config(worker_id: str = Depends(worker_auth)):
        st = await coordinator.get_state()
        if not st:
            raise HTTPException(status_code=404, detail="No active run")
        return st.config

    # This endpoint MUST NOT 404 when no run exists. Workers poll it.
    @app.get(f"{API_PREFIX}/training/run")
    async def get_current_run(worker_id: str = Depends(worker_auth)):
        st = await coordinator.get_state()
        if not st:
            return {"run_id": None, "state": "idle", "current_round": 0, "num_rounds": 0, "message": "no run"}
        return {
            "run_id": st.run_id,
            "state": st.state,
            "current_round": st.current_round,
            "num_rounds": st.num_rounds,
            "message": st.message,
        }

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
    async def submit_update(
        update_file: UploadFile = File(...),
        metadata_json: str = Form(...),
        worker_id: str = Depends(worker_auth),
    ):
        # Parse metadata from JSON form field
        try:
            meta = SubmitUpdateRequest(**json.loads(metadata_json))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid metadata: {e}")

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

        # Try aggregation immediately after receiving update
        await _try_aggregate(coordinator, app)

        return SubmitUpdateResponse(accepted=ok, message=msg)

    # ─── Results ────────────────────────────────────────────

    @app.get(f"{API_PREFIX}/training/results", response_model=TrainingResults)
    async def training_results():
        st = await coordinator.get_state()
        if not st:
            raise HTTPException(status_code=404, detail="No run")

        total_time = round(time.time() - st.created_at, 1) if st.created_at else None

        out = TrainingResults(
            run_id=st.run_id,
            state=st.state,
            history=st.history,
            total_rounds_completed=st.current_round,
            total_training_time_sec=total_time,
        )
        if st.final_test:
            out.final_test_loss = st.final_test.get("test_loss")
            out.final_test_accuracy = st.final_test.get("test_accuracy")
        return out

    return app


# ─── Aggregation helpers ─────────────────────────────────────────

async def _get_active_worker_ids(app: FastAPI) -> List[str]:
    """Get IDs of workers seen in the last 90 seconds."""
    cutoff = _utcnow() - dt.timedelta(seconds=90)
    try:
        async with app.state.db() as s:
            res = await s.execute(select(Worker))
            workers = res.scalars().all()

        active: List[str] = []
        for w in workers:
            if not w.last_seen:
                continue
            last_seen = w.last_seen
            if last_seen.tzinfo is None:
                last_seen = last_seen.replace(tzinfo=dt.timezone.utc)
            if last_seen >= cutoff:
                active.append(w.id)
        return active
    except Exception:
        return []


async def _try_aggregate(coordinator: Coordinator, app: FastAPI) -> None:
    """Attempt aggregation and handle post-aggregation tasks."""
    active_ids = await _get_active_worker_ids(app)
    agg = await coordinator.maybe_aggregate(active_worker_ids=active_ids)
    if agg is None:
        return

    st = await coordinator.get_state()
    if not st:
        return

    # Persist round metric + training run state
    try:
        async with app.state.db() as s:
            s.add(RoundMetric(run_id=st.run_id, round_idx=agg["round_idx"], payload=agg))
            await s.execute(
                update(TrainingRun)
                .where(TrainingRun.id == st.run_id)
                .values(
                    current_round=st.current_round,
                    state=st.state,
                    updated_at=_utcnow(),
                    message=st.message,
                )
            )
            await s.commit()
    except Exception as e:
        log.error("db_persist_failed", error=str(e))

    # If finished, do evaluation
    if st.state == "completed" and st.final_test is None:
        try:
            loss, acc = evaluate_cifar10(st.global_state, batch_size=256, device="cpu")
            await coordinator.attach_final_test({"test_loss": loss, "test_accuracy": acc})
            async with app.state.db() as s:
                await s.execute(
                    update(TrainingRun)
                    .where(TrainingRun.id == st.run_id)
                    .values(state="completed", updated_at=_utcnow(), message="training completed")
                )
                await s.commit()
            log.info("final_evaluation_done", run_id=st.run_id, test_loss=loss, test_accuracy=acc)
        except Exception as e:
            log.error("final_evaluation_failed", run_id=st.run_id, error=str(e))


async def _aggregation_watchdog(coordinator: Coordinator, app: FastAPI) -> None:
    """Background task: periodically check if a round should be aggregated."""
    while True:
        try:
            await asyncio.sleep(30)
            st = await coordinator.get_state()
            if st and st.state == "running":
                await _try_aggregate(coordinator, app)
        except asyncio.CancelledError:
            break
        except Exception as e:
            log.warning("watchdog_error", error=str(e))


# ─── Entry Point ───────────────────────────────────────────────

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

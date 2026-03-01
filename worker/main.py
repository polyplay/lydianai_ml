from __future__ import annotations
import argparse
import asyncio
import json
import socket
import time
from typing import Optional

import structlog

from common.schemas import WorkerRegisterRequest, GPUInfo, HeartbeatRequest, SubmitUpdateRequest
from worker.config import WorkerConfig
from worker.logging_setup import configure_logging
from worker.gpu_manager import get_gpu_inventory, get_gpu_utilization_snapshot
from worker.comms import ServerClient
from worker.trainer import train_one_round

log = structlog.get_logger()

def guess_local_ip() -> str:
    # Best-effort IP selection; with Tailscale you can also pass explicitly via CLI.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"

async def heartbeat_loop(client: ServerClient, worker_id: str, interval_s: int):
    while True:
        snap = get_gpu_utilization_snapshot()
        payload = HeartbeatRequest(worker_id=worker_id, gpu_utilization=snap, extra=None).model_dump()
        try:
            await client.heartbeat(payload)
            log.info("heartbeat_sent", worker_id=worker_id)
        except Exception as e:
            log.warn("heartbeat_failed", worker_id=worker_id, error=str(e))
        await asyncio.sleep(interval_s)

async def training_loop(cfg: WorkerConfig):
    configure_logging("INFO")
    client = ServerClient(cfg.server_url)

    gpu_count, gpus = get_gpu_inventory()
    req = WorkerRegisterRequest(
        name=cfg.name,
        tailscale_ip=cfg.tailscale_ip,
        gpu_count=gpu_count,
        gpus=[GPUInfo(**g) for g in gpus],
    )
    resp = await client.with_retries(client.register, req, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)
    log.info("registered", worker_id=resp.worker_id)

    asyncio.create_task(heartbeat_loop(client, resp.worker_id, cfg.heartbeat_interval_s))

    last_round_seen: Optional[int] = None

    while True:
        run = await client.with_retries(client.get_run, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)
        if run.get("state") != "running":
            log.info("no_active_training", state=run.get("state"), message=run.get("message"))
            await asyncio.sleep(5)
            continue

        run_id = run["run_id"]
        round_idx = int(run["current_round"])

        if last_round_seen is not None and round_idx == last_round_seen:
            await asyncio.sleep(2)
            continue

        config = await client.with_retries(client.get_config, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)
        assignment = await client.with_retries(client.get_data_assignment, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)

        idx_bytes = await client.with_retries(client.download_indices, assignment.indices_path, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)
        indices = json.loads(idx_bytes.decode("utf-8"))

        model_blob = await client.with_retries(client.get_model_blob, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)

        device = "cuda:0" if gpu_count > 0 else "cpu"
        log.info("round_start", run_id=run_id, round_idx=round_idx, num_samples=len(indices), device=device)

        update_blob, loss, acc, wall = train_one_round(
            model_blob=model_blob,
            indices=indices,
            data_dir=cfg.data_dir,
            local_epochs=config.local_epochs,
            batch_size=config.batch_size,
            lr=config.learning_rate,
            momentum=config.momentum,
            device=device,
            use_data_parallel=True,
        )

        meta = SubmitUpdateRequest(
            run_id=run_id,
            round_idx=round_idx,
            worker_id=resp.worker_id,
            num_samples=len(indices),
            train_loss=float(loss),
            train_accuracy=float(acc),
            wall_time_s=float(wall),
        )

        sr = await client.with_retries(client.submit_update, meta, update_blob, attempts=cfg.retry_attempts, backoff_base=cfg.retry_backoff_base)
        log.info("round_submitted", run_id=run_id, round_idx=round_idx, accepted=sr.accepted, message=sr.message)

        last_round_seen = round_idx
        await asyncio.sleep(1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", required=True, help="Server base URL, e.g. http://100.x.x.x:8000")
    parser.add_argument("--name", required=True, help="Worker name")
    parser.add_argument("--tailscale-ip", default=None, help="This machine's Tailscale IP (100.x.x.x). If omitted, best-effort guess.")
    args = parser.parse_args()

    tailscale_ip = args.tailscale_ip or guess_local_ip()
    cfg = WorkerConfig(server_url=args.server, name=args.name, tailscale_ip=tailscale_ip)

    asyncio.run(training_loop(cfg))

if __name__ == "__main__":
    main()

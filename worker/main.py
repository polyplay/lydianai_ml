"""LydianAI ML Worker Agent — Main entry point.
"""

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
    # Best-effort IP selection; with Tailscale it can also be passed explicitly via CLI.
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "0.0.0.0"


def _get_power_draw_watts() -> Optional[float]:
    """Get total GPU power draw in watts via nvidia-smi (best-effort)."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            total = sum(float(line.strip()) for line in result.stdout.strip().split("\n") if line.strip())
            return total
    except Exception:
        pass
    return None


async def heartbeat_loop(client: ServerClient, worker_id: str, interval_s: int):
    """Send periodic heartbeats with retry on consecutive failures."""
    consecutive_failures = 0
    max_backoff = 60

    while True:
        snap = get_gpu_utilization_snapshot()
        payload = HeartbeatRequest(worker_id=worker_id, gpu_utilization=snap, extra=None).model_dump()
        try:
            await client.heartbeat(payload)
            consecutive_failures = 0
        except Exception as e:
            consecutive_failures += 1
            if consecutive_failures <= 3:
                log.warning("heartbeat_failed", worker_id=worker_id, error=str(e),
                            consecutive_failures=consecutive_failures)
            elif consecutive_failures % 10 == 0:
                log.error("heartbeat_persistent_failure", worker_id=worker_id,
                          consecutive_failures=consecutive_failures, error=str(e))

        # Back off on consecutive failures
        wait = min(interval_s * (2 ** min(consecutive_failures, 4)), max_backoff)
        await asyncio.sleep(wait if consecutive_failures > 0 else interval_s)


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
    resp = await client.with_retries(client.register, req,
                                     attempts=cfg.retry_attempts,
                                     backoff_base=cfg.retry_backoff_base)
    log.info("registered", worker_id=resp.worker_id, gpu_count=gpu_count)

    hb_task = asyncio.create_task(
        heartbeat_loop(client, resp.worker_id, cfg.heartbeat_interval_s)
    )

    last_round_seen: Optional[int] = None
    completed_logged = False

    try:
        while True:
            try:
                run = await client.with_retries(
                    client.get_run,
                    attempts=cfg.retry_attempts,
                    backoff_base=cfg.retry_backoff_base,
                )
            except Exception as e:
                log.warning("get_run_failed", error=str(e))
                await asyncio.sleep(5)
                continue

            state = run.get("state", "idle")

            # BUG-4 FIX: Exit cleanly on training completion
            if state == "completed":
                if not completed_logged:
                    log.info("training_completed", run_id=run.get("run_id"),
                             message=run.get("message"))
                    completed_logged = True
                log.info("worker_idle_after_completion",
                         message="Training finished. Worker will stay alive for new runs.")
                last_round_seen = None  # Reset for potential next run
                await asyncio.sleep(10)
                continue

            if state == "failed":
                log.error("training_failed", run_id=run.get("run_id"),
                          message=run.get("message"))
                last_round_seen = None
                await asyncio.sleep(10)
                continue

            if state != "running":
                log.debug("waiting_for_training", state=state)
                completed_logged = False
                await asyncio.sleep(5)
                continue

            # Training is running — reset completion flag
            completed_logged = False
            run_id = run["run_id"]
            round_idx = int(run["current_round"])

            if last_round_seen is not None and round_idx == last_round_seen:
                await asyncio.sleep(2)
                continue

            config = await client.with_retries(
                client.get_config,
                attempts=cfg.retry_attempts,
                backoff_base=cfg.retry_backoff_base,
            )
            assignment = await client.with_retries(
                client.get_data_assignment,
                attempts=cfg.retry_attempts,
                backoff_base=cfg.retry_backoff_base,
            )
            idx_bytes = await client.with_retries(
                client.download_indices, assignment.indices_path,
                attempts=cfg.retry_attempts,
                backoff_base=cfg.retry_backoff_base,
            )
            indices = json.loads(idx_bytes.decode("utf-8"))

            model_blob = await client.with_retries(
                client.get_model_blob,
                attempts=cfg.retry_attempts,
                backoff_base=cfg.retry_backoff_base,
            )

            device = "cuda:0" if gpu_count > 0 else "cpu"
            log.info("round_start", run_id=run_id, round_idx=round_idx,
                     num_samples=len(indices), device=device)

            # ISSUE-21: Record power before training for energy estimate
            power_before = _get_power_draw_watts()
            train_start = time.time()

            # ISSUE-14 FIX: pass weight_decay from config
            update_blob, loss, acc, wall = train_one_round(
                model_blob=model_blob,
                indices=indices,
                data_dir=cfg.data_dir,
                local_epochs=config.local_epochs,
                batch_size=config.batch_size,
                lr=config.learning_rate,
                momentum=config.momentum,
                weight_decay=config.weight_decay,
                device=device,
                use_data_parallel=True,
            )

            # ISSUE-21: Estimate energy consumed
            train_duration = time.time() - train_start
            power_after = _get_power_draw_watts()
            power_consumed_wh = None
            if power_before is not None and power_after is not None:
                avg_power = (power_before + power_after) / 2.0
                power_consumed_wh = avg_power * (train_duration / 3600.0)

            log.info("round_complete", run_id=run_id, round_idx=round_idx,
                     loss=round(loss, 4), accuracy=round(acc, 4),
                     wall_time_s=round(wall, 1), samples=len(indices),
                     power_consumed_wh=round(power_consumed_wh, 2) if power_consumed_wh else None)

            meta = SubmitUpdateRequest(
                run_id=run_id,
                round_idx=round_idx,
                worker_id=resp.worker_id,
                num_samples=len(indices),
                train_loss=float(loss),
                train_accuracy=float(acc),
                wall_time_s=float(wall),
                power_consumed_wh=power_consumed_wh,
            )

            sr = await client.with_retries(
                client.submit_update, meta, update_blob,
                attempts=cfg.retry_attempts,
                backoff_base=cfg.retry_backoff_base,
            )
            log.info("round_submitted", run_id=run_id, round_idx=round_idx,
                     accepted=sr.accepted, message=sr.message)

            last_round_seen = round_idx
            await asyncio.sleep(1)

    except KeyboardInterrupt:
        log.info("worker_interrupted")
    except asyncio.CancelledError:
        log.info("worker_cancelled")
    finally:
        hb_task.cancel()
        log.info("worker_shutdown")


def main():
    parser = argparse.ArgumentParser(description="LydianAI ML Worker Agent")
    parser.add_argument("--server", required=True,
                        help="Server base URL, e.g. http://100.x.x.x:8000")
    parser.add_argument("--name", required=True, help="Worker name")
    parser.add_argument("--tailscale-ip", default=None,
                        help="This machine's Tailscale IP (100.x.x.x). If omitted, best-effort guess.")
    parser.add_argument("--data-dir", default=None,
                        help="Local data directory (default: ./data)")
    args = parser.parse_args()

    tailscale_ip = args.tailscale_ip or guess_local_ip()
    cfg = WorkerConfig(server_url=args.server, name=args.name, tailscale_ip=tailscale_ip)
    if args.data_dir:
        # WorkerConfig is frozen, so we need to reconstruct
        cfg = WorkerConfig(
            server_url=args.server, name=args.name,
            tailscale_ip=tailscale_ip, data_dir=args.data_dir,
        )

    asyncio.run(training_loop(cfg))


if __name__ == "__main__":
    main()

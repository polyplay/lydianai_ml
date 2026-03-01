from __future__ import annotations
import argparse
import asyncio
import time

import httpx

from common.schemas import TrainingConfig, TrainingStartRequest

async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--server", required=True, help="Server base URL, e.g. http://100.x.x.x:8000")
    p.add_argument("--rounds", type=int, default=20)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--round-timeout-s", type=int, default=900)
    p.add_argument("--min-updates", type=int, default=1)
    args = p.parse_args()

    cfg = TrainingConfig(
        num_rounds=args.rounds,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        momentum=args.momentum,
        round_timeout_s=args.round_timeout_s,
        min_updates_per_round=args.min_updates,
    )

    req = TrainingStartRequest(config=cfg)

    async with httpx.AsyncClient(timeout=None) as client:
        r = await client.post(f"{args.server.rstrip('/')}/api/v1/training/start", json=req.model_dump())
        r.raise_for_status()
        st = r.json()
        run_id = st.get("run_id")
        print("Started run:", run_id)

        # Poll status until completed
        while True:
            s = await client.get(f"{args.server.rstrip('/')}/api/v1/training/status")
            s.raise_for_status()
            j = s.json()
            print("Status:", j)
            if j.get("state") in ("completed", "failed"):
                break
            await asyncio.sleep(5)

        res = await client.get(f"{args.server.rstrip('/')}/api/v1/training/results")
        res.raise_for_status()
        print("Results:", res.json())

if __name__ == "__main__":
    asyncio.run(main())

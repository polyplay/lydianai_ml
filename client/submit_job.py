"""LydianAI Client CLI — Submit training jobs and monitor progress.

Subcommands:
  start    - Start a new federated training session
  status   - Check current training progress
  monitor  - Continuously watch training progress
  results  - Fetch final training results
  workers  - List registered workers
  health   - Check server health
"""

from __future__ import annotations
import argparse
import asyncio
import sys
import time

import httpx

API = "/api/v1"


def _url(server: str, path: str) -> str:
    return f"{server.rstrip('/')}{API}{path}"


# ─── Subcommands ─────────────────────────────────────────────────────

async def cmd_health(args):
    """Check server health."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(_url(args.server, "/health"))
            r.raise_for_status()
            h = r.json()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.server}: {e}")
        sys.exit(1)

    print(f"Status:          {h.get('status', '?')}")
    print(f"Version:         {h.get('version', '?')}")
    print(f"Uptime:          {h.get('uptime_sec', 0):.0f}s")
    print(f"Active workers:  {h.get('active_workers', 0)}")
    print(f"Active training: {h.get('active_training', 'none')}")


async def cmd_start(args):
    """Start a new training session."""
    # Pre-flight health check
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(_url(args.server, "/health"))
            r.raise_for_status()
            health = r.json()
    except Exception as e:
        print(f"ERROR: Cannot reach server at {args.server}: {e}")
        sys.exit(1)

    active = health.get("active_workers", 0)
    print(f"Server: {health.get('status', '?')} | {active} active worker(s)")

    if active == 0:
        print("ERROR: No active workers. Start worker agents first.")
        sys.exit(1)

    config = {
        "dataset": "cifar10",
        "num_rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "momentum": args.momentum,
        "weight_decay": args.weight_decay,
        "round_timeout_s": args.round_timeout,
        "min_updates_per_round": args.min_updates,
    }

    print(f"\nStarting training:")
    for k, v in config.items():
        print(f"  {k}: {v}")
    print()

    try:
        async with httpx.AsyncClient(timeout=30) as c:
            r = await c.post(
                _url(args.server, "/training/start"),
                json={"config": config},
            )
            r.raise_for_status()
            result = r.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        print(f"ERROR: {detail}")
        sys.exit(1)

    print(f"Training started!")
    print(f"  Run ID:   {result.get('run_id', '?')}")
    print(f"  Workers:  {result.get('active_workers', '?')}")
    print(f"  Rounds:   {result.get('num_rounds', '?')}")


async def cmd_status(args):
    """Check training status."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(_url(args.server, "/training/status"))
            r.raise_for_status()
            st = r.json()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"Run ID:    {st.get('run_id', 'none')}")
    print(f"State:     {st.get('state', '?')}")
    print(f"Round:     {st.get('current_round', 0)}/{st.get('num_rounds', 0)}")
    print(f"Workers:   {st.get('active_workers', 0)} active")
    if st.get("last_aggregate_time"):
        print(f"Last agg:  {st['last_aggregate_time']}")
    if st.get("message"):
        print(f"Message:   {st['message']}")


async def cmd_monitor(args):
    """Continuously monitor training progress."""
    last_round = -1
    print("Monitoring training progress (Ctrl+C to stop)...\n")

    try:
        while True:
            try:
                async with httpx.AsyncClient(timeout=10) as c:
                    r = await c.get(_url(args.server, "/training/status"))
                    r.raise_for_status()
                    st = r.json()

                current = st.get("current_round", 0)
                state = st.get("state", "idle")

                if current != last_round:
                    ts = time.strftime("%H:%M:%S")
                    print(f"  [{ts}] round {current}/{st.get('num_rounds', '?')} "
                          f"state={state} workers={st.get('active_workers', '?')} "
                          f"{st.get('message', '')}")
                    last_round = current

                if state == "completed":
                    print(f"\nTraining completed!")
                    await cmd_results(args)
                    return

                if state == "failed":
                    print(f"\nTraining failed: {st.get('message', '')}")
                    return

            except Exception as e:
                print(f"  (connection error: {e}, retrying...)")

            await asyncio.sleep(args.interval)

    except KeyboardInterrupt:
        print("\nMonitoring stopped.")


async def cmd_results(args):
    """Get final training results."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(_url(args.server, "/training/results"))
            r.raise_for_status()
            res = r.json()
    except httpx.HTTPStatusError as e:
        detail = e.response.json().get("detail", str(e)) if e.response else str(e)
        print(f"ERROR: {detail}")
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    print(f"\n{'='*55}")
    print(f"  TRAINING RESULTS")
    print(f"{'='*55}")
    print(f"  Run ID:          {res.get('run_id', '?')}")
    print(f"  State:           {res.get('state', '?')}")

    acc = res.get("final_test_accuracy")
    loss = res.get("final_test_loss")
    if acc is not None:
        print(f"  Final Accuracy:  {acc:.4f} ({acc*100:.1f}%)")
    if loss is not None:
        print(f"  Final Loss:      {loss:.4f}")

    total_rounds = res.get("total_rounds_completed", 0)
    total_time = res.get("total_training_time_sec")
    print(f"  Rounds:          {total_rounds}")
    if total_time:
        print(f"  Total Time:      {total_time:.1f}s ({total_time/60:.1f}min)")

    history = res.get("history", [])
    if history:
        print(f"\n  Round History ({len(history)} rounds):")
        print(f"  {'Round':>5}  {'Workers':>7}  {'Aggregated At':<26}")
        print(f"  {'─'*5}  {'─'*7}  {'─'*26}")
        for h in history:
            workers = h.get("num_updates", "?")
            agg_at = h.get("aggregated_at", "?")
            ridx = h.get("round_idx", "?")
            print(f"  {ridx:>5}  {workers:>7}  {agg_at:<26}")
    print()


async def cmd_workers(args):
    """List registered workers."""
    try:
        async with httpx.AsyncClient(timeout=10) as c:
            r = await c.get(_url(args.server, "/workers"))
            r.raise_for_status()
            data = r.json()
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    workers = data.get("workers", [])
    total = data.get("total", 0)
    print(f"Workers ({total}):\n")
    print(f"  {'Name':<20} {'ID':<38} {'Active':<8} {'GPUs':>4} {'IP':<16} {'Last Seen'}")
    print(f"  {'─'*20} {'─'*38} {'─'*8} {'─'*4} {'─'*16} {'─'*20}")
    for w in workers:
        print(f"  {w.get('name', '?'):<20} "
              f"{w.get('worker_id', '?'):<38} "
              f"{'✓' if w.get('is_active') else '✗':<8} "
              f"{w.get('gpu_count', 0):>4} "
              f"{w.get('tailscale_ip', '?'):<16} "
              f"{w.get('last_seen', '?')}")


# ─── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LydianAI PoC Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m client.submit_job health   --server http://100.64.1.1:8000
  python -m client.submit_job start    --server http://100.64.1.1:8000 --rounds 20
  python -m client.submit_job monitor  --server http://100.64.1.1:8000
  python -m client.submit_job results  --server http://100.64.1.1:8000
  python -m client.submit_job workers  --server http://100.64.1.1:8000
""",
    )
    parser.add_argument("--server", default="http://localhost:8000", help="Server URL")

    sub = parser.add_subparsers(dest="command", help="Command")

    # health
    sub.add_parser("health", help="Check server health")

    # start
    sp = sub.add_parser("start", help="Start training")
    sp.add_argument("--rounds", type=int, default=20, help="Federated rounds (default: 20)")
    sp.add_argument("--local-epochs", type=int, default=1, help="Local epochs per round (default: 1)")
    sp.add_argument("--batch-size", type=int, default=64, help="Batch size (default: 64)")
    sp.add_argument("--lr", type=float, default=0.01, help="Learning rate (default: 0.01)")
    sp.add_argument("--momentum", type=float, default=0.9, help="SGD momentum (default: 0.9)")
    sp.add_argument("--weight-decay", type=float, default=0.0001, help="Weight decay (default: 0.0001)")
    sp.add_argument("--round-timeout", type=int, default=900, help="Round timeout in seconds (default: 900)")
    sp.add_argument("--min-updates", type=int, default=1, help="Min worker updates per round (default: 1)")

    # status
    sub.add_parser("status", help="Check training status")

    # monitor
    mp = sub.add_parser("monitor", help="Continuously monitor training")
    mp.add_argument("--interval", type=int, default=5, help="Poll interval in seconds (default: 5)")

    # results
    sub.add_parser("results", help="Get final training results")

    # workers
    sub.add_parser("workers", help="List registered workers")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    dispatch = {
        "health": cmd_health,
        "start": cmd_start,
        "status": cmd_status,
        "monitor": cmd_monitor,
        "results": cmd_results,
        "workers": cmd_workers,
    }

    asyncio.run(dispatch[args.command](args))


if __name__ == "__main__":
    main()

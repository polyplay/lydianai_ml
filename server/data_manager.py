from __future__ import annotations
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import structlog

log = structlog.get_logger()

def ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)

def create_cifar10_partitions(
    artifacts_dir: str,
    run_id: str,
    worker_gpu_counts: Dict[str, int],
    total_train_samples: int = 50000,
    seed: int = 1337,
) -> Dict[str, List[int]]:
    """Create index partitions proportional to gpu_count for CIFAR-10 train set.

    CIFAR-10 train split size is 50,000 samples.
    """
    ensure_dir(artifacts_dir)
    run_dir = Path(artifacts_dir) / run_id
    ensure_dir(str(run_dir))

    workers = list(worker_gpu_counts.items())
    if not workers:
        raise ValueError("No workers available for partitioning")

    total_weight = sum(max(1, g) for _, g in workers)  # treat 0 GPU as 1 (CPU fallback)
    # Shuffle indices deterministically
    rng = random.Random(seed)
    indices = list(range(total_train_samples))
    rng.shuffle(indices)

    # Allocate sizes
    sizes = {}
    remaining = total_train_samples
    for i, (wid, g) in enumerate(workers):
        weight = max(1, g)
        if i == len(workers) - 1:
            n = remaining
        else:
            n = int(round(total_train_samples * (weight / total_weight)))
            n = max(1, min(n, remaining - (len(workers)-i-1)))  # keep at least 1 for others
            remaining -= n
        sizes[wid] = n

    # Slice
    out: Dict[str, List[int]] = {}
    cursor = 0
    for wid, _ in workers:
        n = sizes[wid]
        out[wid] = indices[cursor:cursor+n]
        cursor += n

    # Persist
    for wid, idxs in out.items():
        p = run_dir / f"cifar10_train_indices__{wid}.json"
        p.write_text(json.dumps(idxs), encoding="utf-8")

    meta = {
        "run_id": run_id,
        "total_train_samples": total_train_samples,
        "worker_gpu_counts": worker_gpu_counts,
        "sizes": {k: len(v) for k, v in out.items()},
    }
    (run_dir / "partition_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    log.info("partition_created", **meta)
    return out

def get_indices_file_path(artifacts_dir: str, run_id: str, worker_id: str) -> str:
    return str(Path(artifacts_dir) / run_id / f"cifar10_train_indices__{worker_id}.json")

from __future__ import annotations

import asyncio
import io
import time
import datetime as dt
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import torch
import structlog

from common.model import CifarCNN, state_dict_to_cpu
from common.schemas import TrainingConfig

log = structlog.get_logger()

def serialize_state_dict(state_dict: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict_to_cpu(state_dict), buf)
    return buf.getvalue()

def deserialize_state_dict(blob: bytes) -> dict:
    buf = io.BytesIO(blob)
    return torch.load(buf, map_location="cpu", weights_only=True)

def federated_average(states: List[dict], weights: List[int]) -> dict:
    """FedAvg: weighted average of model parameters by sample count.

    Handles:
    - Float/parameter tensors: weighted average
    - num_batches_tracked (BatchNorm): take from first worker (not averaged)
    - Non-tensor values: take from first worker
    """
    if not states:
        raise ValueError("No states to average")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Total weight must be > 0")
    avg: Dict[str, Any] = {}
    keys = states[0].keys()
    for k in keys:
        v0 = states[0][k]
        if not torch.is_tensor(v0):
            # Non-tensor (rare): preserve from first worker
            avg[k] = v0
            continue
        if "num_batches_tracked" in k:
            # BatchNorm counter: take from first worker, not averaged
            avg[k] = v0.clone()
            continue
        # Standard FedAvg: weighted sum
        vals = []
        for st, w in zip(states, weights):
            vals.append(st[k].float() * (float(w) / total))
        if vals:
            avg[k] = torch.stack(vals, dim=0).sum(dim=0)
    return avg

@dataclass
class UpdateRecord:
    worker_id: str
    round_idx: int
    num_samples: int
    train_loss: float
    train_accuracy: float
    wall_time_s: float
    received_at: float = field(default_factory=lambda: time.time())
    state_blob: bytes = b""

@dataclass
class TrainingState:
    run_id: str
    config: TrainingConfig
    state: str = "idle"  # idle|running|completed|failed
    current_round: int = 0
    num_rounds: int = 0
    created_at: float = field(default_factory=lambda: time.time())
    last_aggregate_time: Optional[float] = None
    message: str = ""
    global_state: dict = field(default_factory=dict)
    updates_by_round: Dict[int, Dict[str, UpdateRecord]] = field(default_factory=dict)
    history: List[dict] = field(default_factory=list)
    final_test: Optional[dict] = None

class Coordinator:
    """In-memory coordinator.

    - Maintains a global model.
    - Accepts worker updates per round.
    - Aggregates with FedAvg when enough updates arrive or timeout expires.
    """
    def __init__(self, artifacts_dir: str):
        self.artifacts_dir = artifacts_dir
        Path(self.artifacts_dir).mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._state: Optional[TrainingState] = None

    async def start(self, run_id: str, config: TrainingConfig) -> TrainingState:
        async with self._lock:
            if self._state and self._state.state == "running":
                raise RuntimeError("A training run is already in progress")
            model = CifarCNN()
            self._state = TrainingState(
                run_id=run_id,
                config=config,
                state="running",
                current_round=0,
                num_rounds=config.num_rounds,
                global_state=state_dict_to_cpu(model.state_dict()),
                message="training started",
            )
            log.info("training_started", run_id=run_id, config=config.model_dump())
            return self._state

    async def get_state(self) -> Optional[TrainingState]:
        async with self._lock:
            return self._state

    async def get_global_model_blob(self) -> bytes:
        async with self._lock:
            if not self._state:
                raise RuntimeError("No training run")
            return serialize_state_dict(self._state.global_state)

    async def submit_update(self, upd: UpdateRecord) -> Tuple[bool, str]:
        async with self._lock:
            if not self._state or self._state.state != "running":
                return False, "No active training run"
            if upd.round_idx != self._state.current_round:
                return False, f"Server is on round {self._state.current_round}; update is for {upd.round_idx}"
            bucket = self._state.updates_by_round.setdefault(upd.round_idx, {})
            bucket[upd.worker_id] = upd
            log.info("update_received", run_id=self._state.run_id, round_idx=upd.round_idx,
                     worker_id=upd.worker_id, num_samples=upd.num_samples,
                     train_loss=upd.train_loss, train_accuracy=upd.train_accuracy,
                     wall_time_s=upd.wall_time_s)
            return True, "accepted"

    async def maybe_aggregate(self, active_worker_ids: List[str]) -> Optional[dict]:
        """Aggregate if conditions are met. Returns aggregation payload if aggregated."""
        async with self._lock:
            st = self._state
            if not st or st.state != "running":
                return None

            round_idx = st.current_round
            bucket = st.updates_by_round.get(round_idx, {})
            got = list(bucket.values())

            # Trigger aggregation if enough updates arrived
            enough = len(got) >= max(1, st.config.min_updates_per_round)
            # Or if timeout since round start (approx as since last aggregate or start)
            now = time.time()
            last = st.last_aggregate_time or st.created_at
            timed_out = (now - last) >= st.config.round_timeout_s

            # Also aggregate if all active workers submitted
            all_in = set(active_worker_ids).issubset(set(bucket.keys())) and len(active_worker_ids) > 0

            if not (enough and (all_in or timed_out)):
                return None

            # Aggregate
            states = [deserialize_state_dict(u.state_blob) for u in got]
            weights = [max(1, u.num_samples) for u in got]
            avg = federated_average(states, weights)

            # Update global model
            st.global_state.update(avg)
            st.last_aggregate_time = now

            payload = {
                "run_id": st.run_id,
                "round_idx": round_idx,
                "num_updates": len(got),
                "workers": [
                    {
                        "worker_id": u.worker_id,
                        "num_samples": u.num_samples,
                        "train_loss": u.train_loss,
                        "train_accuracy": u.train_accuracy,
                        "wall_time_s": u.wall_time_s,
                    }
                    for u in got
                ],
                "aggregated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            }
            st.history.append(payload)
            log.info("round_aggregated", **payload)

            # Advance round / finish
            st.current_round += 1
            if st.current_round >= st.num_rounds:
                st.state = "completed"
                st.message = "training completed (final eval may still be pending)"
                log.info("training_completed", run_id=st.run_id)
            else:
                st.message = f"moved to round {st.current_round}"
            return payload

    async def attach_final_test(self, results: dict) -> None:
        async with self._lock:
            if not self._state:
                return
            self._state.final_test = results
            if self._state.state == "completed":
                self._state.message = "training completed"

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List

# ─── GPU / Worker ────────────────────────────────────────────────────

class GPUInfo(BaseModel):
    index: int
    name: str
    total_memory_mb: int
    uuid: Optional[str] = None

class WorkerRegisterRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=80)
    tailscale_ip: str
    gpu_count: int = Field(..., ge=0, le=16)
    gpus: List[GPUInfo] = Field(default_factory=list)
    cpu_count: int = 0
    ram_total_mb: float = 0

class WorkerRegisterResponse(BaseModel):
    worker_id: str
    access_token: str

class HeartbeatRequest(BaseModel):
    worker_id: str
    gpu_utilization: Optional[Dict[str, Any]] = None
    extra: Optional[Dict[str, Any]] = None

class WorkerListEntry(BaseModel):
    worker_id: str
    name: str
    tailscale_ip: str
    gpu_count: int
    gpus: Any = None
    is_active: bool = True
    last_seen: Optional[str] = None

# ─── Training ────────────────────────────────────────────────────────

class TrainingConfig(BaseModel):
    dataset: str = "cifar10"
    num_rounds: int = 20
    local_epochs: int = 1
    batch_size: int = 64
    learning_rate: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 0.0001
    optimizer: str = "sgd"
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    round_timeout_s: int = 900
    min_updates_per_round: int = 1

class TrainingStartRequest(BaseModel):
    config: TrainingConfig

class TrainingStatus(BaseModel):
    run_id: Optional[str] = None
    state: str  # idle | running | completed | failed
    current_round: int = 0
    num_rounds: int = 0
    active_workers: int = 0
    last_aggregate_time: Optional[str] = None
    message: Optional[str] = None

class DataAssignment(BaseModel):
    run_id: str
    worker_id: str
    num_samples: int
    indices_path: str

class SubmitUpdateRequest(BaseModel):
    run_id: str
    round_idx: int
    worker_id: str
    num_samples: int
    train_loss: float
    train_accuracy: float
    wall_time_s: float
    power_consumed_wh: Optional[float] = None

class SubmitUpdateResponse(BaseModel):
    accepted: bool
    message: str

class TrainingResults(BaseModel):
    run_id: str
    state: str
    final_test_accuracy: Optional[float] = None
    final_test_loss: Optional[float] = None
    total_rounds_completed: int = 0
    total_training_time_sec: Optional[float] = None
    history: List[Dict[str, Any]] = Field(default_factory=list)

# ─── Health ──────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str = "0.1.0-poc"
    uptime_sec: float = 0
    active_workers: int = 0
    active_training: Optional[str] = None

from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class WorkerConfig:
    server_url: str
    name: str
    tailscale_ip: str
    heartbeat_interval_s: int = int(os.getenv("LYDIAN_HEARTBEAT_S", "30"))
    retry_attempts: int = int(os.getenv("LYDIAN_RETRY_ATTEMPTS", "5"))
    retry_backoff_base: float = float(os.getenv("LYDIAN_RETRY_BACKOFF_BASE", "2"))
    data_dir: str = os.getenv("LYDIAN_DATA_DIR", "./data")

from __future__ import annotations
from dataclasses import dataclass
import os

def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).strip().lower() in {"1", "true", "yes", "y", "on"}

@dataclass(frozen=True)
class WorkerConfig:
    server_url: str
    name: str
    tailscale_ip: str
    heartbeat_interval_s: int = int(os.getenv("LYDIAN_HEARTBEAT_S", "30"))
    retry_attempts: int = int(os.getenv("LYDIAN_RETRY_ATTEMPTS", "5"))
    retry_backoff_base: float = float(os.getenv("LYDIAN_RETRY_BACKOFF_BASE", "2"))
    data_dir: str = os.getenv("LYDIAN_DATA_DIR", "./data")

    # Manual switches:
    # - Set to 1 when running an older PyTorch that doesn't support newer torch.load args.
    legacy_torch: bool = _env_bool("LYDIAN_TORCH_LEGACY", "1")

    # - Force CPU even if GPUs exist (useful when GPU arch isn't supported by installed PyTorch).
    force_cpu: bool = _env_bool("LYDIAN_FORCE_CPU", "0")

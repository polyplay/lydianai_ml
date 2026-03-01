from __future__ import annotations
from dataclasses import dataclass
import os

@dataclass(frozen=True)
class ServerConfig:
    host: str = os.getenv("LYDIAN_HOST", "0.0.0.0")
    port: int = int(os.getenv("LYDIAN_PORT", "8000"))
    jwt_secret: str = os.getenv("LYDIAN_JWT_SECRET", "change-in-production")
    db_url: str = os.getenv("LYDIAN_DB_URL", "sqlite+aiosqlite:///./lydian.db")
    artifacts_dir: str = os.getenv("LYDIAN_ARTIFACTS_DIR", "./artifacts")
    log_level: str = os.getenv("LYDIAN_LOG_LEVEL", "INFO")

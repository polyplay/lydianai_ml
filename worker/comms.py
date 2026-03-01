from __future__ import annotations
import asyncio
import math
from typing import Optional, Dict, Any

import httpx
import structlog

from common.schemas import (
    WorkerRegisterRequest, WorkerRegisterResponse, TrainingConfig, DataAssignment,
    SubmitUpdateRequest, SubmitUpdateResponse
)

log = structlog.get_logger()

class ServerClient:
    def __init__(self, server_url: str, timeout_s: int = 20):
        self.server_url = server_url.rstrip("/")
        self.timeout_s = timeout_s
        self._token: Optional[str] = None
        self._worker_id: Optional[str] = None

    @property
    def worker_id(self) -> str:
        assert self._worker_id is not None
        return self._worker_id

    def _headers(self) -> Dict[str, str]:
        h = {}
        if self._token:
            h["Authorization"] = f"Bearer {self._token}"
        return h

    async def register(self, req: WorkerRegisterRequest) -> WorkerRegisterResponse:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.server_url}/api/v1/workers/register", json=req.model_dump())
            r.raise_for_status()
            resp = WorkerRegisterResponse(**r.json())
        self._token = resp.access_token
        self._worker_id = resp.worker_id
        return resp

    async def heartbeat(self, payload: dict) -> None:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.post(f"{self.server_url}/api/v1/workers/heartbeat", json=payload, headers=self._headers())
            r.raise_for_status()

    async def get_run(self) -> dict:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.server_url}/api/v1/training/run", headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def get_config(self) -> TrainingConfig:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.server_url}/api/v1/training/config", headers=self._headers())
            r.raise_for_status()
            return TrainingConfig(**r.json())

    async def get_model_blob(self) -> bytes:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.get(f"{self.server_url}/api/v1/training/model", headers=self._headers())
            r.raise_for_status()
            return r.content

    async def get_data_assignment(self) -> DataAssignment:
        async with httpx.AsyncClient(timeout=self.timeout_s) as client:
            r = await client.get(f"{self.server_url}/api/v1/training/data-assignment", headers=self._headers())
            r.raise_for_status()
            return DataAssignment(**r.json())

    async def download_indices(self, indices_url: str) -> bytes:
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.get(f"{self.server_url}{indices_url}", headers=self._headers())
            r.raise_for_status()
            return r.content

    async def submit_update(self, meta: SubmitUpdateRequest, blob: bytes) -> SubmitUpdateResponse:
        files = {"update_file": ("state.pt", blob, "application/octet-stream")}
        async with httpx.AsyncClient(timeout=None) as client:
            r = await client.post(
                f"{self.server_url}/api/v1/training/submit-update",
                data=meta.model_dump(),
                files=files,
                headers=self._headers(),
            )
            r.raise_for_status()
            return SubmitUpdateResponse(**r.json())

    async def with_retries(self, coro_fn, *args, attempts: int = 5, backoff_base: float = 2.0, **kwargs):
        last = None
        for i in range(attempts):
            try:
                return await coro_fn(*args, **kwargs)
            except Exception as e:
                last = e
                sleep_s = (backoff_base ** i) + (0.1 * i)
                log.warn("request_failed_retrying", attempt=i+1, attempts=attempts, sleep_s=sleep_s, error=str(e))
                await asyncio.sleep(sleep_s)
        raise last  # type: ignore

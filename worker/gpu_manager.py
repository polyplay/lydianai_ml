from __future__ import annotations
from typing import List, Optional, Dict, Any

import structlog

log = structlog.get_logger()

def get_gpu_inventory() -> tuple[int, list[dict]]:
    """Return (gpu_count, list_of_gpu_dicts).

    Tries pynvml; falls back to torch.cuda.
    """
    gpus = []
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(h).decode("utf-8", errors="ignore")
            mem = pynvml.nvmlDeviceGetMemoryInfo(h).total // (1024 * 1024)
            uuid = pynvml.nvmlDeviceGetUUID(h).decode("utf-8", errors="ignore")
            gpus.append({"index": i, "name": name, "total_memory_mb": int(mem), "uuid": uuid})
        return int(n), gpus
    except Exception as e:
        log.warn("pynvml_unavailable", error=str(e))
        try:
            import torch
            n = torch.cuda.device_count()
            for i in range(n):
                name = torch.cuda.get_device_name(i)
                # torch doesn't expose total mem reliably without extra calls
                gpus.append({"index": i, "name": name, "total_memory_mb": 0, "uuid": None})
            return int(n), gpus
        except Exception as e2:
            log.warn("cuda_unavailable", error=str(e2))
            return 0, []

def get_gpu_utilization_snapshot() -> Optional[Dict[str, Any]]:
    """Best-effort snapshot of utilization/power if NVML exists."""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out = {"gpus": []}
        for i in range(n):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            power_mw = None
            try:
                power_mw = pynvml.nvmlDeviceGetPowerUsage(h)
            except Exception:
                pass
            out["gpus"].append({
                "index": i,
                "util_gpu": int(util.gpu),
                "util_mem": int(util.memory),
                "power_w": (float(power_mw) / 1000.0) if power_mw is not None else None,
            })
        return out
    except Exception:
        return None

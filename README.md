# LydianAI Distributed Training PoC (FedAvg + FastAPI)

Federated training across **heterogeneous hardware** (CPU-only server + mixed GPU workers, including **legacy Pascal GPUs** like GTX 1080 / 1080 Ti).

## What this is

- **Server/Coordinator (macOS, CPU-only)**: FastAPI + FedAvg aggregation + CIFAR‑10 sharding + metrics
- **Workers (Ubuntu, GPU/CPU)**: register → poll → download model → local train → submit update → repeat
- **Dataset**: CIFAR‑10 (50K train / 10K test)
- **Training**: FedAvg rounds, local SGD, proportional data assignment (more GPUs → bigger shard)

## Networking (Tailscale)

All machines must be on the same Tailscale tailnet.

```bash
sudo tailscale up
tailscale ip -4   # shows your 100.x.x.x address
```

---

# Choose your worker setup: NEW vs LEGACY GPUs

PyTorch CUDA wheels have dropped support for older NVIDIA architectures over time.
**Pascal (sm_61)** GPUs (e.g. **GTX 1080**) require an older PyTorch build *and* (usually) an older Python version.

## Decision table

| Worker GPU | Typical compute capability | Recommended option |
|---|---:|---|
| RTX 20xx / 30xx / 40xx, A100/H100, etc. | sm_75+ | **NEW GPU** |
| GTX 1080 / 1080 Ti (Pascal) | sm_61 | **LEGACY GPU** |
| No CUDA GPU (CPU-only) | — | NEW or LEGACY (both work), or force CPU |

---

# Option A — NEW GPUs (modern CUDA wheels)

### Requirements file
Use: **`requirements_new_gpu.txt`**

This option assumes:
- modern GPU (sm_70+)
- Python 3.12+ recommended (3.13 also OK)

## 1) Create venv (Ubuntu worker / macOS server)

**Ubuntu (worker):**
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements_new_gpu.txt
```

**macOS (server):**
```bash
python3 -m venv venv
source venv/bin/activate
pip install -U pip
pip install -r requirements_new_gpu.txt
```

## 2) Start the server (macOS)

```bash
source venv/bin/activate
python -m server.main --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://<server-tailscale-ip>:8000/api/v1/health
```

## 3) Start a worker (Ubuntu)

```bash
source venv/bin/activate
python -m worker.main   --server http://<server-tailscale-ip>:8000   --name worker-pc1   --tailscale-ip <this-worker-tailscale-ip>
```

## 4) Start training (any machine)

```bash
python -m client.submit_job --server http://<server-ip>:8000 start   --rounds 20 --local-epochs 1 --batch-size 64 --lr 0.01
```

## 5) Monitor & results

```bash
python -m client.submit_job --server http://<server-ip>:8000 monitor
python -m client.submit_job --server http://<server-ip>:8000 status
python -m client.submit_job --server http://<server-ip>:8000 results
python -m client.submit_job --server http://<server-ip>:8000 workers
python -m client.submit_job --server http://<server-ip>:8000 health
```

---

# Option B — LEGACY GPUs (Pascal sm_61: GTX 1080 / 1080 Ti)

If you see errors like:
```
sm_61 is not compatible with the current PyTorch installation
```
…you must use **older PyTorch wheels** that still include Pascal kernels.

### Requirements file
Use: **`requirements_legacy_gpu.txt`**

This option assumes:
- Pascal GPU (sm_61)
- **Python 3.10 recommended** (critical: older CUDA wheels often do **not** exist for Python 3.12+)

## 0) Install Python 3.10 (Ubuntu)

Verify you have it:
```bash
python3.10 --version
```

## 1) Create venv (Ubuntu legacy worker)

```bash
python3.10 -m venv venv
source venv/bin/activate
pip install -U pip
```

## 2) Install legacy PyTorch + torchvision

Pinned combination known to work with Pascal + CUDA 11.7:
- torch **1.13.1+cu117**
- torchvision **0.14.1+cu117**
- numpy pinned to **<2** to avoid binary ABI issues with torchvision ops

```bash
pip install -r requirements_legacy_gpu.txt
```

## 3) Verify CUDA works

```bash
python - <<'PY'
import numpy as np
import torch, torchvision
print("numpy:", np.__version__)
print("torch:", torch.__version__)
print("torchvision:", torchvision.__version__)
print("cuda available:", torch.cuda.is_available())
print("cuda version:", torch.version.cuda)
print("device count:", torch.cuda.device_count())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    torch.zeros(1).cuda()
    print("tensor cuda ok")
PY
```

## 4) Start the legacy worker

The worker supports older torch versions via:
- env var: `LYDIAN_TORCH_LEGACY=1`
- or CLI: `--legacy-torch`

```bash
source venv/bin/activate
export LYDIAN_TORCH_LEGACY=1
python -m worker.main   --server http://<server-tailscale-ip>:8000   --name worker-gtx1080   --tailscale-ip <this-worker-tailscale-ip>
```

### Optional: force CPU

```bash
export LYDIAN_FORCE_CPU=1
python -m worker.main   --server http://<server-tailscale-ip>:8000   --name worker-cpu   --tailscale-ip <this-worker-tailscale-ip>
```

---

# Notes on GPU telemetry (NVML)

PyTorch may warn that `pynvml` is deprecated. The maintained package is **`nvidia-ml-py`**.
This project uses NVML only for **best-effort GPU detection**; NVML failures should not crash the worker.

If you see something like:
```json
{"event": "pynvml_unavailable", "error": "'str' object has no attribute 'decode'"}
```
…it means NVML bindings on that machine are flaky/mismatched. It’s annoying, but non-fatal if `torch.cuda.is_available()` is true.

---

# Troubleshooting (quick hits)

## Torch/torchvision + NumPy ABI warning
If you see:
```
A module that was compiled using NumPy 1.x cannot be run in NumPy 2.x
```
Pin NumPy to 1.26.x (already done in both requirements files here).

## Worker registers but uses 0 GPUs
Either:
- CUDA not available in torch (`torch.cuda.is_available()==False`), or
- you forced CPU with `LYDIAN_FORCE_CPU=1` / `--force-cpu`


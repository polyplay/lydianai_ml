# LydianAI Distributed Training (FedAvg + FastAPI)

System architecture:

- **MacBook**: CPU-only **server/coordinator** (FastAPI) doing FedAvg aggregation.
- **Ubuntu PC #1**: **worker** with 1 GPU.
- **Ubuntu PC #2**: **worker** with 2 GPUs (uses DataParallel locally in this PoC).
- **Dataset**: CIFAR-10
- **Training**: Federated rounds (FedAvg), local SGD per worker.

## 0) Network: Tailscale
Ensure all machines are on the same Tailscale tailnet (encrypted WireGuard mesh).

## 1) Install deps

### Server (Mac)
```bash
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Workers (Ubuntu)
Install a CUDA-enabled PyTorch build, e.g.:
```bash
python3.12 -m venv venv
source venv/bin/activate
# Choose the correct CUDA index-url for your CUDA version, example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
```

Verify GPUs:
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.device_count())"
```

## 2) Start the server (MacBook)

```bash
source venv/bin/activate
python -m server.main --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://<server-ip>:8000/health
```

## 3) Start workers (Ubuntu PCs)

On each worker:
```bash
source venv/bin/activate
python -m worker.main --server http://<server-tailscale-ip>:8000 --name worker-pc1 --tailscale-ip <this-worker-tailscale-ip>
```

Use different `--name` values (e.g. worker-pc1 / worker-pc2).

## 4) Start a training run (any machine)

```bash
source venv/bin/activate
python -m client.submit_job --server http://<server-tailscale-ip>:8000 --rounds 20 --local-epochs 1 --batch-size 64 --lr 0.01
```

The client will:
- start a run
- poll `/training/status`
- fetch `/training/results` once completed.

## Notes
- **Local multi-GPU** on a 2-GPU worker uses `torch.nn.DataParallel` for simplicity.
- Server orchestration is **in-memory** (Coordinator) with minimal SQLite persistence for workers/runs/round metrics.
- Authentication is **JWT (shared secret)**, and transport security is assumed via **Tailscale**.

## Project layout
- `server/` FastAPI server + coordinator + FedAvg aggregation
- `worker/` worker agent (register/heartbeat/poll/train/submit)
- `client/` minimal CLI to start and monitor training
- `common/` shared model + schemas

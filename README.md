# LydianAI Distributed Training PoC (FedAvg + FastAPI)

End-to-end federated ML training across heterogeneous GPU hardware.

- **MacBook**: CPU-only **server/coordinator** (FastAPI) — FedAvg aggregation, CIFAR-10 partitioning
- **Ubuntu PC #1**: **worker** with 1 GPU
- **Ubuntu PC #2**: **worker** with 2 GPUs (DataParallel locally)
- **Dataset**: CIFAR-10 (50K train / 10K test)
- **Training**: Federated rounds (FedAvg), local SGD, proportional data sharding by GPU count

## Network: Tailscale

All machines must be on the same Tailscale tailnet (WireGuard-encrypted mesh VPN).

```bash
# Install: https://tailscale.com/download
sudo tailscale up
tailscale ip -4   # shows your 100.x.x.x address
```

## Quick Start

### 1. Setup

**Server (macOS):**
```bash
./scripts/setup_macos_server.sh
```

**Workers (Ubuntu):**
```bash
./scripts/setup_ubuntu_worker.sh
```

### 2. Start the server (MacBook)

```bash
source venv/bin/activate
python -m server.main --host 0.0.0.0 --port 8000
```

Health check:
```bash
curl http://<server-tailscale-ip>:8000/api/v1/health
```

### 3. Start workers (Ubuntu PCs)

```bash
source venv/bin/activate
python -m worker.main \
  --server http://<server-tailscale-ip>:8000 \
  --name worker-pc1 \
  --tailscale-ip <this-machine-tailscale-ip>
```

Use different `--name` values per machine (e.g. `worker-pc1`, `worker-pc2`).

### 4. Start training (any machine)

```bash
source venv/bin/activate
python -m client.submit_job start \
  --server http://<server-tailscale-ip>:8000 \
  --rounds 20 --local-epochs 1 --batch-size 64 --lr 0.01
```

### 5. Monitor & results

```bash
# Live monitoring
python -m client.submit_job monitor --server http://<server-ip>:8000

# Check status
python -m client.submit_job status --server http://<server-ip>:8000

# Final results
python -m client.submit_job results --server http://<server-ip>:8000

# List workers
python -m client.submit_job workers --server http://<server-ip>:8000
```

## Client CLI Commands

| Command   | Description                            |
|-----------|----------------------------------------|
| `health`  | Check server health, uptime, workers   |
| `start`   | Start a new federated training session |
| `status`  | Check current training progress        |
| `monitor` | Continuously poll and display progress |
| `results` | Fetch final accuracy and loss          |
| `workers` | List all registered workers            |

## API Endpoints

| Method | Path                              | Auth     | Description                    |
|--------|-----------------------------------|----------|--------------------------------|
| GET    | `/api/v1/health`                  | —        | Health check with uptime/stats |
| POST   | `/api/v1/workers/register`        | —        | Register a worker, get JWT     |
| POST   | `/api/v1/workers/heartbeat`       | Bearer   | Worker heartbeat               |
| GET    | `/api/v1/workers`                 | —        | List all workers               |
| POST   | `/api/v1/training/start`          | —        | Start training session         |
| GET    | `/api/v1/training/status`         | —        | Training progress              |
| GET    | `/api/v1/training/config`         | Bearer   | Training config for workers    |
| GET    | `/api/v1/training/run`            | Bearer   | Current run state for workers  |
| GET    | `/api/v1/training/model`          | Bearer   | Download global model weights  |
| GET    | `/api/v1/training/data-assignment`| Bearer   | Worker's data shard indices    |
| POST   | `/api/v1/training/submit-update`  | Bearer   | Submit trained weights         |
| GET    | `/api/v1/training/results`        | —        | Final training results         |

## Architecture

```
Tailscale VPN (100.x.x.x mesh, WireGuard encrypted)
├─ MacBook (Server)
│  ├─ FastAPI + Uvicorn
│  ├─ Coordinator: FedAvg aggregation, round management
│  ├─ Data Manager: CIFAR-10 proportional partitioning
│  ├─ SQLite: worker registry, run metadata, round metrics
│  └─ Evaluator: global model test-set evaluation
├─ Ubuntu PC #1 (1 GPU)
│  ├─ Worker Agent: register → poll → train → submit → repeat
│  └─ Single-GPU training on assigned data shard
└─ Ubuntu PC #2 (2 GPUs)
   ├─ Worker Agent: same lifecycle
   └─ DataParallel across both GPUs on larger data shard
```

## Project Layout

```
lydian-poc/
├── server/           # FastAPI server + coordinator + FedAvg
│   ├── main.py       # FastAPI app, routes, aggregation watchdog
│   ├── coordinator.py# In-memory training state, FedAvg logic
│   ├── data_manager.py # CIFAR-10 partitioning
│   ├── eval.py       # Global model evaluation
│   ├── auth.py       # JWT authentication
│   ├── db.py         # SQLAlchemy models (SQLite)
│   └── config.py     # Server configuration
├── worker/           # Worker agent
│   ├── main.py       # Registration, heartbeat, training loop
│   ├── trainer.py    # Local SGD training (single + DataParallel)
│   ├── comms.py      # HTTP client with retry/backoff
│   ├── gpu_manager.py# GPU detection (pynvml + torch.cuda)
│   └── config.py     # Worker configuration
├── client/           # CLI client
│   └── submit_job.py # start/status/monitor/results/workers/health
├── common/           # Shared code
│   ├── model.py      # CifarCNN model definition
│   ├── schemas.py    # Pydantic request/response models
│   └── constants.py  # Shared constants
├── scripts/          # Setup scripts
│   ├── setup_macos_server.sh
│   └── setup_ubuntu_worker.sh
├── config.yaml       # Reference configuration
└── requirements.txt  # Python dependencies
```

## Key Design Decisions

- **FedAvg over DDP**: WAN-tolerant (sync once per round, not per batch), supports heterogeneous GPUs
- **DataParallel for local multi-GPU**: Simpler than DDP; swap to DDP with `mp.spawn` later
- **Tailscale for transport security**: WireGuard encryption always-on, no manual TLS cert management
- **JWT authentication**: Shared-secret tokens; upgrade to mutual TLS + OAuth 2.0 for production
- **SQLite**: Zero-config persistence for worker registry and round metrics
- **Background aggregation watchdog**: Ensures timeout-based aggregation even if a worker crashes mid-round

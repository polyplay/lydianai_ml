from __future__ import annotations
import io
import time
from typing import List, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

import structlog

from common.model import CifarCNN, state_dict_to_cpu

log = structlog.get_logger()


def _sanitize_state_dict(state_dict: dict) -> dict:
    """Strip 'module.' prefix from DataParallel-wrapped state dicts."""
    cleaned = {}
    for k, v in state_dict.items():
        new_key = k.replace("module.", "") if k.startswith("module.") else k
        cleaned[new_key] = v
    return cleaned


def deserialize_state_dict(blob: bytes) -> dict:
    buf = io.BytesIO(blob)
    return torch.load(buf, map_location="cpu", weights_only=True)

def serialize_state_dict(state_dict: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state_dict_to_cpu(state_dict), buf)
    return buf.getvalue()

def load_cifar10_subset(indices: List[int], data_dir: str, batch_size: int, device: str) -> DataLoader:
    tfm = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    ds = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tfm)
    subset = Subset(ds, indices)
    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=device.startswith("cuda"),
    )
    return loader

def train_one_round(
    model_blob: bytes,
    indices: List[int],
    data_dir: str,
    local_epochs: int,
    batch_size: int,
    lr: float,
    momentum: float,
    device: str,
    weight_decay: float = 0.0001,
    use_data_parallel: bool = True,
) -> Tuple[bytes, float, float, float]:
    """Train locally for E epochs and return new state dict blob + (loss, acc, wall_time)."""
    t0 = time.time()
    model = CifarCNN()

    # BUG-3 FIX: strict=True with key sanitization to catch mismatches
    raw_state = deserialize_state_dict(model_blob)
    clean_state = _sanitize_state_dict(raw_state)
    model.load_state_dict(clean_state, strict=True)

    if device.startswith("cuda"):
        model = model.to(device)
        if use_data_parallel and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
            log.info("train.using_data_parallel", gpu_count=torch.cuda.device_count())

    loader = load_cifar10_subset(indices, data_dir, batch_size, device)
    crit = nn.CrossEntropyLoss()

    # ISSUE-14 FIX: include weight_decay
    opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum,
                          weight_decay=weight_decay)

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for epoch in range(local_epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            loss = crit(logits, y)
            loss.backward()
            opt.step()

            batch_n = x.size(0)
            epoch_loss += float(loss.item()) * batch_n
            epoch_correct += int((logits.argmax(dim=1) == y).sum().item())
            epoch_total += batch_n

        total_loss += epoch_loss
        correct += epoch_correct
        total += epoch_total

        log.info("train.epoch",
                 epoch=epoch + 1, total_epochs=local_epochs,
                 loss=round(epoch_loss / max(1, epoch_total), 4),
                 accuracy=round(epoch_correct / max(1, epoch_total), 4),
                 samples=epoch_total)

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    wall = time.time() - t0

    # Unwrap DataParallel
    if hasattr(model, "module"):
        sd = model.module.state_dict()
    else:
        sd = model.state_dict()

    return serialize_state_dict(sd), avg_loss, acc, wall

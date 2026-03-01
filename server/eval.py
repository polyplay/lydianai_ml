from __future__ import annotations
import time
from typing import Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.model import CifarCNN

def evaluate_cifar10(state_dict: dict, batch_size: int = 256, device: str = "cpu") -> Tuple[float, float]:
    model = CifarCNN()
    # Sanitize keys (strip DataParallel 'module.' prefix if present)
    clean = {k.replace("module.", "") if k.startswith("module.") else k: v
             for k, v in state_dict.items()}
    model.load_state_dict(clean, strict=True)
    model.to(device)
    model.eval()

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_ds = datasets.CIFAR10(root="./data", train=False, download=True, transform=tfm)
    loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=(device.startswith("cuda")))

    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            logits = model(x)
            loss = crit(logits, y)
            total_loss += float(loss.item()) * x.size(0)
            pred = logits.argmax(dim=1)
            correct += int((pred == y).sum().item())
            total += int(x.size(0))

    avg_loss = total_loss / max(1, total)
    acc = correct / max(1, total)
    return avg_loss, acc

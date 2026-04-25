"""Data, sparsity helpers, plotting."""
from __future__ import annotations

import os
import random
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from model import SelfPruningNet


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def set_seed(seed: int = 42) -> None:
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_cifar10_loaders(data_dir: str = "./data", batch_size: int = 128,
                       num_workers: int = 2) -> Tuple[DataLoader, DataLoader]:
    os.makedirs(data_dir, exist_ok=True)
    norm = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([transforms.ToTensor(), norm])

    train_set = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=train_tf
    )
    test_set = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=test_tf
    )

    return (
        DataLoader(train_set, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, pin_memory=True, drop_last=True),
        DataLoader(test_set, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers, pin_memory=True),
    )


@torch.no_grad()
def collect_gate_values(model: SelfPruningNet) -> torch.Tensor:
    return torch.cat([
        l.gates().flatten().detach().cpu()
        for l in model.prunable_layers()
    ])


@torch.no_grad()
def compute_sparsity(model: SelfPruningNet, threshold: float = 1e-2) -> float:
    g = collect_gate_values(model)
    return (g < threshold).float().mean().item() if g.numel() else 0.0


@torch.no_grad()
def evaluate(model: SelfPruningNet, loader: DataLoader, device) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        correct += (model(x).argmax(1) == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


def plot_gate_distribution(model: SelfPruningNet, out_path: str,
                           title: str = "Gate value distribution",
                           bins: int = 60, threshold: float = 1e-2) -> None:
    gates = collect_gate_values(model).numpy()

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(gates, bins=bins, range=(0.0, 1.0),
            color="#3b6fb5", edgecolor="white")

    ax.axvline(threshold, color="crimson", linestyle="--",
               label=f"prune threshold = {threshold}")

    ax.set_xlabel("sigmoid(gate_score)")
    ax.set_ylabel("count")
    ax.set_title(title)
    ax.legend()

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
"""Train the Self-Pruning Net on CIFAR-10."""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, asdict
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import SelfPruningNet
from utils import (compute_sparsity, evaluate, get_cifar10_loaders,
                   get_device, plot_gate_distribution, set_seed)


@dataclass
class TrainConfig:
    epochs: int = 25
    batch_size: int = 128
    lr: float = 1e-3
    lambda_sparsity: float = 1e-4
    prune_threshold: float = 5e-2
    seed: int = 42
    data_dir: str = "./data"
    num_workers: int = 2
    out_dir: str = "./outputs"


def train_one_epoch(model, loader, optimizer, criterion, lam, device, epoch):
    model.train()
    cls_sum = spa_sum = 0.0
    correct = total = n = 0
    t0 = time.time()
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        cls = criterion(logits, y)
        spa = model.sparsity_loss()
        loss = cls + lam * spa

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        cls_sum += cls.item(); spa_sum += spa.item(); n += 1
        with torch.no_grad():
            correct += (logits.argmax(1) == y).sum().item(); total += y.size(0)
    return {
        "epoch": epoch,
        "cls_loss": cls_sum / n,
        "spa_loss": spa_sum / n,
        "train_acc": correct / total,
        "time_sec": time.time() - t0,
    }


def run_training(cfg: TrainConfig) -> dict:
    set_seed(cfg.seed)
    device = get_device()
    print(f"[device] {device}")

    train_loader, test_loader = get_cifar10_loaders(
        cfg.data_dir, cfg.batch_size, cfg.num_workers)
    model = SelfPruningNet().to(device)
    print(f"[model] gates = {model.num_gates():,}")

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, cfg.epochs + 1):
        s = train_one_epoch(model, train_loader, optimizer, criterion,
                            cfg.lambda_sparsity, device, epoch)
        acc = evaluate(model, test_loader, device)
        spa = compute_sparsity(model, cfg.prune_threshold)
        print(f"λ={cfg.lambda_sparsity:.0e} | epoch {epoch:3d} | "
              f"cls {s['cls_loss']:.4f} | spa {s['spa_loss']:.0f} | "
              f"train {s['train_acc']*100:5.2f}% | test {acc*100:5.2f}% | "
              f"sparsity {spa*100:5.2f}% | {s['time_sec']:.1f}s")

    final_acc = evaluate(model, test_loader, device)
    final_spa = compute_sparsity(model, cfg.prune_threshold)

    os.makedirs(cfg.out_dir, exist_ok=True)
    tag = f"lambda_{cfg.lambda_sparsity:.0e}"
    plot_path = os.path.join(cfg.out_dir, f"gate_dist_{tag}.png")
    ckpt_path = os.path.join(cfg.out_dir, f"model_{tag}.pt")
    plot_gate_distribution(model, plot_path,
                           title=f"Final gate distribution ({tag})",
                           threshold=cfg.prune_threshold)
    torch.save({"state_dict": model.state_dict(), "config": asdict(cfg)},
               ckpt_path)
    print(f"\n[final] λ={cfg.lambda_sparsity:.0e}  "
          f"test_acc={final_acc*100:.2f}%  sparsity={final_spa*100:.2f}%")
    return {"lambda_sparsity": cfg.lambda_sparsity,
            "test_accuracy": final_acc, "sparsity": final_spa,
            "checkpoint": ckpt_path, "plot": plot_path}


def run_sweep(base_cfg: TrainConfig, lambdas: List[float]) -> None:
    results = []
    for lam in lambdas:
        print("\n" + "=" * 70)
        print(f"  Training with lambda = {lam:.0e}")
        print("=" * 70)
        cfg = TrainConfig(**{**asdict(base_cfg), "lambda_sparsity": lam})
        results.append(run_training(cfg))

    print("\n" + "=" * 70 + "\n  Summary\n" + "=" * 70)
    print(f"{'Lambda':>10} | {'Test Acc (%)':>13} | {'Sparsity (%)':>13}")
    print("-" * 44)
    for r in results:
        print(f"{r['lambda_sparsity']:>10.0e} | "
              f"{r['test_accuracy']*100:>13.2f} | "
              f"{r['sparsity']*100:>13.2f}")
    with open(os.path.join(base_cfg.out_dir, "sweep_results.json"), "w") as f:
        json.dump(results, f, indent=2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lambda-sparsity", type=float, default=1e-4)
    p.add_argument("--prune-threshold", type=float, default=1e-2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", type=str, default="./outputs")
    p.add_argument("--sweep", action="store_true")
    p.add_argument("--lambdas", type=float, nargs="+",
                   default=[1e-5, 1e-4, 1e-3])
    return p.parse_args()


def main():
    a = parse_args()
    base = TrainConfig(epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                       lambda_sparsity=a.lambda_sparsity,
                       prune_threshold=a.prune_threshold,
                       seed=a.seed, data_dir=a.data_dir,
                       num_workers=a.num_workers, out_dir=a.out_dir)
    if a.sweep:
        run_sweep(base, a.lambdas)
    else:
        run_training(base)


if __name__ == "__main__":
    main()
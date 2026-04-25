"""Quick sanity-check on a saved model's gate distribution.

Run after training:
    python inspect_gates.py outputs/model_lambda_1e-04.pt
"""
from __future__ import annotations

import sys
import torch

from model import SelfPruningNet
from utils import collect_gate_values


def inspect(ckpt_path: str) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = SelfPruningNet()
    model.load_state_dict(ckpt["state_dict"])

    g = collect_gate_values(model)
    n = g.numel()

    print(f"checkpoint:           {ckpt_path}")
    print(f"total gates:          {n:,}")
    print(f"min / mean / max:     {g.min():.4f} / {g.mean():.4f} / {g.max():.4f}")
    print()

    for thr in (0.01, 0.05, 0.1, 0.5, 0.9):
        below = (g < thr).float().mean().item() * 100
        print(f"  % gates < {thr:<5}:   {below:6.2f} %")

    print()

    pruned = (g < 0.01).float().mean().item() * 100
    kept   = (g > 0.90).float().mean().item() * 100
    middle = 100 - pruned - kept

    print(f"pruned (<0.01):       {pruned:6.2f} %")
    print(f"kept   (>0.90):       {kept:6.2f} %")
    print(f"middle (0.01-0.90):   {middle:6.2f} %")

    if pruned > 30 and kept > 10 and middle < 30:
        print("\n[✓] Distribution looks BIMODAL — good for submission.")
    else:
        print("\n[!] Distribution is NOT clearly bimodal.")
        print("    Try a smaller lambda (e.g. 1e-4) or train longer.")


if __name__ == "__main__":
    inspect(sys.argv[1] if len(sys.argv) > 1 else "outputs/model_lambda_1e-04.pt")
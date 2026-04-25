"""PrunableLinear + SelfPruningNet (CIFAR-10)."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class PrunableLinear(nn.Module):
    """Linear layer where each weight is multiplied by a learnable sigmoid gate.

    Both `weight` and `gate_scores` are nn.Parameters, so the optimizer updates
    them and gradients flow into both.
    """

    def __init__(self, in_features: int, out_features: int,
                 bias: bool = True, gate_init: float = 2.0) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        if not bias:
            self.register_parameter("bias", None)

        # gate_init=2.0 -> sigmoid(2.0) ~= 0.88, so gates start *open*.
        self.gate_scores = nn.Parameter(
            torch.full((out_features, in_features), float(gate_init))
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def gates(self) -> torch.Tensor:
        return torch.sigmoid(self.gate_scores)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pruned_weights = self.weight * self.gates()
        return F.linear(x, pruned_weights, self.bias)


class SelfPruningNet(nn.Module):
    """3072 -> 1024 -> 512 -> 10 MLP, every Linear is a PrunableLinear."""

    def __init__(self, input_dim: int = 3 * 32 * 32,
                 hidden_dims: List[int] | None = None,
                 num_classes: int = 10) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [1024, 512]
        dims = [input_dim, *hidden_dims, num_classes]

        layers: List[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(PrunableLinear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(nn.ReLU(inplace=True))

        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x.flatten(start_dim=1))

    def prunable_layers(self) -> List[PrunableLinear]:
        return [m for m in self.modules() if isinstance(m, PrunableLinear)]

    def sparsity_loss(self) -> torch.Tensor:
        """L1 norm of all sigmoid gates (gates are >=0, so L1 == sum)."""
        total = self.classifier[0].gates().new_zeros(())
        for layer in self.prunable_layers():
            total = total + layer.gates().sum()
        return total

    def num_gates(self) -> int:
        return sum(l.gate_scores.numel() for l in self.prunable_layers())
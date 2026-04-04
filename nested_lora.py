"""
Nested LoRA — One Particle, Multiple Orbitals
===============================================

Single LoRA adapter pair with dynamic rank via slicing.
r4 ⊂ r8 ⊂ r16 — descending pauses dimensions, ascending resumes them.
Zero cold start on transitions.

This module is the "engine" — pure architecture, no control logic.
Pair with OrbitalController for adaptive rank decisions.

Author: Simona Vargiu
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


class NestedLoRALinear(nn.Module):
    """
    Single LoRA adapter with dynamic rank via slicing.

    A single pair of matrices A(max_rank, in) and B(out, max_rank) is shared
    across all rank levels. The active rank is controlled by slicing:

        r=4  → A[:4, :],  B[:, :4]
        r=8  → A[:8, :],  B[:, :8]
        r=16 → A[:16,:],  B[:, :16]

    When descending from r=16 to r=4, dimensions 0-3 retain all learned
    weights. Dimensions 4-15 are paused (no gradient), not destroyed.
    When ascending back, they resume exactly where they left off.

    Output is scaled by max_rank/active_rank to maintain consistent
    magnitude across rank changes (analogous to alpha/r in standard LoRA).

    Args:
        linear: Original nn.Linear layer to wrap
        max_rank: Maximum LoRA rank (default: 16)

    Example:
        >>> layer = NestedLoRALinear(original_linear, max_rank=16)
        >>> layer.set_rank(4)    # use 4 dimensions
        >>> out = layer(x)       # forward with r=4
        >>> layer.set_rank(16)   # expand to full rank
        >>> out = layer(x)       # forward with r=16, dimensions 0-3 unchanged
    """

    def __init__(self, linear: nn.Linear, max_rank: int = 16):
        super().__init__()
        self.linear = linear
        self.max_rank = max_rank
        self.active_rank = max_rank

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False

        # One particle: single A and B
        self.lora_A = nn.Parameter(torch.empty(max_rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, max_rank))

        # Standard LoRA init: A = kaiming, B = zeros → initial delta = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def set_rank(self, r: int):
        """Set the active orbital. Must be <= max_rank."""
        self.active_rank = min(r, self.max_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        r = self.active_rank

        h = F.linear(x, self.lora_A[:r, :])
        delta = F.linear(h, self.lora_B[:, :r])

        scale = self.max_rank / r
        return base + delta * scale


def inject_nested_lora(model: nn.Module, max_rank: int = 16) -> nn.Module:
    """
    Replace attention Linear layers with NestedLoRALinear.

    Targets any nn.Linear whose full name contains "attention".
    Original weights are frozen; only LoRA parameters are trainable.

    Args:
        model: PyTorch model
        max_rank: Maximum LoRA rank

    Returns:
        Model with NestedLoRA injected
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "attention" in name:
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, NestedLoRALinear(module, max_rank))
    return model


def set_rank(model: nn.Module, r: int):
    """Set active rank on all NestedLoRALinear modules in the model."""
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            m.set_rank(r)


def get_lora_params(model: nn.Module) -> List[nn.Parameter]:
    """Get all LoRA parameters (for optimizer setup)."""
    params = []
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            params.extend([m.lora_A, m.lora_B])
    return params


def count_params(model: nn.Module) -> dict:
    """Count total, trainable, and LoRA parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    lora = sum(p.numel() for p in get_lora_params(model))
    return {"total": total, "trainable": trainable, "lora": lora}

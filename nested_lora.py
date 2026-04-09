"""
NestedLoRA — Execution Engine
==============================
LoRA adapter where rank is controlled by matrix slicing, not by
swapping separate adapter pairs.

Architecture:
    A single (max_rank × d) matrix pair is allocated once.
    Active rank is a *slice* of that matrix: r4 ⊂ r8 ⊂ r16.
    Changing rank = changing the slice boundary. Zero re-allocation,
    zero cold-start, because lower-rank parameters are always a
    subset of higher-rank ones.

    Forward:  h = x @ W + (x @ A[:, :r] @ B[:r, :]) * (α / r)

Author: Simona Vargiu
License: Apache 2.0
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Optional


class NestedLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with nested-rank LoRA."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        max_rank: int = 16,
        alpha: float = 16.0,
        bias: bool = True,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.alpha = alpha
        self.active_rank = max_rank  # start at full capacity

        # Frozen pretrained weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA matrices — allocated once at max_rank
        self.lora_A = nn.Parameter(torch.empty(in_features, max_rank))
        self.lora_B = nn.Parameter(torch.zeros(max_rank, out_features))

        # Kaiming init for A, zero init for B (standard LoRA)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def scaling(self) -> float:
        """Standard LoRA scaling: α / active_rank."""
        return self.alpha / self.active_rank if self.active_rank > 0 else 0.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base linear
        h = nn.functional.linear(x, self.weight, self.bias_param)

        # LoRA delta via nested slicing
        if self.active_rank > 0:
            A_slice = self.lora_A[:, : self.active_rank]       # (in, r)
            B_slice = self.lora_B[: self.active_rank, :]       # (r, out)
            h = h + (x @ A_slice @ B_slice) * self.scaling

        return h

    def extra_repr(self) -> str:
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"max_rank={self.max_rank}, active_rank={self.active_rank}, "
            f"α={self.alpha}"
        )


# ── Injection helpers ───────────────────────────────────────────


def inject_nested_lora(
    model: nn.Module,
    target_modules: list[str],
    max_rank: int = 16,
    alpha: float = 16.0,
) -> Dict[str, NestedLoRALinear]:
    """
    Replace target Linear layers with NestedLoRALinear.

    Args:
        model: The base model (weights will be frozen).
        target_modules: List of substrings to match layer names
            (e.g. ["q_proj", "v_proj"]).
        max_rank: Maximum rank allocated per adapter.
        alpha: LoRA scaling factor.

    Returns:
        Dictionary mapping layer name → NestedLoRALinear module.
    """
    adapters: Dict[str, NestedLoRALinear] = {}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        if not any(t in name for t in target_modules):
            continue

        nested = NestedLoRALinear(
            in_features=module.in_features,
            out_features=module.out_features,
            max_rank=max_rank,
            alpha=alpha,
            bias=module.bias is not None,
        )

        # Copy frozen weights
        nested.weight.data.copy_(module.weight.data)
        if module.bias is not None and nested.bias_param is not None:
            nested.bias_param.data.copy_(module.bias.data)

        # Replace in model
        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, nested)
        adapters[name] = nested

    return adapters


def set_rank(adapters: Dict[str, NestedLoRALinear], rank: int) -> None:
    """Set active rank globally across all adapters."""
    for adapter in adapters.values():
        adapter.active_rank = min(rank, adapter.max_rank)


def get_lora_params(adapters: Dict[str, NestedLoRALinear]):
    """Yield only the trainable LoRA parameters."""
    for adapter in adapters.values():
        yield adapter.lora_A
        yield adapter.lora_B


def count_params(adapters: Dict[str, NestedLoRALinear], active_only: bool = True) -> int:
    """Count LoRA parameters (active slice or full allocation)."""
    total = 0
    for adapter in adapters.values():
        r = adapter.active_rank if active_only else adapter.max_rank
        total += adapter.in_features * r  # A
        total += r * adapter.out_features  # B
    return total

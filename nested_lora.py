"""
NestedLoRA — Execution Engine (Weight-Control Aware)
===================================================

LoRA adapter where rank is controlled by matrix slicing, not by
swapping separate adapter pairs.

Architecture
------------
A single pair of matrices is allocated once:

    A ∈ ℝ^(d × R_max)
    B ∈ ℝ^(R_max × d)

Active rank r is a slice of these matrices:

    ΔW = A[:, :r] @ B[:r, :]

with:

    r_small ⊂ r_medium ⊂ r_large

Changing rank = moving the slice boundary.

Properties:
- No re-allocation
- No cold-start
- Strict parameter nesting across ranks

Forward:
    h = x @ W + (x @ A[:, :r] @ B[:r, :]) * (α / r)

--------------------------------------------------

Weight Control Interpretation
-----------------------------

NestedLoRA is not only a memory optimization.

It provides direct control over the update matrix:

    ΔW = A_r @ B_r

where r defines the dimensionality of the update subspace.

Implications:

- Low rank → constrained ΔW (restricted update space)
- High rank → expanded ΔW (higher expressivity, higher variance)

Therefore:

    Rank ≠ capacity only
    Rank = control over ΔW

--------------------------------------------------

Control Perspective
-------------------

In a dynamic setting, rank becomes a control variable:

    controller → sets r → defines ΔW space → shapes weight dynamics

This enables:

- Progressive expansion of learning capacity without reset
- Smooth transitions between regimes (stable ↔ unstable)
- Direct modulation of weight drift

When combined with a controller (e.g. FSM / Orbital LR),
NestedLoRA becomes part of a closed-loop system:

    loss dynamics → controller → rank → ΔW → training stability

--------------------------------------------------

Key Insight
-----------

    Rank is not just capacity.
    Rank is a control mechanism over weight updates.

--------------------------------------------------

Author: Simona Vargiu
License: Apache 2.0
"""

import torch
import torch.nn as nn
import math


class NestedLoRALinear(nn.Module):
    """Drop-in replacement for nn.Linear with nested-rank LoRA.

    Active rank is a slice of a single max-rank allocation.
    Rank transitions are instant, with zero re-allocation and
    zero cold-start degradation.
    """

    def __init__(self, in_features, out_features, max_rank=16, alpha=16.0, bias=True):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.max_rank = max_rank
        self.alpha = alpha
        self.active_rank = max_rank

        # Frozen pretrained weight
        self.weight = nn.Parameter(torch.empty(out_features, in_features), requires_grad=False)
        self.bias_param = nn.Parameter(torch.zeros(out_features)) if bias else None

        # LoRA matrices — allocated once at max_rank
        self.lora_A = nn.Parameter(torch.empty(in_features, max_rank))
        self.lora_B = nn.Parameter(torch.zeros(max_rank, out_features))

        # Kaiming init for A, zero init for B (standard LoRA)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def scaling(self):
        """Standard LoRA scaling: α / active_rank."""
        return self.alpha / self.active_rank if self.active_rank > 0 else 0.0

    def forward(self, x):
        base = nn.functional.linear(x, self.weight, self.bias_param)

        if self.active_rank > 0:
            A = self.lora_A[:, :self.active_rank]
            B = self.lora_B[:self.active_rank, :]
            return base + (x @ A @ B) * self.scaling

        return base

    def extra_repr(self):
        return (
            f"in={self.in_features}, out={self.out_features}, "
            f"max_rank={self.max_rank}, active_rank={self.active_rank}, "
            f"α={self.alpha}"
        )


# ── Injection ───────────────────────────────────────────────────


def inject_nested_lora(model, target_modules, max_rank=16, alpha=16.0):
    """Replace target Linear layers with NestedLoRALinear.

    Args:
        model: Base model (weights will be frozen).
        target_modules: List of substrings to match layer names
            (e.g. ["q_proj", "v_proj"]).
        max_rank: Maximum rank allocated per adapter.
        alpha: LoRA scaling factor.

    Returns:
        Dict mapping layer name → NestedLoRALinear module.
    """
    adapters = {}

    for name, module in list(model.named_modules()):
        if not isinstance(module, nn.Linear):
            continue

        if not any(t in name for t in target_modules):
            continue

        new_layer = NestedLoRALinear(
            module.in_features,
            module.out_features,
            max_rank=max_rank,
            alpha=alpha,
            bias=module.bias is not None,
        )

        new_layer.weight.data.copy_(module.weight.data)

        if module.bias is not None:
            new_layer.bias_param.data.copy_(module.bias.data)

        parent_name, attr_name = name.rsplit(".", 1) if "." in name else ("", name)
        parent = model.get_submodule(parent_name) if parent_name else model
        setattr(parent, attr_name, new_layer)

        adapters[name] = new_layer

    return adapters


# ── Utilities ───────────────────────────────────────────────────


def set_rank(model, rank):
    """Set active rank globally across all NestedLoRALinear layers."""
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            m.active_rank = min(rank, m.max_rank)


def get_lora_params(model):
    """Yield only the trainable LoRA parameters (A and B matrices)."""
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            yield m.lora_A
            yield m.lora_B


def count_params(model, active_only=True):
    """Count LoRA parameters (active slice or full allocation).

    Args:
        model: Model containing NestedLoRALinear layers.
        active_only: If True, count only the active slice.
            If False, count the full max_rank allocation.

    Returns:
        Total parameter count.
    """
    total = 0
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            r = m.active_rank if active_only else m.max_rank
            total += m.in_features * r      # A slice
            total += r * m.out_features      # B slice
    return total

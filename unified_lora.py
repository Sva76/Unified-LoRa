"""
Legacy Adaptive LoRA (Gradient-based)
====================================

Early experimental version of adaptive LoRA using gradient-based rank updates.

This approach adjusts rank per-layer based on gradient norm dynamics.
However, it suffers from instability and does not provide consistent benefits.

Replaced by:
- NestedLoRA (shared orbital architecture)
- OrbitalController (stress-based closed-loop control)

This file is kept for reference only.

Status: deprecated / legacy
"""

Unified-LoRA Controller
========================
Adaptive per-layer rank controller for LoRA fine-tuning.
Drop-in module — works with any model that uses LoRA adapters.

Usage:
    from unified_lora import LoRALinear, get_lora_modules

    # Replace linear layers with adaptive LoRA
    layer.q_proj = LoRALinear(layer.q_proj, max_r=16)

    # In training loop, after loss.backward():
    for m in get_lora_modules(model):
        m.update_rank()
"""

import copy
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """
    LoRA adapter with per-layer adaptive rank.

    The rank adjusts based on gradient stress:
    - Gradient stress increasing → rank goes up (more capacity)
    - Gradient stress decreasing → rank goes down (less capacity)

    Parameters
    ----------
    base : nn.Linear
        The original linear layer to wrap.
    max_r : int
        Maximum rank (default 16).
    min_r : int
        Minimum rank (default 4).
    alpha : float
        Scaling factor for LoRA output. Uses alpha/active_r scaling.
    layer_name : str
        Optional name for logging.
    """

    def __init__(self, base, max_r=16, min_r=4, alpha=16.0, layer_name=""):
        super().__init__()
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False

        self.max_r = max_r
        self.min_r = min_r
        self.alpha = alpha
        self.layer_name = layer_name

        self.A = nn.Parameter(torch.randn(max_r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, max_r))
        self.active_r = min_r

        # Stress tracking
        self.grad_ema = None
        self.prev_grad_ema = None

    def set_rank(self, r):
        self.active_r = max(self.min_r, min(r, self.max_r))

    def update_rank(self):
        """Call after loss.backward(), before optimizer.step()."""
        if self.A.grad is None:
            return

        grad_norm = self.A.grad[:self.active_r].norm().item()

        if self.grad_ema is None:
            self.grad_ema = grad_norm
            self.prev_grad_ema = grad_norm
            return

        self.prev_grad_ema = self.grad_ema
        self.grad_ema = 0.9 * self.grad_ema + 0.1 * grad_norm

        delta = self.grad_ema - self.prev_grad_ema
        threshold = 0.01 * self.grad_ema if self.grad_ema > 0 else 0.01

        if delta > threshold:
            self.active_r = min(self.max_r, self.active_r + 2)
        elif delta < -threshold:
            self.active_r = max(self.min_r, self.active_r - 2)

    def forward(self, x):
        base_out = self.base(x)
        A = self.A[:self.active_r]
        B = self.B[:, :self.active_r]
        lora_out = x @ A.t() @ B.t()
        scale = self.alpha / self.active_r
        return base_out + scale * lora_out

    def extra_repr(self):
        return (f"in={self.base.in_features}, out={self.base.out_features}, "
                f"max_r={self.max_r}, min_r={self.min_r}, alpha={self.alpha}, "
                f"active_r={self.active_r}, name={self.layer_name}")


def get_lora_modules(model):
    """Return all LoRALinear modules in a model."""
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def inject_lora(model, target_modules, max_r=16, min_r=4, alpha=16.0):
    """
    Replace target linear layers with LoRALinear adapters.

    Parameters
    ----------
    model : nn.Module
        The model to modify.
    target_modules : list of str
        Names of linear layers to replace (e.g. ["q_proj", "v_proj"]).
    max_r, min_r, alpha : passed to LoRALinear.

    Returns
    -------
    model : nn.Module
        Modified model with LoRA adapters.

    Example
    -------
    # DistilBERT
    inject_lora(model, ["q_lin", "v_lin"])

    # Llama / Mistral
    inject_lora(model, ["q_proj", "v_proj"])

    # All attention projections
    inject_lora(model, ["q_proj", "k_proj", "v_proj", "o_proj"])
    """
    replace_list = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if any(name.endswith(t) for t in target_modules):
                replace_list.append(name)

    for name in replace_list:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        original = getattr(parent, parts[-1])
        setattr(parent, parts[-1], LoRALinear(
            original, max_r=max_r, min_r=min_r, alpha=alpha, layer_name=name
        ))

    print(f"Injected LoRA into {len(replace_list)} layers: {replace_list}")
    return model


def setup_trainable(model):
    """Freeze base model, unfreeze LoRA params and classifier."""
    for p in model.parameters():
        p.requires_grad = False

    for m in get_lora_modules(model):
        m.A.requires_grad = True
        m.B.requires_grad = True

    # Unfreeze common classifier head names
    for n, p in model.named_parameters():
        if any(k in n for k in ["classifier", "pre_classifier", "score", "lm_head"]):
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return model

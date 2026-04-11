"""
Unified-LoRA Controller
======================

Convenience wrapper that exposes the full Unified-LoRA stack:

- nested_lora.py        → execution engine (LoRA with dynamic rank slicing)
- orbital_controller.py → control logic (dual ΔW control: rank + LR)

Use this module for simple integration, or import submodules directly
for fine-grained control.

Author: Simona Vargiu
License: Apache 2.0
"""

# ── ENGINE ──────────────────────────────────────────
from nested_lora import (
    NestedLoRALinear,
    inject_nested_lora,
    set_rank,
    get_lora_params,
    count_params,
)

# ── CONTROLLER ──────────────────────────────────────
from orbital_controller import (
    OrbitalController,
    setup_unified_lora,
)

# ── EXPORT ──────────────────────────────────────────
__all__ = [
    "NestedLoRALinear",
    "inject_nested_lora",
    "set_rank",
    "get_lora_params",
    "count_params",
    "OrbitalController",
    "setup_unified_lora",
]

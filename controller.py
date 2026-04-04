"""
Unified LoRA Controller
========================

Convenience wrapper that re-exports from the two core modules:
- nested_lora.py          (engine: NestedLoRALinear, inject, set_rank)
- orbital_controller.py   (intelligence: OrbitalController)

Import from here for quick usage, or from the individual modules
for finer control.

Author: Simona Vargiu
License: Apache 2.0
"""

# Engine
from nested_lora import (
    NestedLoRALinear,
    inject_nested_lora,
    set_rank,
    get_lora_params,
    count_params,
)

# Intelligence
from orbital_controller import (
    OrbitalController,
    setup_unified_lora,
)

__all__ = [
    "NestedLoRALinear",
    "inject_nested_lora",
    "set_rank",
    "get_lora_params",
    "count_params",
    "OrbitalController",
    "setup_unified_lora",
]

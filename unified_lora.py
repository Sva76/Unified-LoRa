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



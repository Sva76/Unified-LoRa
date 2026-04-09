"""
NestedLoRA — Execution Engine (Weight-Control Aware)
===================================================

LoRA adapter where rank is controlled by matrix slicing, not by
swapping separate adapter pairs.

Architecture:
    A single (max_rank × d) matrix pair is allocated once.
    Active rank is a *slice* of that matrix: r4 ⊂ r8 ⊂ r16.
    Changing rank = changing the slice boundary. Zero re-allocation,
    zero cold-start, because lower-rank parameters are always a
    subset of higher-rank ones.

    Forward:  h = x @ W + (x @ A[:, :r] @ B[:r, :]) * (α / r)

------------------------------------------------------------------

⚠️ IMPORTANT: Weight Control Interpretation

NestedLoRA is not only a memory-efficient implementation.

It introduces a direct control mechanism over weight updates (ΔW):

    ΔW = A_r @ B_r

Where r (active_rank) defines the dimensionality of the update space.

This means:

- Lower rank → constrained ΔW (limited update subspace)
- Higher rank → expanded ΔW (higher flexibility, higher risk)

Rank is therefore not just a capacity parameter —
it is a control knob over weight dynamics.

This enables:

- Progressive expansion of learning capacity without reset
- Smooth adaptation under changing training conditions
- Direct regulation of weight drift and instability

In combination with a controller (e.g. Orbital LR / FSM),
NestedLoRA becomes a component of a full weight control system:

    controller → adjusts rank → modulates ΔW → stabilizes training

------------------------------------------------------------------

Key Insight:

    Rank is not capacity — it is control over ΔW.

------------------------------------------------------------------

Author: Simona Vargiu
License: Apache 2.0
"""

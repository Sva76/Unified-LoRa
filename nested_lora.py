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

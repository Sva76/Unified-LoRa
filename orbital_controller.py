"""
OrbitalLRController — Weight Dynamics Stabilizer
================================================

Adaptive learning rate controller designed to regulate weight updates (ΔW)
during training by reacting to instability signals in real time.

Unlike static schedulers (cosine, linear), this controller operates as a
closed-loop system:

    loss → instability signal (φ) → LR adjustment → ΔW modulation

------------------------------------------------------------------

⚠️ CORE IDEA: CONTROL OF ΔW (WEIGHT UPDATES)

Learning rate directly scales weight updates:

    ΔW ∝ LR × ∇L

This means:

- High LR → large ΔW → fast learning but unstable
- Low LR  → small ΔW → stable but slow

OrbitalLRController dynamically adjusts LR to keep ΔW within a stable regime.

------------------------------------------------------------------

CONTROL MECHANISM

We define an instability signal φ based on:

- deviation from EMA(loss)
- loss spikes (Δloss)

    φ = |loss - EMA(loss)| + spike_component

The controller maintains two adaptive thresholds:

- stress threshold → instability detected → reduce LR
- stable threshold → recovery detected → increase LR

This creates an orbital behavior:

    HIGH → BASE → LOW → BASE → HIGH

Each transition corresponds to regulating ΔW magnitude.

------------------------------------------------------------------

BEHAVIOR UNDER STRESS

Without control:
    ΔW grows uncontrollably → weight drift → collapse (NaN / divergence)

With Orbital control:
    ΔW is reduced during instability
    → drift is bounded
    → training survives

------------------------------------------------------------------

KEY PROPERTIES

- Closed-loop (feedback-driven)
- No predefined schedule
- Reacts to real training dynamics
- Prevents catastrophic updates
- Enables aggressive training regimes

------------------------------------------------------------------

SYSTEM VIEW

OrbitalLRController is one component of a broader weight control system:

    controller → adjusts LR → modulates ΔW → stabilizes weights

Combined with NestedLoRA:

    rank → defines ΔW subspace
    LR   → defines ΔW magnitude

Together:

    control over direction + control over magnitude = full ΔW control

------------------------------------------------------------------

Key Insight:

    Training stability is a function of controlled ΔW,
    not just optimized loss.

------------------------------------------------------------------

Author: Simona Vargiu
License: Apache 2.0
"""

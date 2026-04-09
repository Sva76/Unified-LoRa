# Unified-LoRA Architecture

## The Cold-Start Problem

Standard adaptive LoRA approaches (including our earlier per-layer controller) use **independent adapter pairs** for each rank: a dedicated (A₄, B₄) for rank 4, a separate (A₈, B₈) for rank 8, etc. When the controller decides to switch from rank 4 to rank 8, it activates a fresh adapter pair — and the new parameters know nothing about what was learned at rank 4.

**Measured impact**: 3–6 point F1 degradation at each rank transition, with recovery taking 50–100 steps. In noisy-data scenarios where rank changes are frequent, this creates cumulative instability.

## The Nested Solution

NestedLoRA allocates a **single matrix pair** at the maximum rank and controls active capacity via slicing:

```
Full allocation:    A ∈ ℝ^(d × 16)      B ∈ ℝ^(16 × d)

Rank 4 active:     A[:, :4]  @ B[:4, :]
Rank 8 active:     A[:, :8]  @ B[:8, :]
Rank 16 active:    A[:, :16] @ B[:16, :]
```

Because r4 parameters are literally the first 4 columns/rows of the r8 parameters (and r8 of r16), all learning at lower ranks is **preserved** at higher ranks. Rank transitions are instant (change an integer), cost zero re-allocation, and cause zero degradation.

This is the "nested orbital" analogy: the adapter occupies energy levels r4 ⊂ r8 ⊂ r16, like an electron in nested orbitals. Promotion/demotion changes the energy level without destroying the particle.

## Orbital Controller

The OrbitalController monitors gradient stress per adapter and decides when to promote or demote.

### Stress Signal

Each adapter's stress is the L2 norm of its LoRA gradients, smoothed by exponential moving average:

```
stress_t = (1 - α) · stress_{t-1} + α · ||∇(A, B)||₂
```

### Adaptive Thresholds

Rather than fixed thresholds, the controller computes them from a sliding window of recent stress values:

```
promote_threshold = μ + k · σ
demote_threshold  = μ - k · σ
```

Where μ and σ are the mean and standard deviation of the last N stress values. This makes the controller self-calibrating across different models, tasks, and training phases.

### Symmetric Return Logic

Promotion and demotion use the same threshold structure (k standard deviations from mean), ensuring the controller can both expand and contract capacity with equal sensitivity. This was critical for the noise resilience results — the controller must contract quickly when noise shocks subside, not just expand.

### Orbital Memory

Each adapter maintains an `orbit_stack`: a log of all transitions with step number, rank, and direction. This enables post-hoc analysis of the controller's decisions and correlation with training dynamics (loss spikes, task switches, noise injection).

## Design Decisions

### Why not SVD-based (AdaLoRA)?
SVD at every step is expensive. For DistilBERT it's tolerable; for 7B+ models it dominates training cost. The EMA + threshold approach is O(1) per step.

### Why not train all ranks simultaneously (DyLoRA)?
DyLoRA trains on all ranks in every forward pass, then selects post-hoc. This is wasteful when the optimal rank is stable for long stretches. Unified-LoRA only uses the active rank.

### Why not continuous rank (any integer)?
We tested continuous rank (±2 at each step). It oscillates. Discrete levels (4, 8, 16) with hysteresis from the adaptive thresholds produce stable behavior.

### Why these specific rank levels?
Powers-of-2 alignment with GPU tensor cores. The specific levels (4, 8, 16) are configurable; these are defaults that worked across our test suite.

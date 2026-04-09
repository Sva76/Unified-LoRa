# Unified-LoRA Architecture

## Two Problems, One Solution

### Problem 1: Cold-Start on Rank Transitions

Standard adaptive LoRA uses independent adapter pairs per rank. Switching from rank 4 to rank 8 activates a fresh (A₈, B₈) that knows nothing about what (A₄, B₄) learned. Measured impact: 3–6 F1 point degradation per transition.

**Solution — NestedLoRA:** A single matrix pair at max rank, with active rank = a slice boundary. r4 ⊂ r8 ⊂ r16, so all learning is preserved across transitions. Zero re-allocation, zero degradation.

### Problem 2: Rank Adjustment ≠ Mode Switching

Existing controllers (AdaLoRA, DyLoRA, simple EMA-threshold) treat rank adaptation as a quantitative dial: more stress → more rank, less stress → less rank. This misses the fact that different stress levels demand qualitatively different responses:

- **Low stress**: You want efficiency — use minimal rank.
- **Moderate stress**: Standard learning — use moderate rank.
- **High stress**: You need protection — max rank PLUS the ability to undo changes if the stress was a transient spike (noisy batch, data corruption, task switch).

A PID controller can adjust rank. It cannot save a snapshot and decide whether to roll back.

**Solution — FSM φ(t):** Three discrete modes (SINGLE, MULTI, MIRROR) with a composite stress signal and hysteresis. Mirror mode isn't just "more rank" — it's a different operational regime.

## The Synaptic Stress Signal φ(t)

Inspired by neurobiological stress response: neurons under stress don't just increase firing rate, they switch between potentiation, depression, and protective mechanisms.

### Components

**C — Convergence:** Is the model improving or diverging?

The ratio of fast EMA to slow EMA of the loss. When fast > slow, loss is trending upward → stress. This captures the "direction" of training.

**E — Entropy:** Are gradients coherent or chaotic?

Cosine similarity between consecutive gradient vectors. High similarity = aligned gradients = organized learning. Low similarity = chaotic gradient directions = the model is confused. This captures the "quality" of the learning signal.

**S — Stress Magnitude:** How large are the gradient forces?

EMA-smoothed L2 norm of LoRA gradients. This captures the raw "intensity" of parameter updates.

### Combination

```
φ(t) = 0.3·C + 0.3·E + 0.4·S
```

The weights are configurable. S gets slightly more weight because gradient magnitude is the most directly measurable signal.

### Normalization

φ_raw is normalized to [0, 1] via running z-score over a sliding window. This makes the controller self-calibrating: the same thresholds (φ_low=0.3, φ_high=0.7) work across different models, tasks, and training phases without manual tuning.

## FSM Transitions

```
           φ > φ_low            φ > φ_high
  SINGLE ─────────→ MULTI ─────────→ MIRROR
         ←─────────       ←─────────
           φ < φ_low            φ < φ_high
```

Rules:
- **No skip transitions**: SINGLE cannot jump directly to MIRROR.
- **Hysteresis**: A mode must be held for `hysteresis_steps` before any transition. This prevents oscillation on noisy φ values.
- **Mirror entry**: Saves a snapshot of current LoRA weights.
- **Mirror exit**: Computes relative drift. If weights moved <5%, the stress was transient → restore snapshot. If weights moved significantly, the stress was real → keep new weights.

The 5% threshold was determined empirically: transient noise spikes (corrupt batches, outlier gradients) cause <2% drift; real task shifts cause >10% drift. The 5% boundary sits cleanly between them.

## The Mirror Mechanism

Mirror mode is the key differentiator. When φ crosses the high threshold:

1. Controller saves `{lora_A, lora_B, step}` as a snapshot.
2. Rank expands to maximum (r=16) to absorb the stress.
3. Training continues with full capacity.

When φ drops below the threshold (recovery):

4. Controller measures how much weights drifted from the snapshot.
5. **If drift < 5%**: The stress was noise. Restore the pre-stress weights. The model "forgets" the noisy period.
6. **If drift ≥ 5%**: The stress was real (e.g., new task, distribution shift). Keep the new weights. The model "learned through" the stress.

This is analogous to synaptic consolidation in neuroscience: short-term stress is reversible; sustained stress leads to structural change.

## Design Decisions

### Why three modes, not continuous rank?
We tested continuous rank (±2 at each step). It oscillates. Discrete modes with hysteresis produce stable, interpretable behavior. Three modes map naturally to the neurobiological metaphor (efficient cruise / active learning / protective stabilization).

### Why φ(t) instead of just gradient norm?
Gradient norm (S alone) can't distinguish between "the model is learning fast" (high S, low E, low C) and "the model is confused by noise" (high S, high E, high C). The composite signal catches the difference.

### Why not SVD-based (AdaLoRA)?
Cost. SVD per layer per step is O(d²r), which is affordable for DistilBERT but prohibitive at 7B+. φ(t) is O(d) (a gradient norm + a cosine similarity).

### Why adaptive normalization?
Fixed thresholds would require different values for every model/task/learning rate. Running z-score makes {0.3, 0.7} universal defaults.

### Why 0.3 and 0.7?
These map to ±1.2σ from the mean in the normalized scale, which empirically gave the best balance between responsiveness and stability. They're configurable.

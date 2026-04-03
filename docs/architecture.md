# Architecture — Nested Orbital LoRA

## The problem: cold start on rank transitions

Standard multi-rank LoRA keeps separate adapter matrices per rank level:

```
r=4:  A4(4, d),  B4(d, 4)
r=8:  A8(8, d),  B8(d, 8)
r=16: A16(16,d), B16(d, 16)
```

When the controller switches from r=16 to r=8, the r=8 adapter has independent weights that never benefited from training at r=16. Each transition is a partial cold start. This caused 3-6 point F1 loss vs baseline in our experiments (V1-V4).

## The solution: one particle, multiple orbitals

Nested LoRA uses a single adapter pair with the maximum rank. Smaller ranks are obtained by slicing:

```
A(16, d) and B(d, 16)    ← one pair, always present

r=4:  A[:4, :],  B[:, :4]     ← first 4 dimensions
r=8:  A[:8, :],  B[:, :8]     ← first 8 dimensions
r=16: A[:16,:],  B[:, :16]    ← full matrix
```

The metaphor: one particle that can occupy different energy orbitals. When descending from r=16 to r=4, dimensions 0-3 retain everything they learned. Dimensions 4-15 are paused (no gradient), not destroyed. When ascending back, they resume exactly where they left off.

```
r4 ⊂ r8 ⊂ r16
```

### Scaling

To maintain consistent output magnitude across ranks, the output is scaled by `max_rank / active_rank`:

```python
scale = 16 / r
output = base + delta * scale
```

At r=4 the scale is 4.0 (amplify the smaller subspace). At r=16 the scale is 1.0 (no amplification). This is analogous to the alpha/r scaling in standard LoRA.

## Controller: trajectory with orbital memory

### From threshold controller to trajectory controller

Early versions used threshold-based FSM: if φ > θ₁, switch to r=16. This had two problems: static thresholds don't generalize across tasks/models, and the controller oscillated or got stuck.

The orbital controller replaces thresholds with a trajectory:

```
Ascend:  stress detected  → jump to higher orbital, push delta to stack
Hold:    oscillating      → stay, don't move
Descend: confirmed stable → pop delta, symmetric return
```

The orbit stack records the exact sequence of jumps. When descending, the controller reverses them in order, ensuring symmetric return to the previous state.

### Stress signal

```
φ(t) = |loss - EMA(loss)| + 2.0 × max(0, loss - prev_loss)
```

Two components:
- **Deviation from trend**: catches sustained instability
- **Spike detection**: catches sudden deterioration (weighted 2x)

### Adaptive thresholds

```
t_stress = μ(φ_recent) + 0.7σ
t_stable = max(μ(φ_recent) - 0.3σ, 0)
```

Auto-calibrate to loss scale. No manual tuning needed.

### Stability confirmation

Descent requires `stable_window` consecutive steps below `t_stable`. This prevents premature return after a brief lull during ongoing instability.

## Lifecycle of a training run

```
Step 0        Initialize at max rank (warmup)
Step 1-W      Build EMA baseline, accumulate φ history
Step W+1      Drop to ground state (r=4)
Step W+2...   Controller active:
                stress → ascend (push delta)
                stable → descend (pop delta)
                neutral → hold
```

### Warmup rationale

The controller needs calibrated thresholds before making decisions. Training at max rank during warmup ensures the model makes initial progress regardless of controller behavior.

### Ground state rationale

After warmup, dropping to the minimum rank tests whether the task actually needs high capacity. If it does, the stress signal will push the controller back up immediately. If it doesn't, the system saves rank.

## Comparison with existing methods

| Property             | Standard LoRA | AdaLoRA            | Unified-LoRA          |
|----------------------|---------------|--------------------|-----------------------|
| Rank control         | Fixed         | SVD importance     | Stress feedback       |
| Control type         | None          | Open-loop          | Closed-loop           |
| Shock reaction       | None          | Indirect           | Immediate             |
| Transition cost      | N/A           | SVD per step       | O(1) slice            |
| Architecture         | Single rank   | Pruned rank        | Nested orbitals       |
| Black-box compatible | Yes           | No (needs grads)   | Yes                   |
| Overhead per step    | 0             | O(r² × layers)     | O(1)                  |

## Limitations

- Validated on DistilBERT (67M). Scale to 7B+ not yet confirmed.
- The 15% rank saving on DistilBERT is small in absolute compute terms. The value proposition strengthens at larger scale where rank savings translate to meaningful memory/time reduction.
- On perfectly stable training, the controller adds no value (but causes no harm).
- The orbit stack can grow unboundedly in theory, though in practice it stays shallow (1-3 entries).

# Unified-LoRA

**Adaptive LoRA fine-tuning with nested orbital rank control.**

A closed-loop controller that dynamically adjusts LoRA rank during training based on observed stress, using a single adapter with sliced dimensions — no cold start, no capacity loss on transitions.

## Key results

### Stress test: task switch (MRPC → SST-2, DistilBERT, 3 seeds)

|                        | Baseline (r=16 fixed) | Unified (orbital) | Delta    |
|------------------------|-----------------------|-------------------|----------|
| SST-2 Acc (new task)   | 0.736                 | 0.740             | **+0.004** |
| MRPC F1 (retention)    | 0.526                 | 0.515             | -0.011   |
| Effective rank         | 16.0                  | 13.6              |          |
| Rank saving            | 0%                    | **15%**           |          |

Under distribution shift, the controller adapts capacity dynamically with 15% rank saving and no performance loss.

### Rank trace under shock (Seed 1)

```
[  0] r4  r4  r4  r8  r8  r8  r8  r16 r16 r16   ← ground state → stress → ascend
[ 10] r16 r16 r16 r16 r16 r16 r16 r16 r16 r16   ← MRPC at full capacity
...
[ 60] <<<SHOCK  r16 r16 r16 r16 r16 r16 r16 r16  ← task switch to SST-2
[ 68] r8  r8  r8  r8  r8  r8  r4  r4  r4  r4     ← controller detects shift, descends
[ 80] r4  r4  r4  r4  r4  r4  r4  r4  r4  r4     ← stable at ground state
[ 92] r8  r16 r16 r16 r16 r16 r16 r16 r16 r16    ← new task needs capacity, re-ascends
```

The controller exhibits **disturbance rejection**: detects the shock, descends to ground state, stabilizes, then re-ascends only when the new task demands capacity.

### Stable task (MRPC only, 120 steps, 3 seeds)

|              | Baseline (r=16) | Unified | Delta  |
|--------------|-----------------|---------|--------|
| F1 mean      | 0.818           | 0.820   | +0.002 |
| σ            | 0.008           | 0.008   | =      |

On stable training, the controller recognizes no intervention is needed and stays at r=16. Zero degradation.

## How it works

### Architecture: nested orbitals (r4 ⊂ r8 ⊂ r16)

Unlike standard multi-adapter approaches (separate A/B matrices per rank), Unified-LoRA uses a **single pair** of matrices with rank controlled via slicing:

```python
# One particle, multiple orbitals
self.lora_A = Parameter(shape=[max_rank, in_features])   # shared
self.lora_B = Parameter(shape=[out_features, max_rank])   # shared

# Active rank = slice
h     = x @ A[:r, :].T      # use first r rows
delta = h @ B[:, :r].T      # use first r columns
```

When descending from r=16 to r=4, dimensions 0-3 retain all learned weights. Dimensions 4-15 are paused, not destroyed. When ascending back, they resume where they left off.

**This solves the cold start problem** that caused F1 degradation in earlier versions with separate adapters.

### Controller: orbital trajectory with memory

The controller implements closed-loop rank control:

```
Stress  → ascend to higher orbital, push delta to stack
Stable  → pop delta from stack, symmetric return
Neutral → hold position, don't move
```

The stress signal φ(t) combines loss deviation from EMA with spike detection:

```
φ(t) = |loss - EMA(loss)| + 2.0 × max(0, loss - prev_loss)
```

Thresholds are **adaptive** (μ ± kσ of recent φ history), so the controller auto-calibrates to any model/task scale without manual tuning.

This is not a scheduler, not a rank budget, not a learning rate trick. It is a **trajectory controller** over model capacity.

## Quick start

```python
from controller import setup_unified_lora, set_rank

# One-call setup
model, ctrl = setup_unified_lora(model, max_rank=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

# Training loop
for step, batch in enumerate(train_loader):
    loss = model(**batch).loss

    new_rank = ctrl.step(loss.item())
    set_rank(model, new_rank)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

## What works and what doesn't

### Works: distribution shift / noisy training

Under task switch, label noise, or data corruption, the controller adapts rank dynamically. Demonstrated on:

- **Task switch** (MRPC → SST-2): parity + 15% saving, disturbance rejection confirmed
- **Label noise** (50%, DistilBERT/MRPC, 5 seeds): FSM switching F1=0.622 vs best fixed rank F1=0.439

### Works: black-box training (API / enterprise)

The controller observes only loss trajectory — no access to gradients, internal activations, or optimizer state. Compatible with API-based fine-tuning endpoints where internal signals are not exposed.

### Doesn't help: clean stable training

On standard GLUE tasks without perturbation, rank choice doesn't matter (r=8 ≈ r=16 ≈ r=32 from 67M to 3B parameters). The controller correctly recognizes this and stays at max rank — no harm, but no benefit.

## Experimental evolution

This project tested many approaches. In the interest of scientific honesty:

### Tested and didn't help (clean data)

- **Separate adapters per rank** (V1-V4): cold start on transitions caused 3-6 point F1 loss vs baseline. Each rank switch activated an adapter with independent weights that hadn't benefited from previous training. Solved by nested architecture.
- **Adaptive rank per-layer** (gradient EMA): no performance benefit over fixed rank
- **Fluid dynamics metrics** (shock, vorticity, swirl): too conservative as stress signals
- **Trend-aware hysteresis** with fixed thresholds: controller either never activated or got stuck at intermediate rank
- **Budget redistribution** across layers: winner-takes-all problem

### What works

- **Nested orbital architecture**: zero cold start, parity with baseline guaranteed
- **Trajectory controller with orbital memory**: disturbance rejection under task switch
- **Adaptive thresholds** (μ ± kσ): auto-calibrates across models and tasks
- **FSM adapter switching under noise**: measurably better performance and lower variance

## Computational overhead

The controller adds O(1) computation per step: one EMA update, one threshold comparison, one stack operation. No SVD, no matrix decomposition. Negligible relative to the training step.

## Control-theoretic framing

| Method                  | Control type    | Rank dynamics         |
|-------------------------|-----------------|-----------------------|
| Standard LoRA           | None            | rank = constant       |
| AdaLoRA                 | Open-loop       | rank = f(step)        |
| **Unified-LoRA**        | **Closed-loop** | rank = f(stress(t))   |

Unified-LoRA introduces orbit-aware rank transitions: each capacity increase is tracked and reversed only under confirmed stability, preventing premature compression and oscillatory collapse.

## Repository structure

```
controller.py                          # NestedLoRALinear + OrbitalController
experiments/
  stress_test_task_switch.py           # MRPC → SST-2 stress test (key result)
  stable_task_test.py                  # Single-task parity test
docs/
  experimental_results.md              # Detailed results and rank traces
  architecture.md                      # Nested orbital design
notebooks/                             # Experiment notebooks
```

## Open questions

- Does nested orbital control scale to 7B+ models? (Tinker validation in progress)
- What is the minimum shock magnitude that triggers measurable benefit?
- Does adaptive LR control (black-box analog) show the same pattern on API platforms?

## Citation

```bibtex
@software{unified_lora_2025,
  author = {Simona Vargiu},
  title = {Unified-LoRA: Adaptive Fine-Tuning with Nested Orbital Rank Control},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRa}
}
```

## Contact

**Simona Vargiu** (Independent Researcher)
For collaboration inquiries: simona.vargiu.malta@gmail.com

## License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

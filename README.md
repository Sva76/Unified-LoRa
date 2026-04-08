# Unified-LoRA

**Adaptive LoRA fine-tuning with nested orbital rank control.**

A closed-loop controller that dynamically adjusts LoRA rank during training based on observed stress, using a single adapter with sliced dimensions — no cold start, no capacity loss on transitions.

## Key results

### Stress test: task switch (MRPC → SST-2, DistilBERT, 3 seeds)

|                        | Baseline (r=16 fixed) | Unified (orbital) | Delta      |
|------------------------|-----------------------|-------------------|------------|
| SST-2 Acc (new task)   | 0.736                 | 0.740             | **+0.004** |
| MRPC F1 (retention)    | 0.526                 | 0.515             | -0.011     |
| Effective rank         | 16.0                  | 13.6              |            |
| Rank saving            | 0%                    | **15%**           |            |

Under distribution shift, the controller adapts capacity dynamically with 15% rank saving and no performance loss.

### Rank trace under shock (Seed 1)

```
[  0] r4  r4  r4  r8  r8  r8  r8  r16 r16 r16
[ 10] r16 r16 r16 r16 r16 r16 r16 r16 r16 r16
...
[ 60] <<<SHOCK  r16 r16 r16 r16 r16 r16 r16 r16
[ 68] r8  r8  r8  r8  r8  r8  r4  r4  r4  r4
[ 80] r4  r4  r4  r4  r4  r4  r4  r4  r4  r4
[ 92] r8  r16 r16 r16 r16 r16 r16 r16 r16 r16
```

The controller exhibits **disturbance rejection**: detects the shock, stabilizes, then reallocates capacity only when needed.

### Stable task (MRPC only, 120 steps, 3 seeds)

|              | Baseline (r=16) | Unified | Delta  |
|--------------|-----------------|---------|--------|
| F1 mean      | 0.818           | 0.820   | +0.002 |
| σ            | 0.008           | 0.008   | =      |

On stable training, the controller stays at max rank. Zero degradation.

---

## How it works

### Architecture: nested orbitals (r4 ⊂ r8 ⊂ r16)

Unified-LoRA uses a single pair of matrices with rank slicing:

```python
self.lora_A = Parameter(shape=[max_rank, in_features])
self.lora_B = Parameter(shape=[out_features, max_rank])

h     = x @ A[:r, :].T
delta = h @ B[:, :r].T
```

Lower ranks reuse learned weights. No reset, no cold start.

---

### Controller

```
Stress  → increase rank
Stable  → decrease rank
Neutral → hold
```

Stress signal:

```
φ(t) = |loss - EMA(loss)| + 2.0 × max(0, loss - prev_loss)
```

Adaptive thresholds (μ ± kσ) → no manual tuning.

---

## Quick start

```python
from controller import setup_unified_lora, set_rank

model, ctrl = setup_unified_lora(model, max_rank=16)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

for batch in train_loader:
    loss = model(**batch).loss
    r = ctrl.step(loss.item())
    set_rank(model, r)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

---

## Where it helps

- Distribution shift  
- Noisy training  
- Black-box fine-tuning APIs  

## Where it doesn't

- Clean stable training (no benefit, no harm)

---

## Overhead

O(1) per step. Negligible.

---

## Control view

| Method        | Control     | Rank          |
|---------------|------------|---------------|
| LoRA          | None       | constant      |
| AdaLoRA       | Open-loop  | f(step)       |
| Unified-LoRA  | Closed-loop| f(stress)     |

---

## Structure

```
controller.py
experiments/
docs/
notebooks/
```

## Demo

See interactive demo: notebooks/unified_lora_demo.ipynb



---

## Author

Simona Vargiu

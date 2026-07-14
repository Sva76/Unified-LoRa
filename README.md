# Unified-LoRA → the φ signal

**Status (July 2026): the original controller hypothesis was falsified.
What survived testing is φ — a gradient-free training-stress signal.
See [`validation/`](validation/) for the evidence bundle.**

## The honest history

This project began as Unified-LoRA: an adaptive controller that observes a
stress signal (φ) during LoRA fine-tuning and dynamically adjusts adapter
rank (NestedLoRA) and learning rate (Orbital controller). The hypothesis was
that this would beat established methods such as AdaLoRA on stability and
quality.

**It doesn't.** Multi-seed comparisons against AdaLoRA (official PEFT
implementation) falsified the hypothesis:

| Setting | AdaLoRA | Unified-LoRA | Fixed LoRA |
|---|---|---|---|
| Toy benchmark, final loss (3 seeds) | **0.198** | 0.226 | 0.227 |
| Toy benchmark, divergences (extreme regime) | **0/3** | 3/3 | 3/3 |
| MRPC, 50% label noise, F1 (3 seeds) | **0.518** | 0.358 | 0.374 |

The cause is structural: this controller regulates update *magnitude*, not
*direction*. AdaLoRA's SVD parametrization constrains direction, and that is
what stabilizes training in hard regimes.

## What survived: φ

φ is an EMA of upward loss jumps — computed from the loss stream alone, with
no access to gradients, weights, or model internals. It can therefore monitor
fine-tuning behind managed APIs (validated on Qwen3-8B via Tinker, where only
per-step loss is exposed).

Preliminary validation results (details, raw traces, and scripts in
[`validation/`](validation/)):

| Result | Value | Seeds/runs |
|---|---|---|
| φ in healthy training | ~0.011 (silent) | multi |
| Healthy vs unstable separation (aggressive LR, no induced shock) | 371× ± 40 | 3 |
| Blind classification, thresholds fixed in advance | 10/12 (extremes 10/10) | 12 |
| False alarms on healthy runs (300 steps each) | 0/6 | 6 |
| Detection latency at collapse | 1–2 steps | 4 |

In the current validation setting, φ behaves as a **specific,
near-instantaneous detector — not a predictor**: a pre-registered lead-time
test (Test 7) found a confound (task saturation) that voids the predictive
verdict; the corrected protocol (v3) is declared in the validation note
before execution.

## What's still in the repo

The controller code remains available as a research artifact:

- `nested_lora.py` — NestedLoRA: single allocation, nested rank slicing
  (zero cold-start rank switching)
- `orbital_controller.py` — FSM controller (HIGH/BASE/LOW) with snapshot
  and rollback
- `controller.py` — orchestration

## Quick start (φ monitor only)

```python
# φ requires only the loss stream:
class PhiMonitor:
    def __init__(self, beta=0.8):
        self.beta, self.ema_jump, self.prev = beta, 0.0, None
    def update(self, loss):
        jump = 0.0 if self.prev is None else max(0.0, loss - self.prev)
        self.prev = loss
        self.ema_jump = self.beta * self.ema_jump + (1 - self.beta) * jump
        return self.ema_jump

phi = PhiMonitor()
for batch in dataloader:
    loss = train_step(batch)
    if phi.update(loss.item()) >= 0.10:   # threshold from validation/
        alert()
```

Full controller (research artifact):

```python
from orbital_controller import setup_unified_lora
adapters, ctrl, opt = setup_unified_lora(model)   # returns 3 values
```

## Limitations

- Validation is preliminary: one model family (Qwen3-8B), one task type,
  3–12 seeds per test
- Predictive lead time untested (Test 7 v3 pending)
- The controller does not outperform AdaLoRA — use AdaLoRA if you need
  adaptive-rank fine-tuning

## License

Apache 2.0

## Contact

Simona Vargiu — Independent Researcher

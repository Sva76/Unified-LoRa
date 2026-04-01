
# Unified-LoRA

**An exploration of adaptive LoRA fine-tuning.**

This project investigated two ideas: (1) dynamically adjusting LoRA rank during training, and (2) using an FSM controller for training stability. Both were tested rigorously. Neither produced measurable benefit over simple baselines in multi-seed evaluation.

This repository documents the exploration honestly, including what worked, what didn't, and why.

## What was built

Two systems:

**Adaptive Rank Controller** — each LoRA layer adjusts its rank during training based on gradient stress (EMA). Layers under stress get more capacity; stable layers get less.

**FSM Mode Controller φ(t)** — a finite state machine that monitors training loss, detects instability, and switches between operational modes (Normal → Multi → Mirror) with adaptive learning rate.

## Results

### Adaptive Rank — DistilBERT (67M), 3 GLUE tasks, 3 seeds

| Task | r=8 (fixed) | r=16 (fixed) | Adaptive | Avg Rank |
|------|-------------|--------------|----------|----------|
| MRPC (F1) | **0.885 ± 0.007** | 0.882 ± 0.006 | 0.862 ± 0.025 | 9.1 |
| CoLA (MCC) | 0.474 ± 0.001 | **0.478 ± 0.011** | 0.477 ± 0.021 | 7.0 |
| RTE (Acc) | **0.560 ± 0.014** | 0.560 ± 0.018 | 0.543 ± 0.010 | 11.7 |

**Finding:** At this scale, r=8 ≈ r=16. The rank choice doesn't matter, so the adaptive controller has no problem to solve.

### Adaptive Rank — Qwen2.5-3B (3B, 4-bit), MRPC, 3 seeds, A100

| Mode | Acc | F1 | Rank |
|------|-----|-----|------|
| r=8 | 0.876 ± 0.008 | 0.913 ± 0.004 | 8 |
| r=16 | 0.875 ± 0.004 | 0.913 ± 0.002 | 16 |
| r=32 | 0.883 ± 0.012 | 0.918 ± 0.008 | 32 |
| Adaptive | 0.870 ± 0.014 | 0.911 ± 0.008 | 10.8 |

**Finding:** Rank doesn't matter at 3B either. Gap between r=8 and r=32 is 0.5%.

### FSM Stability — Qwen2.5-3B + LoRA, MRPC, 3 seeds, A100

| Mode | F1 | F1 Std | Spikes |
|------|-----|--------|--------|
| Baseline (no protection) | **0.916 ± 0.001** | | 330 |
| FSM φ(t) | 0.907 ± 0.005 | | 306 |
| Cosine scheduler | 0.898 ± 0.001 | | 335 |

**Finding:** Training instability (loss spikes) doesn't hurt final performance. The FSM reduces spikes slightly but at the cost of -0.9% F1 and higher variance.

### FSM on Tinker — Llama-3.2-1B (single run, manually induced shock)

```
[250] Mode=1  φ=0.333  (stable)
      SHOCK @ step 300
[350] Mode=2  φ=0.827  (Mirror activated)
      RECOVERY @ step 500
[550] Mode=1  φ=0.371  (return)
[700] Mode=1  φ=0.333  (baseline restored)
```

**Finding:** The FSM mechanism works — it detects shock and recovers. But this was a single run with induced instability, not a multi-seed validation against alternatives.

## What was tested and didn't help

Tested rigorously and documented honestly:

- **Adaptive rank per-layer** (gradient EMA): rank adapts but doesn't improve results
- **Fluid dynamics metrics** (shock, vorticity, swirl): too conservative
- **Budget redistribution** across layers: winner-takes-all problem
- **Adaptive gradient clipping** via swirl: inconsistent
- **Vincolo integration** (StabilityController + rank): zero shock events on stable training
- **Predictive signals** (trend + acceleration): no improvement
- **FSM φ(t) on natural training**: either no instability to handle, or instability doesn't hurt results
- **Stress testing** (high LR + label noise): training collapsed before FSM could act

## What was learned

1. **LoRA rank doesn't matter on classification tasks** from 67M to 3B. r=8 ≈ r=16 ≈ r=32 on MRPC. This means grid search over rank is wasted compute for these tasks.

2. **Training loss instability doesn't equal result instability.** Loss can spike wildly (0.0004 to 2.6) without affecting final metrics. Protecting against spikes is unnecessary on these tasks.

3. **Simplest baseline wins.** Fixed rank, fixed LR, standard grad clipping outperformed every adaptive method tested.

4. **Single-seed results are misleading.** Several configurations showed positive results on single seeds that disappeared on multi-seed evaluation.

5. **Per-layer rank patterns are real.** v_proj consistently needs more rank than q_proj, deep layers need more rank. These patterns reproduce across seeds even though they don't improve performance.

## Per-layer behavior

The adaptive controller discovers consistent patterns:

```
MRPC per-layer rank:
layer0.q: 7.9    layer0.v: 8.8
layer1.q: 7.8    layer1.v: 7.9
layer2.q: 7.9    layer2.v: 8.5
layer3.q: 8.8    layer3.v: 11.3    ← deep v_proj needs more
layer4.q: 10.3   layer4.v: 12.9
layer5.q: 7.6    layer5.v: 11.3
```

## Open questions

- Does rank matter on generation/instruction-following tasks (not classification)?
- Does rank matter at 7B-70B scale with rank ranges of 8-64?
- Is there a training regime (specific LR + noise combination) where the FSM provides measurable benefit?

## Quick start

```python
from unified_lora import inject_lora, get_lora_modules, setup_trainable

model = inject_lora(model, target_modules=["q_proj", "v_proj"])
model = setup_trainable(model)

# In training loop:
for m in get_lora_modules(model):
    m.update_rank()  # adaptive rank (works mechanically, no performance benefit found)
```

## Reproduce

```bash
pip install transformers datasets evaluate accelerate scikit-learn bitsandbytes

# DistilBERT multi-seed validation (~20 min, T4)
python validation_complete.py

# Qwen 3B scale test (~40 min, A100)
python scale_test.py

# Qwen 3B stability test (~40 min, A100)
python stability_test.py
```

## Repository structure

```
unified_lora.py            # Adaptive rank controller (drop-in module)
benchmark.py               # DistilBERT single-run benchmark
validation_complete.py     # Multi-seed + ablation (DistilBERT)
scale_test.py              # Qwen 3B rank test (A100)
stability_test.py          # FSM vs Baseline vs Cosine (A100)
controller.py              # FSM φ(t) controller
Archive/                   # Earlier experimental results
docs/                      # Additional documentation
notebooks/                 # Experiment notebooks
```

## Citation

```
@software{unified_lora_2025,
  author = {Simona Vargiu},
  title = {Unified-LoRA: An Exploration of Adaptive LoRA Fine-tuning},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRa}
}
```

## Contact

Simona Vargiu (Independent Researcher)
For collaboration inquiries: simona.vargiu.malta@gmail.com

## License

Apache License 2.0 — see LICENSE for details.

# Unified-LoRA

**Adaptive per-layer rank controller for LoRA fine-tuning.**

Automatically adjusts LoRA rank during training based on gradient stress. Eliminates rank as a hyperparameter.

## Quick start

```python
from unified_lora import inject_lora, get_lora_modules, setup_trainable

# Works with any model
model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased")
model = inject_lora(model, target_modules=["q_lin", "v_lin"])
model = setup_trainable(model)

# Standard training loop — add one line
for batch in train_loader:
    loss = model(**batch).loss
    loss.backward()
    clip_grad_norm_(model.parameters(), 1.0)

    for m in get_lora_modules(model):
        m.update_rank()  # ← this is the controller

    optimizer.step()
    optimizer.zero_grad()
```

## How it works

Each LoRA adapter tracks an EMA of its gradient norm. When stress increases, rank goes up. When stress decreases, rank goes down. Standard α/r scaling keeps the output magnitude stable across rank changes.

```
stress = 0.9 * stress + 0.1 * grad_norm
if stress_trend > threshold → rank += 2
if stress_trend < -threshold → rank -= 2
```

~30 lines of code. Zero external dependencies beyond PyTorch.

## Results (multi-seed, 3 seeds)

DistilBERT-base-uncased, 3 epochs, LR=5e-4, α=16:

| Task | Metric | r=8 (fixed) | r=16 (fixed) | Adaptive | Avg Rank |
|------|--------|-------------|--------------|----------|----------|
| MRPC | F1 | **0.885 ± 0.007** | 0.882 ± 0.006 | 0.862 ± 0.025 | 9.1 |
| CoLA | MCC | 0.474 ± 0.001 | **0.478 ± 0.011** | 0.477 ± 0.021 | 7.0 |
| RTE | Accuracy | **0.560 ± 0.014** | 0.560 ± 0.018 | 0.543 ± 0.010 | 11.7 |

### What these results show

**The controller works mechanically.** It adapts rank, discovers per-layer patterns (v_proj needs more rank than q_proj, deep layers need more rank), and converges to lower rank over training.

**At this scale, it doesn't beat fixed rank.** On DistilBERT/GLUE, r=8 ≈ r=16 — the rank choice barely matters. The controller has higher variance than fixed-rank baselines.

**The hypothesis:** adaptive rank becomes valuable on larger models (3B-7B+) where the gap between r=8 and r=16 is significant. This has not been tested yet due to compute constraints.

## Per-layer behavior

The controller discovers interpretable patterns consistently across seeds:

```
MRPC per-layer rank:
layer0.q: 7.9    layer0.v: 8.8
layer1.q: 7.8    layer1.v: 7.9
layer2.q: 7.9    layer2.v: 8.5
layer3.q: 8.8    layer3.v: 11.3    ← deep v_proj needs more
layer4.q: 10.3   layer4.v: 12.9    ← deep v_proj needs more
layer5.q: 7.6    layer5.v: 11.3
```

## What was tested and didn't help

- **Fluid dynamics metrics** (shock, vorticity, swirl): too conservative
- **Budget redistribution** across layers: winner-takes-all problem
- **Adaptive gradient clipping** via swirl: inconsistent across tasks
- **Vincolo integration** (LR stability controller): zero shock events detected at this scale — training too stable to trigger
- **Predictive signals** (trend + acceleration): no improvement over simple EMA

The simplest controller works best. Every added complexity hurt or had no effect.

## Two validated systems

### 1. FSM Mode Controller φ(t)

Validated on Tinker with Llama-3.2-1B. Switches between Single/Multi/Mirror modes based on training stress:

```
[250] Mode=1  φ=0.333  (stable)
      SHOCK @ step 300
[350] Mode=2  φ=0.827  (Mirror activated)
      RECOVERY @ step 500
[550] Mode=1  φ=0.371  (return)
[700] Mode=1  φ=0.333  (baseline restored)
```

### 2. Per-layer Adaptive Rank Controller

Validated on DistilBERT across 3 GLUE tasks with 3 seeds (results above).

## Scaling to larger models

**This is the key open question.** The controller needs a setting where rank selection matters.

### Test if rank matters on your model first

```python
# If these three give very different results, the controller can help.
# If they're similar, rank doesn't matter and neither will the controller.
for r in [4, 8, 16]:
    result = train_with_fixed_rank(model, rank=r)
    print(f"r={r}: {result}")
```

### Adapting to different architectures

```python
# Llama / Mistral / Qwen
inject_lora(model, target_modules=["q_proj", "v_proj"])

# All attention projections
inject_lora(model, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

# With 4-bit quantization
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True),
    device_map="auto",
)
inject_lora(model, target_modules=["q_proj", "v_proj"], max_r=32)
```

### What to report

If you test at larger scale, the key numbers are:

1. **Does rank matter?** r=4 vs r=8 vs r=16 performance gap
2. **Does adaptive match the best fixed rank?** Adaptive vs best-r
3. **Variance:** mean ± std over ≥3 seeds
4. **Rank distribution:** per-layer average ranks

## Repository structure

```
unified_lora.py          # Controller module (drop-in)
benchmark.py             # DistilBERT/GLUE benchmark
validation_complete.py   # Multi-seed + ablation
controller.py            # FSM controller φ(t) (legacy)
docs/                    # Additional documentation
notebooks/               # Experiment notebooks
```

## Reproduce

```bash
pip install transformers datasets evaluate accelerate scikit-learn

# Single run (~30 min on T4)
python benchmark.py

# Multi-seed validation (~20 min on T4)
python validation_complete.py
```

## Limitations

- At DistilBERT/GLUE scale, fixed rank works equally well
- Higher variance than fixed-rank baselines
- Not tested on models > 1.1B at multi-seed level
- Classification tasks only — no generation evaluation
- Dynamic rank doesn't reduce peak memory

## Citation

```
@software{unified_lora_2025,
  author = {Simona Vargiu},
  title = {Unified-LoRA: Adaptive Rank Controller for LoRA Fine-tuning},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRa}
}
```

## Contact

Simona Vargiu (Independent Researcher)
For collaboration inquiries: simona.vargiu.malta@gmail.com

## License

Apache License 2.0 — see LICENSE for details.

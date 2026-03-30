# Unified-LoRA

**Adaptive rank controller for LoRA fine-tuning.**

A lightweight per-layer controller that dynamically adjusts LoRA rank during training based on gradient stress, eliminating manual rank selection.

## What it does

Instead of fixing `rank=8` or `rank=16` and hoping it works, Unified-LoRA adapts the rank of each layer independently during training. Layers under stress get more capacity; stable layers get less. No grid search, no guessing.

## Key results

Evaluated on 4 GLUE tasks with DistilBERT-base-uncased, 3 epochs, LR=5e-4, α=16:

| Task | Metric | Baseline (r=16) | Adaptive | Avg Rank | Rank Reduction |
|------|--------|-----------------|----------|----------|----------------|
| MRPC | F1 | 0.882 | **0.886** | 9.3 | 42% |
| SST-2 | Accuracy | **0.898** | 0.885 | 7.0 | 56% |
| CoLA | MCC | 0.488 | **0.491** | 7.1 | 56% |
| RTE | Accuracy | 0.556 | **0.592** | 10.8 | 33% |

**Summary:** Comparable or better performance on 3/4 tasks with 33-56% fewer active rank parameters.

## How it works

Each LoRA adapter tracks the exponential moving average of its gradient norm. When the gradient stress increases (loss landscape is rough), the controller increases rank. When stress decreases (training is stable), rank is reduced.

```
For each layer, at each step:
  1. Compute grad_norm of LoRA parameters
  2. Update EMA: stress = 0.9 * stress + 0.1 * grad_norm
  3. If stress trend is increasing → rank += 2
  4. If stress trend is decreasing → rank -= 2
  5. Forward pass uses α/r scaling (standard LoRA)
```

The controller adds ~30 lines of code and zero computational overhead beyond gradient norm computation.

## Per-layer behavior

The controller discovers meaningful patterns automatically:

- **v_proj consistently needs more rank than q_proj** across all tasks
- **Deep layers (4-5) need more rank** than early layers on complex tasks
- **Easier tasks (SST-2) converge to lower rank** than harder tasks (RTE)

Example per-layer rank on MRPC:
```
layer0.q: 8.5    layer0.v: 8.3
layer1.q: 7.8    layer1.v: 7.6
layer2.q: 10.1   layer2.v: 8.3
layer3.q: 8.1    layer3.v: 10.1
layer4.q: 10.0   layer4.v: 12.1
layer5.q: 8.1    layer5.v: 12.3
```

## What was tested and didn't improve results

In the interest of scientific honesty, the following extensions were tested and did **not** outperform the simple Adaptive controller:

- **Fluid dynamics metrics** (shock, vorticity, swirl as stress signal): controller became too conservative, suppressing rank across all tasks
- **Budget redistribution** (fixed total rank budget shared across layers): "winner takes all" problem — high-stress layers starved low-stress layers
- **Adaptive gradient clipping** driven by swirl: helped on small tasks (RTE +2.5%), hurt on large tasks (SST-2 -1.7%)
- **Scaling without α/r**: performance came from implicit norm regulation, not true capacity control

The simple version works best. Complexity did not pay.

## Comparison with existing methods

| Method | Approach | Overhead | Our advantage |
|--------|----------|----------|---------------|
| AdaLoRA | SVD importance scoring per layer | High (SVD each step) | ~30 lines, zero SVD |
| DyLoRA | Train on multiple ranks simultaneously | Medium | Runtime adaptation, not post-hoc |
| Fixed LoRA | Manual rank selection | None | No guessing required |

Note: Direct numerical comparison with AdaLoRA was attempted but AdaLoRA did not function correctly in our setup (no rank pruning occurred). A fair comparison requires architecture-specific tuning of AdaLoRA scheduling parameters.

## Reproduce

Run `benchmark.py` on Google Colab with a T4 GPU (~30 min):

```bash
pip install transformers datasets evaluate accelerate scikit-learn
python benchmark.py
```

## Limitations

- Validated on DistilBERT (67M) and TinyLlama (1.1B) only
- Single-seed runs (variance not quantified)
- GLUE tasks only — no generation or instruction-following evaluation
- Rank changes don't reduce peak memory (matrices allocated at max_rank)
- Throughput overhead from dynamic slicing on small models

## Adapter size reduction

With average rank ~7 vs fixed rank 16:

| Model | r=16 adapter | r=7 adapter | Reduction |
|-------|-------------|-------------|-----------|
| DistilBERT | 4.3 MB | 1.9 MB | 56% |
| 7B (projected) | ~70 MB | ~31 MB | 56% |
| 70B × 100 tenants | ~7 GB | ~3.1 GB | 56% |

## Two validated systems

Unified-LoRA contains two complementary approaches, both validated:

### 1. FSM Mode Controller (φ(t))

Validated on Tinker with Llama-3.2-1B. A finite state machine driven by a synaptic stress parameter φ(t) = f(C, E, S) that switches between three operational modes:

- **Mode 0 (Single):** shared adapter, low stress (φ < 0.3)
- **Mode 1 (Multi):** task-specific adapters, moderate stress (φ < 0.7)
- **Mode 2 (Mirror):** stability snapshots, high stress (φ ≥ 0.7)

Demonstrated full stress → recovery cycle:
```
[250] Mode=1  φ=0.333  (stable)
      SHOCK @ step 300
[350] Mode=2  φ=0.827  (Mirror activated)
      RECOVERY @ step 500
[550] Mode=1  φ=0.371  (return)
[700] Mode=1  φ=0.333  (baseline restored)
```

Key finding: φ returns to pre-shock regime after recovery (0.33 → 0.83 → 0.33), indicating reversible stress handling.

### 2. Per-layer Adaptive Rank Controller

Validated on DistilBERT across 4 GLUE tasks (results table above). Each layer independently adjusts its LoRA rank based on gradient stress EMA. This is the simpler, more broadly validated system.

### Evolution

The project progressed from discrete mode switching (FSM) to continuous per-layer rank adaptation. Intermediate explorations included fluid dynamics metrics (shock, vorticity, swirl) and budget redistribution — these were tested rigorously but did not outperform the simple per-layer EMA approach. Details in "What was tested" above.

## Citation

If you use this work:

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

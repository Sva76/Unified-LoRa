# Unified-LoRA

**Adaptive rank controller for LoRA fine-tuning via nested orbital slicing.**

Instead of fixing `rank=8` and hoping it works, Unified-LoRA treats the adapter as a single particle occupying nested energy orbitals (r4 ⊂ r8 ⊂ r16). Rank is controlled by matrix slicing — not by swapping separate adapter pairs — eliminating cold-start degradation on rank transitions.

## Architecture: NestedLoRA + OrbitalController

```
                    ┌──────────────────────────┐
  Allocated once:   │   A (d_in × max_rank)    │
                    │   B (max_rank × d_out)    │
                    └──────────────────────────┘
                              │
                    ┌─────────┴─────────┐
  Active slice:     │ A[:, :r]  B[:r, :] │   r ∈ {4, 8, 16}
                    └───────────────────┘
                              │
  Forward:          h = xW + (x @ A_slice @ B_slice) · α/r
```

**NestedLoRA** (`nested_lora.py`): The execution engine. A single weight allocation where active rank = a slice boundary. Rank transitions are instant with zero re-allocation.

**OrbitalController** (`orbital_controller.py`): The control logic. Monitors gradient stress per layer, maintains adaptive thresholds (μ ± kσ) over a sliding window, and promotes/demotes rank with symmetric return logic. Each adapter tracks its own orbital history (`orbit_stack`).

**controller.py**: Convenience wrapper exposing both modules.

## Key Results

### 1. Performance parity with rank savings

Evaluated on GLUE tasks with DistilBERT-base-uncased (3 epochs, LR=5e-4, α=16):

| Task | Metric | Baseline (r=16) | Adaptive | Avg Rank | Rank Reduction |
|------|--------|-----------------|----------|----------|----------------|
| MRPC | F1     | 0.882           | **0.886**| 9.3      | 42%            |
| SST-2| Acc    | **0.898**       | 0.885    | 7.0      | 56%            |
| CoLA | MCC    | 0.488           | **0.491**| 7.1      | 56%            |
| RTE  | Acc    | 0.556           | **0.592**| 10.8     | 33%            |

Comparable or better on 3/4 tasks with 33–56% fewer active rank parameters.

### 2. Noise resilience (validated use case)

Unified-LoRA's clearest advantage: a **safety net for messy data**. Noise sweep across label corruption rates:

| Noise % | Fixed r=8 F1 | Adaptive F1 | Δ F1   | Variance ratio |
|---------|-------------|-------------|--------|----------------|
| 0%      | 0.87        | 0.87        | 0      | 1.0×           |
| 20%     | 0.79        | 0.82        | +3     | 2.1×           |
| 40%     | 0.61        | 0.74        | +13    | 4.7×           |
| 50%     | 0.42        | 0.73        | **+31**| **9.2×**       |

**Pattern**: No benefit on clean data at any scale (67M to 3B parameters). Measurable resilience when label noise exceeds ~40%. The controller absorbs noise shocks by temporarily expanding capacity, then contracts when stress subsides.

### 3. NestedLoRA stress tests

The orbital/nested approach validated against baseline LoRA:

- **Performance parity** across all tested configurations
- **~15% rank saving** from adaptive slicing
- **Zero cold-start**: rank transitions preserve learned parameters (lower ranks are always subsets of higher ranks)

### 4. Scale validation

Tested across model scales: 67M (DistilBERT), 1.1B (TinyLlama), 3B — noise resilience pattern confirmed at all scales.

## Quick Start

```python
from controller import setup_unified_lora

# Inject adapters + create controller
adapters, ctrl = setup_unified_lora(
    model,
    target_modules=["q_proj", "v_proj"],
    max_rank=16,
    rank_levels=[4, 8, 16],
)

# Training loop
optimizer = torch.optim.AdamW(ctrl.adapters.values(), lr=5e-4)
for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()
    transitions = ctrl.step()     # rank adaptation happens here
    optimizer.step()
    optimizer.zero_grad()

# Inspect
print(f"Avg rank: {ctrl.avg_rank():.1f}")
print(f"Rank saving: {ctrl.rank_saving_pct():.0f}%")
print(ctrl.get_summary())
```

## How the Controller Works

```
For each layer, at each step:
  1. Compute grad_norm of active LoRA slice
  2. Update EMA: stress = (1-α)·stress + α·grad_norm
  3. Compute adaptive thresholds: μ ± k·σ over sliding window
  4. If stress > upper threshold → promote to next orbital (r4→r8→r16)
  5. If stress < lower threshold → demote to lower orbital (r16→r8→r4)
  6. Log transition to orbit_stack for diagnostics
```

The controller adds ~100 lines of code and negligible overhead.

## Per-Layer Behavior

The controller discovers meaningful patterns automatically:

- **v_proj consistently needs more rank than q_proj** across all tasks
- **Deep layers need more rank** than early layers on complex tasks
- **Easier tasks converge to lower rank** than harder tasks
- On noise injection, stressed layers promote immediately; clean layers remain low

## File Structure

```
Unified-LoRa/
├── nested_lora.py              # Core: NestedLoRALinear + injection
├── orbital_controller.py       # Core: OrbitalController + setup
├── controller.py               # Convenience wrapper
├── benchmark.py                # GLUE evaluation script
├── notebooks/
│   └── unified_lora_demo.ipynb # Interactive demo
├── docs/
│   ├── ARCHITECTURE.md         # Design rationale
│   └── RESULTS.md              # Full experimental results
├── Unified-LoRA.pdf            # Paper
├── requirements.txt
├── LICENSE                     # Apache 2.0
└── README.md
```

## What Was Tested and Didn't Work

In the interest of scientific honesty:

- **Fluid dynamics metrics** (shock, vorticity, swirl as stress signal): controller became too conservative
- **Budget redistribution** (fixed total rank across layers): "winner takes all" — high-stress layers starved others
- **Separate adapter pairs per rank**: 3–6 point F1 cold-start degradation on transitions — this is the problem NestedLoRA solves
- **Adaptive gradient clipping** driven by swirl: inconsistent across task sizes

The simple nested-slicing + EMA-threshold approach works best.

## Evolution

The project progressed through three phases:
1. **FSM Mode Controller (φ(t))**: Discrete mode switching (Single/Multi/Mirror) driven by synaptic stress. Validated on Llama-3.2-1B via Tinker. Demonstrated reversible stress→recovery cycles.
2. **Per-layer Adaptive Rank**: Continuous EMA-based rank control per layer. Validated on DistilBERT/GLUE.
3. **NestedLoRA + OrbitalController** (current): Matrix slicing eliminates cold-start; orbital model with adaptive thresholds provides robust rank control.

## Comparison with Existing Methods

| Method    | Approach                        | Overhead      | Difference                        |
|-----------|---------------------------------|---------------|-----------------------------------|
| AdaLoRA   | SVD importance scoring          | High (SVD)    | No SVD, ~100 lines                |
| DyLoRA    | Train on multiple ranks jointly | Medium        | Runtime adaptation, not post-hoc  |
| Fixed LoRA| Manual rank selection           | None          | No guessing, noise resilience     |

## Adapter Size Reduction

With average rank ~7 vs fixed rank 16:

| Model               | r=16     | r≈7 adaptive | Reduction |
|---------------------|----------|-------------|-----------|
| DistilBERT (67M)    | 4.3 MB   | 1.9 MB      | 56%       |
| 7B (projected)      | ~70 MB   | ~31 MB      | 56%       |
| 70B × 100 tenants   | ~7 GB    | ~3.1 GB     | 56%       |

## Limitations

- Validated on DistilBERT (67M), TinyLlama (1.1B), and 3B — no 7B+ yet
- Single-seed runs (variance not fully quantified)
- GLUE tasks only — no generation or instruction-following evaluation
- Rank changes don't reduce peak memory (matrices allocated at max_rank)
- Tinker API integration pending bug resolution (Datum format for cross_entropy)

## Reproduce

```bash
pip install -r requirements.txt
python benchmark.py
```

Runs on Google Colab T4 (~30 min).

## Citation

```bibtex
@software{unified_lora_2025,
  author = {Simona Vargiu},
  title = {Unified-LoRA: Adaptive Rank Controller via Nested Orbital Slicing},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRa}
}
```

## Contact

Simona Vargiu (Independent Researcher)
For collaboration inquiries: simona.vargiu.malta@gmail.com

## License

Apache License 2.0 — see LICENSE for details.

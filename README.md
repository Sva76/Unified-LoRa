# Unified-LoRA

**LoRA fine-tuning with synaptic plasticity: a neurobiologically-inspired controller that adapts not just how much capacity to use, but how to use it.**

## The Idea

Biological neurons don't just "turn up the volume" under stress. They switch between qualitatively different operational modes — efficient baseline processing, active learning, and protective stabilization with the ability to roll back failed adaptations.

Unified-LoRA brings this to LoRA fine-tuning:

```
φ(t) low    →  SINGLE mode  →  Efficient cruise (rank 4)
φ(t) medium →  MULTI mode   →  Active learning  (rank 8)
φ(t) high   →  MIRROR mode  →  Max capacity + stability snapshot (rank 16)
                                 ↳ On exit: restore snapshot if stress was transient
```

The controller doesn't just adjust a number. It changes the **kind of response** to training dynamics — including saving and potentially restoring weight snapshots when stress turns out to be a transient spike rather than a real signal.

## φ(t): The Synaptic Stress Signal

The controller makes decisions based on a composite stress signal with three components:

```
φ(t) = w_C · C(t) + w_E · E(t) + w_S · S(t)

C = Convergence  — Is the loss improving or diverging?
                   (fast EMA vs slow EMA of loss)

E = Entropy      — Are gradients aligned or chaotic?
                   (cosine similarity of consecutive gradient directions)

S = Stress       — How large are the gradient forces?
                   (EMA of gradient L2 norm)
```

φ is normalized to [0, 1] via adaptive scaling (running μ ± σ), making it self-calibrating across models, tasks, and training phases.

## NestedLoRA: Zero Cold-Start Rank Transitions

The second key innovation: rank is controlled by **matrix slicing**, not by swapping separate adapter pairs.

```
Allocated once:    A ∈ ℝ^(d × 16)      B ∈ ℝ^(16 × d)

SINGLE  (r=4):     A[:, :4]  @ B[:4, :]
MULTI   (r=8):     A[:, :8]  @ B[:8, :]
MIRROR  (r=16):    A[:, :16] @ B[:16, :]
```

Because r4 ⊂ r8 ⊂ r16 (nested orbitals), rank transitions are **instant** — no re-allocation, no cold-start degradation (we measured 3–6 F1 point drops with separate adapter pairs; zero with nested slicing).

## Why This Matters

The combination of FSM + nested slicing gives you something no existing LoRA controller does:

1. **Qualitative mode switching** — Mirror mode isn't just "more rank." It saves a snapshot and can revert if the stress was transient. This is a fundamentally different response than SINGLE or MULTI.

2. **Reversible stress handling** — The system returns to pre-shock baseline after recovery:
   ```
   [250] MULTI   φ=0.33  (stable)
         SHOCK @ step 300
   [350] MIRROR  φ=0.83  (snapshot saved)
         RECOVERY @ step 500
   [550] MULTI   φ=0.37  (snapshot evaluated → restored or kept)
   [700] MULTI   φ=0.33  (baseline restored)
   ```

3. **Safety net for messy data** — Under label noise, the controller absorbs shocks via temporary capacity expansion, then contracts:

   | Noise % | Fixed r=8 F1 | Adaptive F1 | Δ F1   | Variance ratio |
   |---------|-------------|-------------|--------|----------------|
   | 0%      | 0.87        | 0.87        | 0      | 1.0×           |
   | 40%     | 0.61        | 0.74        | +13    | 4.7×           |
   | 50%     | 0.42        | 0.73        | **+31**| **9.2×**       |

   No benefit on clean data. Measurable resilience when noise exceeds ~40%. Confirmed at 67M, 1.1B, and 3B scales.

## Key Results

### GLUE Benchmark (DistilBERT-base-uncased)

| Task | Metric | Baseline (r=16) | Adaptive | Avg Rank | Rank Reduction |
|------|--------|-----------------|----------|----------|----------------|
| MRPC | F1     | 0.882           | **0.886**| 9.3      | 42%            |
| SST-2| Acc    | **0.898**       | 0.885    | 7.0      | 56%            |
| CoLA | MCC    | 0.488           | **0.491**| 7.1      | 56%            |
| RTE  | Acc    | 0.556           | **0.592**| 10.8     | 33%            |

### NestedLoRA Stress Tests
- Performance parity with baseline across all configurations
- ~15% rank saving from adaptive slicing
- Zero cold-start degradation on transitions

## Quick Start

```python
from controller import setup_unified_lora

# Inject adapters + create FSM controller
adapters, ctrl = setup_unified_lora(
    model,
    target_modules=["q_proj", "v_proj"],
    max_rank=16,
    rank_levels=[4, 8, 16],  # [SINGLE, MULTI, MIRROR]
    phi_low=0.3,              # SINGLE↔MULTI threshold
    phi_high=0.7,             # MULTI↔MIRROR threshold
)

# Training loop — pass loss for convergence signal
for batch in dataloader:
    outputs = model(**batch)
    outputs.loss.backward()
    transitions = ctrl.step(loss=outputs.loss.item())
    optimizer.step()
    optimizer.zero_grad()

    # Inspect mode transitions
    for t in transitions:
        print(f"[{t.step}] {t.layer_name}: {t.old_mode.name}→{t.new_mode.name} "
              f"φ={t.phi:.3f} rank {t.old_rank}→{t.new_rank}"
              f"{' 📸 snapshot saved' if t.snapshot_saved else ''}"
              f"{' ↩️ snapshot restored' if t.snapshot_restored else ''}")

# Summary
print(ctrl.get_summary())
print(f"Mode distribution: {ctrl.mode_distribution()}")
print(f"Avg rank: {ctrl.avg_rank():.1f} ({ctrl.rank_saving_pct():.0f}% saving)")
```

## File Structure

```
Unified-LoRa/
├── nested_lora.py              # NestedLoRALinear + injection helpers
├── orbital_controller.py       # OrbitalController FSM + φ(t) signal
├── controller.py               # Convenience wrapper
├── benchmark.py                # GLUE evaluation script
├── notebooks/
│   └── unified_lora_demo.ipynb
├── docs/
│   ├── ARCHITECTURE.md         # Design rationale
│   └── RESULTS.md              # Full experimental results
├── Unified-LoRA.pdf
├── requirements.txt
├── LICENSE
└── README.md
```

## How the Controller Works (Detail)

```
For each adapter, at each step:

  1. SENSE
     - Compute gradient norm → update S (stress magnitude EMA)
     - Compute gradient direction cosine similarity → update E (entropy EMA)
     - Compute loss fast/slow EMA ratio → update C (convergence signal)

  2. COMPUTE φ
     - φ_raw = w_C·C + w_E·E + w_S·S
     - Normalize to [0,1] via running z-score

  3. DECIDE (FSM with hysteresis)
     - φ < 0.3 → SINGLE (if sustained for hysteresis_steps)
     - 0.3 ≤ φ < 0.7 → MULTI
     - φ ≥ 0.7 → MIRROR
     - No skip transitions: SINGLE↔MULTI↔MIRROR

  4. ACT
     - SINGLE: slice to r=4, minimal parameters
     - MULTI:  slice to r=8, standard fine-tuning
     - MIRROR: slice to r=16, save weight snapshot
     - On MIRROR→MULTI: evaluate drift. If <5% relative change,
       restore pre-Mirror weights (stress was transient noise).
```

## Comparison with Existing Methods

| Method     | Approach                        | Mode switching | Snapshot/rollback | Overhead     |
|------------|---------------------------------|----------------|-------------------|--------------|
| AdaLoRA    | SVD importance scoring          | No             | No                | High (SVD)   |
| DyLoRA     | Train on multiple ranks jointly | No             | No                | Medium       |
| Fixed LoRA | Manual rank selection           | No             | No                | None         |
| **Unified-LoRA** | **FSM φ(t) + nested slicing** | **Yes (3 modes)** | **Yes (Mirror)** | **Negligible** |

## What Was Tested and Didn't Work

- **Fluid dynamics metrics** (shock, vorticity, swirl): controller too conservative
- **Budget redistribution**: "winner takes all" across layers
- **Separate adapter pairs per rank**: 3–6 F1 cold-start degradation → solved by NestedLoRA
- **Continuous rank (±2/step)**: oscillation → solved by discrete FSM modes + hysteresis

## Limitations

- Validated on DistilBERT (67M), TinyLlama (1.1B), and 3B — no 7B+ yet
- Single-seed runs
- GLUE tasks only — no generation evaluation
- Peak memory unchanged (matrices allocated at max_rank)
- Tinker API integration pending bug resolution

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
  title = {Unified-LoRA: Synaptic Plasticity Controller for Adaptive LoRA Fine-Tuning},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRa}
}
```

## Contact

Simona Vargiu (Independent Researcher)
For collaboration inquiries: simona.vargiu.malta@gmail.com

## License

Apache License 2.0 — see LICENSE for details.

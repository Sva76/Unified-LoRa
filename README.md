# Unified-LoRA

**Adaptive LoRA fine-tuning with FSM-driven adapter switching.**

An exploration of adaptive LoRA fine-tuning that discovered a specific use case: under noisy training conditions, an FSM controller that switches between adapters of different rank based on training stress significantly outperforms fixed-rank LoRA.

## Key finding

Under noisy conditions (label noise), the FSM adapter switching controller provides measurably better performance and lower variance than any fixed-rank baseline.

**5 seeds, DistilBERT + LoRA, MRPC, 50% label noise:**

| Method | Mean F1 | Std | Per-seed F1 |
|--------|---------|-----|-------------|
| r=4 fixed | 0.410 | 0.323 | [0.62, 0.61, 0.04, 0.01, 0.78] |
| r=16 fixed | 0.439 | 0.234 | [0.73, 0.55, 0.31, 0.06, 0.55] |
| **FSM switching** | **0.622** | **0.174** | [0.66, 0.29, 0.70, 0.65, 0.81] |
| Random switching | 0.275 | 0.283 | [0.13, 0.08, 0.35, 0.01, 0.79] |

**Why this matters:**
- FSM has the highest mean F1 (+18 points over best fixed rank)
- FSM has the lowest variance (most robust across seeds)
- Random switching is worst — proving the intelligence of the switching matters, not just having multiple adapters
- Fixed ranks collapse on bad seeds (r4 → 0.007, r16 → 0.055); FSM never drops below 0.294

## How it works

The FSM controller monitors training loss and switches between three LoRA adapters (r=4, r=8, r=16) based on a stress signal φ(t):

```
φ(t) = f(loss_EMA, instability, progress)

φ < θ₀  → Mode 0: use r=4 adapter  (low stress, light capacity)
φ < θ₁  → Mode 1: use r=8 adapter  (moderate stress)
φ ≥ θ₁  → Mode 2: use r=16 adapter (high stress, full capacity)
```

Under normal training, the controller stays in low-rank mode (efficient). When noise or instability hits, it switches to higher rank (resilient). When stress passes, it returns to low rank.

## Where it works and where it doesn't

### Works: noisy/unstable training
- Label noise, data corruption, adversarial batches
- The controller acts as a resilience mechanism
- Degrades less than fixed rank under stress

### Doesn't work: clean training
- On standard GLUE tasks without noise, r=8 ≈ r=16 ≈ r=32
- The rank choice doesn't matter, so the controller has no problem to solve
- Tested on DistilBERT (67M), TinyLlama (1.1B), Qwen2.5-3B — same conclusion

### Doesn't work: rank adaptation without switching
- Per-layer gradient EMA rank controller was tested extensively
- Multi-seed validation showed no benefit over fixed rank on clean data
- Higher variance than fixed-rank baselines

## Full experimental history

This project tested many approaches. In the interest of scientific honesty:

**Tested and didn't help on clean data:**
- Adaptive rank per-layer (gradient EMA) — no performance benefit
- Fluid dynamics metrics (shock, vorticity, swirl) — too conservative
- Budget redistribution across layers — winner-takes-all problem
- Adaptive gradient clipping — inconsistent
- Vincolo StabilityController integration — zero shock events on stable training
- FSM with LR control only (no adapter switching) — loses to cosine scheduler

**What works:**
- FSM with adapter switching under noisy conditions (this finding)
- FSM stress-recovery cycle validated on Tinker with Llama-3.2-1B

## Scale test results (clean data)

Qwen2.5-3B, 4-bit, MRPC, 3 seeds, A100:

| Mode | Acc | F1 | Rank |
|------|-----|-----|------|
| r=8 | 0.876 ± 0.008 | 0.913 ± 0.004 | 8 |
| r=16 | 0.875 ± 0.004 | 0.913 ± 0.002 | 16 |
| r=32 | 0.883 ± 0.012 | 0.918 ± 0.008 | 32 |

Rank doesn't matter at 3B on classification. Gap r=8 vs r=32: 0.5%.

## FSM on Tinker (Llama-3.2-1B)

Demonstrated full stress → recovery cycle with manually induced shock:

```
[250] Mode=1  φ=0.333  (stable)
      SHOCK @ step 300
[350] Mode=2  φ=0.827  (Mirror activated)
      RECOVERY @ step 500
[550] Mode=1  φ=0.371  (return)
[700] Mode=1  φ=0.333  (baseline restored)
```

## What was learned

1. **LoRA rank doesn't matter on clean classification tasks** from 67M to 3B
2. **Under noise, adaptive switching beats fixed rank** — the FSM provides resilience
3. **Switching intelligence matters** — random switching is worst
4. **Single-seed results are misleading** — always use multi-seed
5. **The simplest baseline wins on clean data** — complexity only pays under stress

## Reproduce

```bash
pip install transformers datasets evaluate accelerate scikit-learn peft

# Clean data benchmark
python benchmark.py

# Multi-seed validation
python validation_complete.py

# Noisy training FSM test (the key finding)
python fsm_noise_test.py
```

## Open questions

- Does FSM adapter switching help at 7B+ scale under noise?
- What noise levels trigger the benefit? (tested at 50%, untested at 5-20%)
- Does it help on generation/instruction tasks with naturally noisy data?

## Repository structure

```
unified_lora.py            # Adaptive rank controller module
benchmark.py               # Clean data benchmark
validation_complete.py     # Multi-seed clean data validation
fsm_noise_test.py          # FSM adapter switching under noise (key result)
controller.py              # FSM φ(t) controller
Archive/                   # Earlier experimental results
docs/                      # Additional documentation
notebooks/                 # Experiment notebooks
```

## Citation

```
@software{unified_lora_2025,
  author = {Simona Vargiu},
  title = {Unified-LoRA: Adaptive LoRA Fine-tuning with FSM Adapter Switching},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRa}
}
```

## Contact

Simona Vargiu (Independent Researcher)
For collaboration inquiries: simona.vargiu.malta@gmail.com

## License

Apache License 2.0 — see LICENSE for details.

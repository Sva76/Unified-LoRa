# Experimental Results

## 1. GLUE Benchmark (DistilBERT-base-uncased)

Setup: 3 epochs, LR=5e-4, α=16, max_rank=16, rank_levels=[4, 8, 16].

| Task  | Metric   | Baseline (r=16) | Adaptive | Avg Rank | Rank Reduction |
|-------|----------|-----------------|----------|----------|----------------|
| MRPC  | F1       | 0.882           | **0.886**| 9.3      | 42%            |
| SST-2 | Accuracy | **0.898**       | 0.885    | 7.0      | 56%            |
| CoLA  | MCC      | 0.488           | **0.491**| 7.1      | 56%            |
| RTE   | Accuracy | 0.556           | **0.592**| 10.8     | 33%            |

Summary: Comparable or better on 3/4 tasks with 33–56% fewer active parameters.

Per-layer rank on MRPC:

```
layer0.q: 8.5    layer0.v: 8.3
layer1.q: 7.8    layer1.v: 7.6
layer2.q: 10.1   layer2.v: 8.3
layer3.q: 8.1    layer3.v: 10.1
layer4.q: 10.0   layer4.v: 12.1
layer5.q: 8.1    layer5.v: 12.3
```

Pattern: v_proj in deep layers consistently receives higher rank.

## 2. Noise Sweep

Setup: MRPC task, label noise injected at various rates. Fixed r=8 baseline vs adaptive controller.

| Noise % | Fixed r=8 F1 | Adaptive F1 | Δ F1  | Variance ratio (fixed/adaptive) |
|---------|-------------|-------------|-------|----------------------------------|
| 0%      | 0.87        | 0.87        | 0     | 1.0×                             |
| 10%     | 0.84        | 0.85        | +1    | 1.3×                             |
| 20%     | 0.79        | 0.82        | +3    | 2.1×                             |
| 30%     | 0.72        | 0.79        | +7    | 3.2×                             |
| 40%     | 0.61        | 0.74        | +13   | 4.7×                             |
| 50%     | 0.42        | 0.73        | +31   | 9.2×                             |

Key finding: No benefit on clean data at any tested scale. Measurable resilience when noise exceeds ~40%. The controller acts as a "safety net for messy data."

Validated by community member POIZONE (AMD Developer Community Discord).

## 3. Scale Validation

The noise resilience pattern holds across model sizes:

| Model           | Parameters | Clean Δ F1 | 50% Noise Δ F1 |
|-----------------|-----------|------------|-----------------|
| DistilBERT      | 67M       | 0          | +31             |
| TinyLlama       | 1.1B      | 0          | +28             |
| 3B model        | 3B        | 0          | +26             |

Pattern: Noise benefit is model-size-independent. Clean-data parity is consistent.

## 4. NestedLoRA Stress Tests

Comparison of nested slicing (NestedLoRA) vs independent adapter pairs:

- **Performance**: Parity with baseline across all configurations
- **Rank saving**: ~15% from adaptive slicing
- **Cold-start**: Zero degradation on transitions (vs 3–6 F1 point drops with separate pairs)

## 5. FSM Controller (Tinker / Llama-3.2-1B)

Earlier validation of the finite state machine approach with synaptic stress φ(t):

```
[250] Mode=1  φ=0.333  (stable)
      SHOCK @ step 300
[350] Mode=2  φ=0.827  (Mirror activated)
      RECOVERY @ step 500
[550] Mode=1  φ=0.371  (return)
[700] Mode=1  φ=0.333  (baseline restored)
```

Key finding: φ returns to pre-shock regime (0.33 → 0.83 → 0.33), indicating fully reversible stress handling. This motivated the symmetric return logic in the OrbitalController.

## 6. Negative Results

The following were tested rigorously and did not improve over the simple approach:

| Extension                     | Result                                              |
|-------------------------------|-----------------------------------------------------|
| Fluid dynamics stress signal  | Controller too conservative, suppressed all ranks    |
| Budget redistribution         | "Winner takes all" across layers                     |
| Adaptive gradient clipping    | +2.5% on RTE, −1.7% on SST-2 — inconsistent        |
| Scaling without α/r           | Gains from implicit norm regulation, not capacity    |
| Continuous rank (±2/step)     | Oscillation; discrete orbitals with hysteresis better|
| Separate adapter pairs        | 3–6 F1 cold-start degradation per transition         |

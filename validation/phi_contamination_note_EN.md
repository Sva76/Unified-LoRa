# Sensor–actuator contamination in adaptive LoRA control

**Technical note — local (Colab) experiments on Qwen2.5-0.5B-Instruct**

Companion to `phi_validation_note_EN.pdf`. That note validated φ as a
gradient-free stress signal via the Tinker API, where only the high-level
signal could be computed. This note reports the first experiments running the
**full Unified-LoRA controller** (OrbitalController + NestedLoRA) on a real
LLM, and the contamination effect they uncovered.

---

## 1. Summary

In an adaptive controller, the signal used to make decisions can be corrupted
by those same decisions. We measured this in Unified-LoRA and, in a control
experiment, found it **absent** in AdaLoRA. The difference is mechanistic:
contamination appears when the actuator acts on the same channel the sensor
reads.

| | Unified-LoRA | AdaLoRA |
|---|---|---|
| Actuator | global rank + learning rate | masking of singular values |
| Sensor channel | loss | gradients |
| Same channel? | **yes** | no |
| Contamination | **measured** | not detectable (0.4%, within noise) |

---

## 2. Supported claims

**2.1 The controller contaminates its own sensor.**
In runs with **no shock at all**, mean φ over the reference window rose with
FSM activity: 1.68 at `stable_window=30` → 2.15 at `stable_window=10`. Roughly
half the apparent "background stress" was produced by the actuator itself.

**2.2 Separating sensor from actuator restores detection.**
Same signal, same threshold rule (μ + kσ), same data, same seeds. The only
change: training ran at fixed rank and fixed LR while the controller observed
without acting.

| | φ inside the control loop | φ separated |
|---|---|---|
| Shocks detected | 2/5 seeds | **5/5** |
| False positives | 0/5 | 0/5 |
| Detection latency | 4 steps | **0–2 steps** |
| Clean calibration range | unstable point | **k ∈ [1.1, 4.0]** |
| φ shock/healthy ratio | ~2.9× | **~5.8×** |

Per-seed detection steps at k=1.5 (shock onset = step 80): 82, 80, 81, 80, 81.
No alarm fired in any control run.

**2.3 AdaLoRA does not show the effect (counter-case).**
Arm A ran `update_and_allocate` (scores **and** rank reallocation); arm B ran
`update_ipt` only (scores, rank frozen). Importance-signal volatility in
control runs: 5.834e-05 (A) vs 5.857e-05 (B) — 0.4%, opposite to the predicted
direction. Discrimination 0.60× vs 0.59×. Consistent across 3 seeds.

Actuator activity was verified, not assumed: effective rank fell 576 → 384
(**33%**) in arm A and stayed at 576 (0%) in arm B.

---

## 3. Falsified claims

**3.1 The FSM is not a reliable detector inside the control loop.**
At the best calibration found on one seed (`stable_window=10, stress_k=1.5`),
replication over 5 seeds gave **2/5** clean discriminations. Neighbouring cells
(sw=8, sw=12) gave 2/3 each, but *which* seeds succeeded varied
unsystematically — a band of uniform unreliability, not a plateau.

**3.2 Detection inside the loop is not monotone in signal strength.**
Seed 41 had the **strongest** φ separation (4.29×) and never triggered; seed 11,
the weakest (1.97×), did. The adaptive threshold (μ + kσ computed over φ's own
history) tracks the signal it measures, so sustained elevation raises the bar
along with it. Only relative dynamics matter; absolute level does not.

**3.3 Contamination is not a category defect of adaptive methods.**
This was our working hypothesis before test 13. The AdaLoRA counter-case
falsifies it. The condition is narrower: the actuator must act on the sensor's
channel.

**3.4 An absolute threshold is not the fix.**
Tested against the adaptive rule with leave-one-seed-out calibration on control
runs. Absolute thresholds were slower or produced false positives (4/5 with 1
FP at margin ×1.0; 5/5 at ×1.25 but latency 2.2 vs 0.8). Absolute φ levels vary
across seeds (0.39–0.97 in control runs); the adaptive rule normalises per run.
**The threshold rule was never the problem — the loop was.**

---

## 4. Protocol

- Model: Qwen2.5-0.5B-Instruct, float32, adapters on `q_proj`/`v_proj`
- 200 steps, induced shock at steps 80–120 (corrupted targets)
- Seeds: 11, 23, 37, 41, 53 (3 seeds for the AdaLoRA arms)
- Paired comparison: shock vs control arm, same seed, same data, **same step
  window** — an in-run "inside vs outside shock" comparison is biased, because
  φ's baseline is non-stationary
- Detector evaluation is offline on saved traces: data generated once, multiple
  detectors compared on identical inputs
- Pre-onset alarms (before step 80 in a shock arm) are counted explicitly as
  errors, not silently discarded

**Methodological note (PEFT).** `update_and_allocate` must be called **after**
`optimizer.step()`, per PEFT's documented order. Called before, AdamW momentum
immediately restores the just-masked entries and the actuator is effectively
inert — effective rank stays at 100%. Our first AdaLoRA run had this bug and
produced a null result that looked like a genuine negative. Verifying that the
mechanism under test is actually active is a precondition for interpreting any
negative result.

---

## 5. Limitations

- One model (0.5B). Not tested at larger scale.
- One stress condition: abrupt target corruption. Gradual degradation untested.
- **Fixed onset (step 80).** The detector knows where to look; this is not
  blind detection.
- 5 seeds (3 for AdaLoRA). Direction, not statistical significance.
- k ∈ [1.1, 4.0] is the range *tested clean*, not a measured upper bound.
- Detecting stress is not the same as acting usefully on it. No claim is made
  that observing φ improves training outcomes.

---

## 6. Open questions

1. **Blind detection** — randomised onset, mixed with shock-free traces.
2. **Gradual degradation** — does φ detect slow deterioration, or only abrupt
   shocks?
3. **Lead time** — does φ rise *before* collapse, or only during it?
   (See `phi_validation_note_EN.pdf` §4.2: the earlier attempt was void due to
   task saturation.)
4. **Generality** — second model; and a map of contamination as a function of
   how strongly the actuator perturbs the sensor's channel.

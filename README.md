# Unified-LoRA → the φ signal

**Status (August 2026).** The original controller hypothesis was falsified by
its own test campaign. What survives is a loss-only training-stress signal and
a documented failure mode of adaptive controllers that read what they modify.
A new ReViSQL/RLVR validation further shows that the original one-sided φ does
**not** transfer as a reliable predictor of future PPO instability, and that a
symmetric `EMA(|Δloss|)` repair does not recover convincing predictive value in
the tested 50-step trajectory. Evidence, raw logs and reanalysis notes are in
[`validation/`](validation/).

This repository is a research artifact, not a library. Nothing here is
recommended for production fine-tuning.

---

## 1. The honest history

The project began as Unified-LoRA: an adaptive controller that observes a
stress signal during LoRA fine-tuning and adjusts adapter rank (NestedLoRA)
and learning rate (Orbital FSM controller). The hypothesis was that this would
beat established methods such as AdaLoRA on stability and quality.

**It doesn't.** Multi-seed comparisons against AdaLoRA (official PEFT
implementation) falsified it:

| Setting | AdaLoRA | Unified-LoRA | Fixed LoRA |
|---|---|---|---|
| Toy benchmark, final loss (3 seeds) | **0.198** | 0.226 | 0.227 |
| Toy benchmark, divergences (extreme regime) | **0/3** | 3/3 | 3/3 |
| MRPC, 50% label noise, F1 (3 seeds) | **0.518** | 0.358 | 0.374 |

The cause appears structural: this controller regulates update *magnitude*,
not *direction*. AdaLoRA's SVD parametrisation constrains direction, and that
is what stabilises training in hard regimes.

---

## 2. Two different signals share the name φ

This is a naming defect in earlier versions of this README, corrected here.
The repository contains **two distinct signals**, on scales roughly 170×
apart. They are not interchangeable and their thresholds do not transfer.

| | **φ_jump** (Tests 5–7, Tinker) | **φ_dev** (Tests 8–14, local controller) |
|---|---|---|
| Definition | `EMA_0.8(max(0, Δloss))` | `EMA( abs(loss − EMA(loss)) + 0.5·max(0, Δloss) )` |
| Typical value, healthy run | ~0.011 | ~1.9 |
| Threshold rule | fixed, 0.10 | adaptive, `μ + k·σ` over φ history |
| Where implemented | `validation/test5–7*.py` | `orbital_controller.py` |

**φ_jump is loss-only.** Verified numerically: the φ values logged in
`validation/phi_lead_time_log.json` reproduce the pure loss-derived EMA to
within 6e-14. (`PhiMonitor.update` accepts an optional `grad_norm` term
weighted 0.01; Tinker did not expose `grad_norm`, so the term was inert
throughout. It should be removed — see §6.)

---

## 3. What φ_jump does, on Qwen3-8B via Tinker

Thresholds were fixed from Test 5 data and reused unchanged in Tests 6–7.
Verdict criteria, including failure conditions, are pre-registered inside the
Test 5–7 scripts.

| Result | Value | Runs |
|---|---|---|
| φ_jump in healthy training | ~0.011 (silent) | 3 |
| Healthy vs unstable separation (aggressive LR, no induced shock) | 371× | 3 |
| Blind classification, thresholds fixed in advance | 10/12 (extremes 10/10) | 12 |
| False alarms on healthy runs (300 steps each) | 0/6 | 6 |
| Detection latency at collapse | 1–2 steps | 4 |

φ_jump behaves as a **specific, near-instantaneous detector — not a
predictor.** A pre-registered lead-time test (Test 7) hit a confound: the task
saturated to near-zero loss (memorisation), so no loss-derived signal could
have shown precursors. That verdict is declared **void**, not favourable. The
corrected protocol (v3, non-memorisable stream) is declared in the validation
note before execution and has not yet been run.

**On novelty.** φ_jump is close to existing loss-domain spike statistics
(K2-V2's local robust z-score on loss; ZClip's EMA z-score on gradient norms;
SPAM's Gradient Spike Score). The signal itself is a one-sided variant of a
known family. The only part that is plausibly distinctive is the operating
constraint: it needs nothing but the loss stream, so it runs behind managed
fine-tuning APIs where gradients and internals are not exposed.

The earlier version of this README stated that comparison against rolling STD
and robust MAD was untested. That is no longer fully true: the August 2026
ReViSQL/PPO diagnostic below includes those loss-domain baselines. In that
setting, neither φ nor the simple baselines established reliable prediction of
future PPO-KL excursions.

---

## 4. ReViSQL / RLVR: φ does not transfer as a reliable PPO predictor

A separate August 2026 experiment tested `φ_jump` passively in a real RLVR/PPO
workload: ReViSQL text-to-SQL training on `Qwen/Qwen3-8B` through Tinker.
The purpose was deliberately different from Tests 5–7: instead of asking
whether φ detects an induced/current loss event, this experiment asked whether
a loss-only signal can **anticipate future PPO instability**.

### Setup

- 50 PPO training steps / 100 examples
- LoRA rank 16, learning rate `1e-4`
- group size 2, batch size 2
- φ was passive: no LR/rank/control intervention
- Tinker `loss:sum` was normalised by action tokens to obtain `loss_per_token`
- comparisons used the same loss stream: `|Δloss|`, rolling STD(5), rolling
  MAD(5), causal z-score and PPO-KL
- horizons inspected: `t+1`, `t+2`, `t+5`
- first 60% of the trajectory used for threshold calibration; final 40% used
  for chronological evaluation

### Result

The original one-sided

```text
φ_jump(t) = 0.8 φ_jump(t-1) + 0.2 max(0, Δloss(t))
```

did **not** reliably anticipate important future PPO-KL excursions. In the
late evaluation region φ decreased while several large KL values appeared:

| step | φ_jump | PPO-KL |
|---:|---:|---:|
| 43 | 0.01649 | 0.62240 |
| 44 | 0.01319 | 1.25181 |
| 45 | 0.01134 | 1.05359 |
| 48 | 0.00581 | 1.11044 |

The failure has a plausible structural component: PPO loss is signed, while
`max(0, Δloss)` is blind to negative loss changes. Step 48, for example, has
`loss_per_token ≈ -0.07561`, `|Δloss| ≈ 0.07561`, high `PPO-KL ≈ 1.11044`,
but low `φ_jump ≈ 0.00581`. This is evidence of one-sided blindness; because
loss and KL here are simultaneous, it is not itself evidence of prediction.

### Symmetric repair also fails to establish predictive value

The natural offline follow-up was

```text
φ_abs(t) = 0.8 φ_abs(t-1) + 0.2 |Δloss(t)|
```

which uses both signs of PPO loss change. It did not recover convincing
predictive performance. Exploratory ROC-AUC values for future KL-spike ranking
on the chronological evaluation segment were:

| horizon | φ_jump | φ_abs | STD(5) | |Δloss| | |z-score| |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.458 | 0.375 | 0.333 | 0.000 | 0.146 |
| t+2 | 0.578 | 0.533 | 0.411 | 0.333 | 0.222 |
| t+5 | 0.556 | 0.500 | 0.486 | 0.125 | 0.444 |

These are **exploratory numbers**, not population-performance estimates: the
held-out segment is small and contains few KL-spike events. They provide no
convincing evidence that φ_abs adds predictive value over simple loss-domain
statistics.

With a φ_abs threshold calibrated only on the first 60% (P90 ≈ 0.054), the
final 40% produced no φ_abs alarms while containing three KL-spike events
under the corresponding calibration rule: 0 true positives and 3 false
negatives in this small evaluation segment.

Reward prediction was likewise inconclusive/negative: φ_abs exploratory AUC
was approximately 0.466 (`t+1`), 0.292 (`t+2`) and 0.600 (`t+5`); the last
number is based on too few points/events to support a positive claim.

### Verdict

The experiment rejects the **setting-specific transfer claim** that
`EMA(max(0, Δloss))` is a generally useful predictor when applied directly to
signed PPO loss in ReViSQL/RLVR. The obvious symmetric repair
`EMA(|Δloss|)` is also not supported as a useful predictor by this run.

This does **not** contradict the earlier Tests 5–7 result: φ_jump remains an
observed near-instantaneous detector in those specific experiments. Detection
in one regime and prediction in a different PPO regime are different claims.
Nor does one 50-step trajectory prove that loss-only prediction is impossible
in PPO. It shows that these two φ formulations did not provide convincing
future-KL prediction here.

Full protocol, limitations and interpretation:
[`validation/revisql_ppo_phi_validation.md`](validation/revisql_ppo_phi_validation.md).

---

## 5. What the local controller experiments showed

Tests 8–14, on Qwen2.5-0.5B-Instruct, single T4, seed 11 unless stated.
These runs use φ_dev and the full controller.

**Sensor–actuator contamination.** The controller's actuator changes the
signal its own sensor reads. In a shock-free phase, φ_dev rose to 3.05 while
the FSM was in its HIGH state, against a maximum of 2.37 in a matched control
arm with the actuator disabled. Effects, all same-seed paired comparisons:

| | actuator off | actuator on |
|---|---|---|
| Shock/control φ ratio (seed 11) | 3.32× | 2.13× |
| Detection latency | 0.8 steps (mean, 5 seeds) | 4 steps |
| Detection at default `stable_window=30` | — | never fires |

**What this does not show.** Detection is *degraded*, not destroyed. At
`stable_window=10`, both `stress_k=1.5` and `6.0` detect at step 84 with no
false positive on the control arm — they are valid operating points. An
earlier draft of this README claimed no valid operating point exists with the
actuator enabled; that claim is refuted by Test 14 and has been removed.

**A control case bounds the effect.** AdaLoRA shows no comparable
contamination (0.4%, within noise, 3 seeds), with actuator activity verified
rather than assumed (effective rank 576 → 384). The condition therefore looks
narrow and mechanistic: contamination appears when the actuator acts on the
channel the sensor reads. Note that the AdaLoRA run is **not yet logged in
this repository** — the number above comes from the contamination note and
cannot currently be audited.

**Prior context.** In control theory this is closed-loop identification bias,
long known: under feedback without persistent excitation, the identified
process is biased toward the negative inverse of the controller. As far as we
can find, it has not been measured in adaptive PEFT. That framing is a
hypothesis about where this fits, not a claim of priority.

---

## 6. Known defects and open confounds

Listed because they materially limit every number in §5.

1. **The task saturates.** Tests 8–14 train on 20 fixed prompt/target pairs
   for 200 steps — ten epochs on the same material. Control-arm loss falls
   from 0.41 (steps 80–119) to 0.017 (steps 120–159). The "healthy" arm is
   therefore a *memorised* task, not healthy-but-active training, and the
   shock arm has unlearnable corrupted targets. Separation between the two is
   easier than the intended comparison, so the ratios and the 5/5 detection
   figures in `validation/` are upper bounds. This is the same confound that
   voided Test 7. **It has not yet been fixed.**
2. **Logged φ_dev is not the φ_dev the FSM acts on.**
   `OrbitalController.get_summary()["phi"]` returns the smoothed `phi_ema`,
   while the state machine compares the *raw* per-step φ against `μ + k·σ`.
   At one representative shock step the trace records 4.52 where the
   controller saw 17.8. Consequently, offline reanalysis of the published
   traces cannot reproduce the controller's decisions — including
   `validation/test12_reanalyze.py`, whose `μ + k·σ` rule looks identical to
   the FSM's but runs on a different quantity. Fix: expose `phi_raw`
   alongside `phi_ema`.
3. **`PhiMonitor` carries a dead gradient term** (`+ 0.01 * grad_norm`). Inert
   in all runs here, but it contradicts the loss-only claim on inspection.
4. **Statistical scope.** Tests 8, 10 and 14 are single-seed. Test 12 uses 5
   seeds. The control-arm false positive observed at k=0.1 has not been
   replicated across seeds.
5. **One model family, one task type, one shock type**, fixed shock onset at
   step 80. Latency figures will not survive randomised onset.
6. **ReViSQL/PPO statistical scope.** The new PPO result is one 50-step
   trajectory, one model/workload, two databases and few held-out KL-spike
   events. It rejects a strong transfer claim for the tested formulations; it
   does not establish a universal impossibility result for loss-only PPO
   monitoring.

---

## 7. Quick start

φ_jump needs only the loss stream:

```python
class PhiJumpMonitor:
    """φ_jump = EMA of upward loss jumps. Loss stream only."""
    def __init__(self, beta=0.8):
        self.beta, self.ema_jump, self.prev = beta, 0.0, None

    def update(self, loss):
        jump = 0.0 if self.prev is None else max(0.0, loss - self.prev)
        self.prev = loss
        self.ema_jump = self.beta * self.ema_jump + (1 - self.beta) * jump
        return self.ema_jump

phi = PhiJumpMonitor()
for batch in dataloader:
    loss = train_step(batch)
    if phi.update(loss.item()) >= 0.10:   # threshold from Test 5; see §2
        alert()
```

The 0.10 threshold was calibrated on Qwen3-8B via Tinker with this exact
formula. It does not transfer to φ_dev, to ReViSQL/PPO, to other models, or to
other loss scales without recalibration.

The controller remains available as a research artifact:

```python
from orbital_controller import setup_unified_lora
adapters, ctrl, opt = setup_unified_lora(model)   # returns three values
```

- `nested_lora.py` — NestedLoRA: single allocation, nested rank slicing
  (zero cold-start rank switching)
- `orbital_controller.py` — FSM controller (HIGH/BASE/LOW) with snapshot and
  rollback
- `controller.py` — orchestration

---

## 8. If you need adaptive-rank fine-tuning

Use AdaLoRA. That is the finding, not a disclaimer.

---

## License

Apache 2.0

## Contact

Simona Vargiu — independent researcher

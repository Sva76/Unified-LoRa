# Unified-LoRA → the φ signal

**Status (September 2026).** The original adaptive-controller hypothesis was
not supported by the reported comparisons. The remaining research question is
whether loss-only telemetry can detect or anticipate training failures, and
how control interventions affect that telemetry. ReViSQL/PPO has not provided
convincing predictive evidence for either φ formulation, including the
preliminary 300-step seed 101 run.

**Validation correction:** the historical Tinker Tests 5–7 used a misaligned
completion-loss mask. Their recorded separation and alarms are retained as
observations of that task, not validation of correctly aligned fine-tuning.
Code corrections, timing conventions and their implications are documented in
[`validation/corrections_2026_09.md`](validation/corrections_2026_09.md).
The corrected training scripts have **not** been rerun. Original logs remain
unchanged; the proposed non-memorizing v3 experiment is still outstanding.

This repository is a research artifact, not a library. Nothing here is
recommended for production fine-tuning.

---

## 1. The honest history

The project began as Unified-LoRA: an adaptive controller that observes a
stress signal during LoRA fine-tuning and adjusts adapter rank (NestedLoRA)
and learning rate (Orbital FSM controller). The hypothesis was that this would
beat established methods such as AdaLoRA on stability and quality.

**In the reported settings, it did not.** Multi-seed comparisons against
AdaLoRA (official PEFT implementation) yielded:

| Setting | AdaLoRA | Unified-LoRA | Fixed LoRA |
|---|---|---|---|
| Toy benchmark, final loss (3 seeds) | **0.198** | 0.226 | 0.227 |
| Toy benchmark, divergences (extreme regime) | **0/3** | 3/3 | 3/3 |
| MRPC, 50% label noise, F1 (3 seeds) | **0.518** | 0.358 | 0.374 |

The mechanism behind this performance gap is not established by these
comparisons. AdaLoRA differs in parameterization, regularization and rank
allocation; controlled ablations would be needed to attribute the gap to any
one of these components.

---

## 2. Two different signals share the name φ

This is a naming defect in earlier versions of this README, corrected here.
The repository contains **two distinct signals**, on scales roughly 170×
apart. They are not interchangeable and their thresholds do not transfer.

| | **φ_jump** (Tests 5–7, Tinker) | **φ_dev** (Tests 8–14, local controller) |
|---|---|---|
| Definition | `EMA_0.8(max(0, Δloss))` | Raw: `abs(loss − EMA(loss)) + 0.5·max(0, Δloss)`; historical trace: its EMA |
| Typical value, historical healthy arm | ~0.011 | ~1.9 |
| Threshold rule | Historical fixed 0.10; Test 7 also uses warmup and persistence | FSM: raw φ against `μ + k·σ` of raw history |
| Where implemented | `validation/test5–7*.py` | `orbital_controller.py` |

**φ_jump is loss-only.** Verified numerically: the φ values logged in
`validation/phi_lead_time_log.json` reproduce the pure loss-derived EMA to
within numerical precision. The historical optional gradient term was inert;
corrected Tests 5–7 use a shared monitor with a loss-only signature.
`OrbitalController.get_summary()` now exposes unrounded `phi_raw` and
`phi_ema`; the legacy `phi` key retains the rounded EMA.

---

## 3. Historical φ_jump observations on Qwen3-8B via Tinker

Historical thresholds were taken from Test 5 and reused in Tests 6–7.
The original scripts and criteria are available at commit
`72b4d08b7fbdcbb6d395db1460a4afd8d0d90884`. Current scripts correct the data
alignment and are diagnostic reruns, not a retrospective preregistration.

| Result | Value | Runs |
|---|---|---|
| Mean φ_jump in the historical healthy arm | ~0.011 | 3 |
| Healthy vs unstable separation (aggressive LR, no induced shock) | 371× | 3 |
| Blind classification, thresholds fixed in advance | 10/12 (extremes 10/10) | 12 |
| False alarms from step 60, three consecutive exceedances (300-step runs) | 0/6 | 6 |
| Retrospective φ-exceedance onset after collapse onset | 1–2 steps | 4 |
| Causally confirmed φ alarm after the same collapse onset | 3–4 steps | same 4 |

These figures describe the recorded historical task. The 371× figure is a
ratio of signal means between regimes, not superiority over another detector.
The 0/6 result excludes the first 60 steps and requires three exceedances:
a single-threshold alarm from step zero would fire in 4/6 healthy runs.

Test 7's predictive verdict remains **void** because of saturation, now
compounded by the identified token-alignment defect. Correcting the code does
not establish detection or prediction on a properly aligned, non-memorizing
training task. The proposed v3 stream has not yet been run.

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

| horizon | φ_jump | φ_abs | STD(5) | abs(Δloss) | abs(z-score) |
|---|---:|---:|---:|---:|---:|
| t+1 | 0.458 | 0.375 | 0.333 | 0.000 | 0.146 |
| t+2 | 0.578 | 0.533 | 0.411 | 0.333 | 0.222 |
| t+5 | 0.556 | 0.500 | 0.486 | 0.125 | 0.444 |

These are **historical exploratory numbers**, not recomputed with the corrected
baseline definitions and not population-performance estimates: the
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
arm with the actuator disabled. Historical comparisons are listed below. The ratio row is same-seed;
the latency row compares different detectors and different seed counts:

| | actuator off | actuator on |
|---|---|---|
| Shock/control φ ratio (seed 11) | 3.32× | 2.13× |
| Detection latency | 0.8 steps (mean, 5 seeds) | 4 steps |
| Detection at default `stable_window=30` | — | never fires |

**What this does not show.** The 0.8-versus-4-step comparison does not isolate
the actuator effect: the offline detector uses an EMA every step, while the
FSM uses raw φ, evaluation intervals and hysteresis. A matched detector with
separate LR/rank interventions is required for a causal latency claim. At
`stable_window=10`, both `stress_k=1.5` and `6.0` detect at step 84 with no
false positive on the control arm — they are valid operating points. An
earlier draft of this README claimed no valid operating point exists with the
actuator enabled; that claim is refuted by Test 14 and has been removed.

**Reported AdaLoRA comparison, pending auditable logs.** The contamination
note reports 0.4% (3 seeds) and effective rank changing from 576 to 384.
The underlying run is **not yet logged in this repository**, so its magnitude
and uncertainty cannot currently be checked. It does not establish that
AdaLoRA is generally immune to this effect.

**Prior context.** Feedback complicates causal attribution because the
controller changes the process it observes. Closed-loop identification is a
relevant theoretical context; mapping this experiment to a specific
identification-bias result requires further analysis. This is not a claim
of priority for the general feedback problem.

---

## 6. Known defects and open confounds

Listed because they materially limit every number in §5.

1. **The task saturates.** Tests 8–14 train on 20 fixed prompt/target pairs
   for 200 steps — ten epochs on the same material. Control-arm loss falls
   from 0.41 (steps 80–119) to 0.017 (steps 120–159). The "healthy" arm is
   therefore a *memorised* task, not healthy-but-active training, and the
   shock arm has unlearnable corrupted targets. Separation between the two is
   easier than the intended comparison, so the ratios and the 5/5 detection
   figures may overestimate performance on more realistic tasks; they are not
   formal statistical upper bounds. This is the same confound that
   voided Test 7. **It has not yet been fixed.**
2. **Historical signal mismatch.** Historical `phi` traces contain the EMA,
   while the FSM acts on raw φ. New summaries and trace collectors expose
   both `phi_raw` and `phi_ema`. The old logs have not been retroactively
   populated, and Test 12 reanalysis remains an offline EMA detector rather
   than an FSM replay.
3. **Corrected code, pending training validation.** Tests 5–7 now align
   completion tokens and loss weights and remove the unused gradient input.
   The ReViSQL derivation fixes rolling MAD and defines z-score from the
   preceding five losses. These changes are versioned; historical results
   and the original preregistration remain available. See the correction note.
4. **Statistical scope.** Tests 8, 10 and 14 are single-seed. Test 12 uses 5
   seeds. The control-arm false positive observed at k=0.1 has not been
   replicated across seeds.
5. **One model family, one task type, one shock type**, fixed shock onset at
   step 80. Transfer of latency figures to randomized onset has not been tested.
6. **ReViSQL/PPO statistical scope.** The 50-step exploratory trajectory and
   preliminary 300-step seed 101 result are not a completed multi-seed test.
   Seed 101 reports φ_jump AUROC 0.474/0.414/0.426 at t+1/t+2/t+5, using
   a descriptive within-run KL threshold. Raw seed 101 telemetry is not
   currently committed, so those scores cannot yet be independently recomputed. It rejects a strong transfer claim for the tested formulations; it
   does not establish a universal impossibility result for loss-only PPO
   monitoring.

---

## 7. Quick start

The monitor needs only the loss stream. This example implements the
**historical Test 7 alarm policy**, including its warmup and persistence;
it is not a validated policy for new tasks or corrected training data.

```python
from validation.phi_utils import PhiJumpAlarm

alarm = PhiJumpAlarm(threshold=0.10, detect_from=60, sustain=3)
for batch in dataloader:
    loss = train_step(batch)
    if alarm.update(loss.item()):
        alert()  # available on the third exceedance, never backdated
```

A non-finite loss raises an error here and must be handled as a separate
invalid/divergent observation. The threshold requires fresh calibration for
corrected Tests 5–7, φ_dev, PPO, other models and other loss scales.

Offline verification without API credits or a GPU:

```bash
python -m unittest discover -s tests -v
python validation/test7_reanalyze.py
python validation/test12_reanalyze.py
```

Test 7 reanalysis prints both retrospective and causal timestamps and writes
nothing unless `--output` is supplied. Corrected training outputs go to new,
timestamped directories under `validation/corrected_runs/`.

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

AdaLoRA is the stronger baseline in the reported comparisons. This repository
does not establish a universal ranking across tasks or training regimes.

---

## License

Apache 2.0

## Contact

Simona Vargiu — independent researcher

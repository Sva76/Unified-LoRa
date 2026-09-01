# Unified-LoRA Architecture — historical controller design

> **Status (August 2026): historical / falsified controller architecture.**
>
> This document describes the architecture that motivated the original
> Unified-LoRA experiments. It is retained for reproducibility and for
> understanding the code, **not as a validated or recommended design**.
> Multi-seed experiments subsequently showed that Unified-LoRA did not beat
> AdaLoRA on quality or stability. Later validation also separated two
> different signals that earlier documentation had both called φ, and a
> ReViSQL/RLVR experiment found that the loss-only φ did not transfer as a
> reliable predictor of future PPO-KL instability.
>
> For the current scientific status, read the main [README](../README.md) and
> the [`validation/`](../validation/) evidence bundle.

---

## 1. What the original architecture attempted

Unified-LoRA combined three ideas:

1. **NestedLoRA** — one maximum-rank adapter allocation with nested active
   slices, intended to avoid cold-start when switching rank.
2. **An FSM controller** — discrete operating states rather than continuously
   changing rank.
3. **A stress signal** — used by the controller to decide when to change
   operating state and learning behaviour.

The original hypothesis was that this combination could improve stability and
quality relative to fixed LoRA and established adaptive-rank methods.

That hypothesis was tested and **falsified**. AdaLoRA was superior in the
multi-seed comparison reported in the main README. The controller code remains
in the repository as a research artifact.

---

## 2. NestedLoRA

The NestedLoRA component uses a single matrix pair at maximum rank and changes
the active rank by slicing it:

```text
r4 ⊂ r8 ⊂ r16
```

The design goal was to preserve already learned low-rank components when
capacity expanded, rather than activating a completely independent adapter
pair.

This mechanism remains an architectural property of the implementation. It
should not be read as evidence that the overall Unified-LoRA controller is
superior: the full controller comparison was negative.

---

## 3. Historical FSM design

The controller was designed around discrete operating modes, with hysteresis,
snapshots and rollback logic. Earlier versions of this document used the
names `SINGLE`, `MULTI` and `MIRROR`; the current implementation and validation
materials should be treated as the source of truth for exact state names and
behaviour.

The conceptual design was:

```text
low stress       -> lower-capacity / efficiency state
baseline stress  -> normal training state
high stress      -> protective / higher-capacity state
```

The controller could also snapshot adapter state and use rollback logic after
a stress episode.

These were design hypotheses, not validated advantages. The subsequent
experiments found an important failure mode: **sensor–actuator contamination**.
When the controller changes the same training dynamics from which its stress
signal is computed, it changes the signal it is trying to interpret.

See [`validation/phi_contamination_note_EN.md`](../validation/phi_contamination_note_EN.md)
and the validation README for the measured results and confounds.

---

## 4. Important correction: there is no single validated “φ(t)” architecture

Earlier documentation conflated multiple generations of the stress signal.
That is now explicitly corrected.

### φ_jump — Tinker loss-only signal

The signal validated in Tests 5–7 is

```text
φ_jump(t) = EMA_0.8(max(0, Δloss(t)))
```

It requires only the loss stream. In the original Qwen3-8B/Tinker experiments
it behaved as a specific, near-instantaneous **detector** of instability. The
result does not establish prediction of future instability.

### φ_dev — local controller signal

The local controller experiments use a different signal, approximately

```text
φ_dev = EMA(abs(loss - EMA(loss)) + 0.5 * max(0, Δloss))
```

with adaptive thresholding over its history. Its numerical scale is very
different from φ_jump. Thresholds and values must not be transferred between
the two.

The validation campaign also found that the logged smoothed φ_dev is not
identical to the raw quantity on which the FSM acts. This is a known
reproducibility defect documented in the validation bundle.

### Historical composite C/E/S formulation

An early design described φ as a weighted combination of convergence,
gradient-direction coherence and gradient magnitude:

```text
φ = 0.3*C + 0.3*E + 0.4*S
```

That formulation belongs to the **historical design phase**. It is not the
loss-only φ_jump validated behind Tinker, and it should not be cited as the
validated φ definition.

Likewise, earlier claims that normalization made thresholds such as `0.3` and
`0.7` universal across models/tasks are **withdrawn**. The later evidence
shows explicitly that signal scales and thresholds are domain-dependent.

---

## 5. ReViSQL / PPO transfer test

In August 2026 the loss-only idea was tested passively in a different regime:
ReViSQL text-to-SQL RLVR/PPO on Qwen3-8B via Tinker.

The question was stricter than the original detector experiments: could φ
anticipate future PPO-KL excursions?

The answer in the tested 50-step trajectory was **no convincing evidence**.
The original one-sided φ_jump did not reliably anticipate future KL spikes.
Because PPO loss is signed, a natural symmetric variant was also tested
offline:

```text
φ_abs(t) = 0.8 φ_abs(t-1) + 0.2 |Δloss(t)|
```

That variant also failed to establish useful predictive value or a convincing
advantage over simple loss-domain baselines such as rolling standard
deviation, absolute loss change and a causal z-score.

This is a setting-specific negative transfer result, not proof that all
loss-only monitoring is impossible.

Full note:
[`validation/revisql_ppo_phi_validation.md`](../validation/revisql_ppo_phi_validation.md).

---

## 6. What survived the architecture campaign

The evidence supports a narrower set of conclusions than the original design
claimed:

- Nested rank slicing is an implemented mechanism, but it did not make the
  complete Unified-LoRA controller outperform AdaLoRA.
- The original adaptive controller hypothesis is falsified in the tested
  regimes.
- φ_jump showed useful near-instantaneous detection in its original Tinker
  experiments, but not demonstrated prediction.
- The local controller exposed a measurable sensor–actuator contamination
  problem.
- φ_jump did not transfer as a reliable future-KL predictor in the ReViSQL/PPO
  run; φ_abs did not rescue that predictive claim.
- Thresholds are **not universal** and must not be transferred between signal
  definitions, models or training regimes without calibration and independent
  validation.

If the goal is adaptive-rank fine-tuning rather than studying these failure
modes, the current repository conclusion is to use **AdaLoRA** rather than
Unified-LoRA.

---

## 7. Source of truth

For current claims and limitations, use:

- [main README](../README.md)
- [`validation/README.md`](../validation/README.md)
- [`validation/phi_validation_note_EN.md`](../validation/phi_validation_note_EN.md)
- [`validation/phi_contamination_note_EN.md`](../validation/phi_contamination_note_EN.md)
- [`validation/revisql_ppo_phi_validation.md`](../validation/revisql_ppo_phi_validation.md)

This document is intentionally retained as an architectural history rather
than deleted: the failed design and the corrections are part of the research
record.

# Measurement and weight verification

**Documentation clarification: 17 October 2026. No new experimental results.**

## Method and results

Unified-LoRA contains two distinct research stages: the original adaptive
rank/LR controller, and subsequent passive monitoring with the controller disabled.
The original controller did not outperform AdaLoRA in the reported multi-seed
comparisons. Historical Tinker Tests 5–7 had a completion-loss alignment defect;
corrected scripts require new training evidence. Their reported separation and
classification remain observations of the original task.

The October study asks whether per-example loss features help predict recovery
after an independently measured drop in generated-answer accuracy.

| Check | Recorded result | Supported conclusion |
|---|---|---|
| Seed 102 | Five forward evaluations at each of four states returned identical recorded mean and per-example losses | Repeatability at tested states and recorded precision |
| Seed 104 | Stability-ranking Spearman 0.689; diagnostic loss-level baseline 0.702 | Frozen criterion met on one held-out trajectory; added value unestablished |
| Seed 107 | Two recoveries and one episode without confirmed recovery within 50 updates; all five rules correct on two of three episodes | Prospective recording and exact offline replay documented; predictive superiority unestablished |

Full methods and evidence: [October report](../validation/monitoring_2026_10/README.md).
Historical [alignment corrections](../validation/corrections_2026_09.md) remain applicable.

## Loss repeatability is not weight identity

Repeated loss agreement does not establish bitwise identity of model weights,
identical training trajectories, or complete checkpoint restoration. Different
parameter states can produce the same recorded losses on a finite evaluation set.

The October bundles preserve protocols, logs, generated answers, checksums and
checkpoint/export references. They do not include model-weight files sufficient
for independent tensor-level verification. The cause of previous restoration
discrepancies remains unresolved; quantization or rounding is not established.

Generation is a separate check. Ten passes on seed-105 saved exports at states
280, 290 and 300 gave score ranges 26–29, 14–14 and 24–25 out of 32.
Text identity, per-example correctness and aggregate score agreement differ.

## Adapter rollback and effective weights

In the historical local controller, snapshots clone and restore LoRA factors
A/B, not optimizer or random number generator state. This is adapter rollback,
not complete training-state restoration. The rollback drift rule uses norms of
factor differences, not the effective adapter matrix.

The descriptive numbers in [weight_control_demo.py](../weight_control_demo.py)
are historical provenance, not an executable demonstration or evidence that
reduced weight drift prevents collapse.

## Requirements for a direct weight-restoration check

A future tensor-level check should compare the same declared parameter set
immediately before saving and immediately after reloading, without an
intervening optimizer update. Record tensor names, shapes, dtypes, exact
equality, maximum absolute difference and an explicitly defined relative
difference. Report frozen base weights and trainable adapter parameters
separately.

For NestedLoRA, distinguish changes in the factors A/B from changes in the
effective adapter matrix:

`U_t = (alpha / r_t) * A_t[:, :r_t] @ B_t[:r_t, :]`

Here U_t follows the implementation's row-vector convention; its transpose
is the corresponding update in PyTorch Linear weight layout. Rank and
scaling must be recorded because changing rank also changes alpha/r.
At rank zero, define U_t as zero, matching the disabled adapter.

Define effective-update drift as `||U_t - U_0||_F`, stepwise change as
`||U_t - U_(t-1)||_F`, and update norm as `||U_t||_F`. State the aggregation
rule across layers. These quantities are descriptive measurements:
smaller drift alone does not demonstrate better quality or collapse prevention.

A complete training-resumption check must additionally account for optimizer
state, scheduler state where applicable, random number generator state and
data position. Fixed-input loss comparisons provide a complementary functional
check, but cannot replace direct parameter comparisons.

This is a proposed verification protocol, not a description of a completed
experiment.

## Existing offline verification

```bash
python validation/monitoring_2026_10/verify_evidence.py
python validation/monitoring_2026_10/verify_evidence.py --replay
```

The first command checks consistency of archives, original JSON, embedded files
and exposed copies against published checksums. The second additionally
recomputes the archived seed-104 and seed-107 analysis results. Neither command
independently reproduces training, compares model-weight tensors, or validates
a recovery predictor. Checksum consistency does not independently authenticate
the origin of the observations.

These commands were not rerun as part of this documentation clarification.

## Preservation

Historical data, protocols and results remain unchanged. Dated notices qualify
retained wording. The pre-update repository is available at
[commit 3dc8034](https://github.com/Sva76/Unified-LoRa/tree/3dc8034ce1c7997e0b8b42f4b7b91666ef2a9e57).

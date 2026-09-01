# ReViSQL Prediction Challenge — preregistered protocol

**Status:** preregistered before new data collection · September 2026

## 1. Research question

Can a black-box observer predict future training instability using only signals exposed by a managed training API?

This protocol is intentionally broader than asking whether one specific φ formulation works. The goal is to compare several frozen loss-only candidates on new, unseen ReViSQL/Tinker PPO trajectories and determine whether any of them contain out-of-sample predictive information about future instability.

The previous 50-step ReViSQL trajectory is treated as exploratory and must not be reused for final model or threshold selection.

---

## 2. Primary target

The primary target is future PPO-KL instability.

For each step t, predictions are evaluated for horizons:

- t+1
- t+2
- t+5

A KL-spike event is defined causally from the calibration portion of the new data using a pre-specified quantile threshold. The default event threshold is the 90th percentile of PPO-KL in the calibration set only.

No threshold may be recomputed using the held-out test trajectories.

---

## 3. Secondary targets

Secondary outcomes are:

- future reward deterioration;
- future clip-fraction excursion, if the metric is available consistently;
- future PPO mean-ratio excursion, if the metric is available consistently.

These are secondary analyses and cannot rescue a failed primary endpoint.

---

## 4. Frozen candidate predictors

The following candidates are fixed before the new runs are collected.

### 4.1 Original one-sided φ

```text
φ_jump(t) = 0.8 φ_jump(t-1) + 0.2 max(0, Δloss_t)
```

### 4.2 Symmetric φ

```text
φ_abs(t) = 0.8 φ_abs(t-1) + 0.2 |Δloss_t|
```

### 4.3 Absolute loss jump

```text
|Δloss_t|
```

### 4.4 Rolling standard deviation

Causal rolling standard deviation of the last 5 normalized loss observations.

### 4.5 Rolling MAD

Causal rolling median absolute deviation of the last 5 normalized loss observations.

### 4.6 Causal absolute z-score

Absolute deviation of the current normalized loss from the mean of the preceding causal window, divided by that window's standard deviation. Zero-variance windows are handled deterministically and never produce infinities.

No candidate may have its formula, decay, window or sign convention modified after the new trajectories are inspected. Any modified signal becomes a new hypothesis requiring a new independent dataset.

---

## 5. Loss variable

Tinker PPO exposes a signed optimization loss. The analysis therefore uses the same normalized quantity defined in the exploratory ReViSQL work:

```text
loss_per_token = loss:sum / action_tokens
```

Exact-zero observations are retained. Signed-loss semantics must be preserved in the raw data.

---

## 6. Experimental regime

The initial confirmatory campaign should use the same broad ReViSQL/Tinker regime as the exploratory run so that the prediction question is isolated from unnecessary implementation changes:

- base model: Qwen3-8B;
- task family: ReViSQL / text-to-SQL RLVR;
- optimizer objective: PPO;
- LoRA rank: 16;
- learning rate: 1e-4;
- group size: 2;
- batch size: 2;
- φ is passive only: it must not alter LR, rank, sampling or optimization;
- no controller intervention;
- no post-hoc signal tuning.

The exact database/task sampling policy must be written to the run manifest before each run begins.

---

## 7. Sample-size target

The confirmatory target is:

- minimum 5 independent seeds;
- preferred 10 independent seeds;
- minimum 300 PPO steps per seed;
- preferred 500 PPO steps per seed.

The campaign should not be declared confirmatory if the held-out set contains too few KL-spike events for meaningful discrimination. Event counts must be reported explicitly.

---

## 8. Data separation

Runs are split by entire seed/trajectory, not by randomly shuffled individual steps.

Recommended structure:

- development/calibration: first 60% of seeds;
- held-out confirmation: final 40% of seeds.

If only 5 seeds are feasible, use 3 calibration seeds and 2 untouched test seeds. If 10 seeds are feasible, use 6 calibration seeds and 4 untouched test seeds.

The held-out trajectories must not be inspected for threshold selection, feature redesign or parameter tuning before the frozen analysis is executed.

---

## 9. Primary metrics

For every predictor and each horizon t+1, t+2 and t+5, report:

- AUROC;
- AUPRC;
- Spearman association with future PPO-KL;
- recall at the pre-specified alarm threshold;
- false-positive rate at that threshold;
- median lead time for detected events where applicable.

Uncertainty must be estimated by bootstrap over complete trajectories/seeds rather than treating every step as fully independent.

---

## 10. Confirmation criterion

A loss-only signal will be considered to have **confirmed predictive information for future PPO-KL in this setting** only if all of the following hold on the untouched test trajectories:

1. AUROC is at least 0.70 for at least one pre-specified horizon;
2. the bootstrap 95% confidence interval for that AUROC excludes 0.50;
3. the signal is not materially worse than the best frozen loss-only baseline;
4. the effect is not driven by a single seed;
5. the direction of the result is replicated across a majority of held-out seeds;
6. alarm performance is operationally non-trivial, with useful recall and a bounded false-positive rate.

For a claim that φ itself adds value, φ must additionally outperform or clearly complement the simple baselines. A positive result for another loss statistic is not counted as confirmation of φ.

Failure to meet these criteria is reported as non-confirmation, not tuned away.

---

## 11. Falsification rules

The following outcomes count against the predictive hypothesis:

- AUROC near chance across horizons;
- predictive performance below simple loss baselines;
- strong calibration performance that disappears on held-out seeds;
- apparent performance driven by one trajectory;
- alarms occurring predominantly at the same step as, rather than before, the instability event;
- substantial false-positive rates required to obtain useful recall.

No β, rolling window, KL threshold or event definition will be retuned on the held-out data to improve the result.

---

## 12. Interpretation rules

Possible outcomes are deliberately separated:

### Outcome A — φ confirmed

φ shows reproducible out-of-sample prediction of future PPO-KL and adds value over frozen baselines.

### Outcome B — another loss-only statistic wins

Black-box loss telemetry contains predictive information, but φ is not the best formulation.

### Outcome C — no loss-only signal predicts reliably

The experiment provides evidence that the exposed loss stream alone is insufficient for useful prediction in this ReViSQL/PPO regime. The next scientific question would then be whether richer exposed telemetry is required.

### Outcome D — insufficient events/data

No predictive conclusion is made. The campaign is extended without changing the frozen predictors.

---

## 13. Reproducibility requirements

Each run must save, at minimum:

```text
seed
step
loss_sum
action_tokens
loss_per_token
ppo_kl
ppo_clip_fraction
ppo_mean_ratio
reward
correct
```

The derived predictor columns must be generated offline from the raw telemetry by a single deterministic analysis script committed to the repository before the held-out results are inspected.

The run manifest must also record model, LoRA rank, LR, batch size, group size, dataset/database identifiers and code commit.

---

## 14. Relation to previous evidence

This protocol does not alter the existing conclusions:

- Unified-LoRA did not outperform AdaLoRA in the tested controller comparisons;
- φ_jump showed preliminary near-instantaneous detector behaviour in the earlier Qwen3-8B/Tinker campaign;
- the original lead-time experiment was void because of saturation/memorization;
- the exploratory 50-step ReViSQL/PPO run did not support transfer of φ_jump as a reliable future-KL predictor;
- φ_abs did not rescue the claim in that exploratory trajectory.

The purpose of this preregistration is to ensure that the next result, positive or negative, is interpretable without post-hoc rescue.

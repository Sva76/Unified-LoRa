# Validation corrections — 5 September 2026

## Research question and evidence status

The research question remains whether inexpensive, loss-only telemetry can
distinguish detection from prediction during post-training, including failure
regimes defined independently of the proposed signal. Negative findings remain
valid outcomes of that research agenda. These corrections improve the measurement
pipeline; they do not constitute new training evidence or confirm a predictor.

The original scripts and evidence are accessible at
[commit 72b4d08](https://github.com/Sva76/Unified-LoRa/tree/72b4d08b7fbdcbb6d395db1460a4afd8d0d90884).
All previously committed JSON logs, figures, the original PDF and the original
ReViSQL preregistration are preserved unchanged. Numbers cited in earlier
summaries remain historical observations, subject to the qualifications below.
No corrected Tinker/GPU training run has been performed as part of this update.

## 1. Completion-loss alignment in Tinker Tests 5–7

The old scripts supplied all prompt and completion tokens as model input,
shifted targets by one position, appended a duplicate of the final token, and
used a weight mask with `len(prompt_tokens)` leading zeros. That excludes the
prediction of the first completion token and supervises repetition of the last
completion token after it has already been observed.

For a one-token completion, the entire supervised task becomes that final
repetition. This is not the intended prompt-to-completion prediction task and
can contribute to rapid loss saturation; its quantitative effect requires a
corrected rerun.

The corrected construction in `phi_utils.completion_example` follows
[Tinker's next-token alignment](https://tinker-docs.thinkingmachines.ai/tinker/losses/):

```python
tokens = prompt_tokens + completion_tokens
inputs = tokens[:-1]
targets = tokens[1:]
weights = [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens)
```

Tests 5–7 now share that helper and a monitor accepting only loss. The historical
optional gradient term was inactive in the saved evidence; removing it does not
change the recomputed historical φ values. API keys are read from the environment
instead of being overwritten by a placeholder. Future outputs use separate,
timestamped `corrected_runs/` directories and include a protocol version;
Tests 5 and 6 also retain per-step traces in those new outputs.
Non-finite loss observations suppress their diagnostic verdicts; Test 6 marks
affected runs `invalid` and counts them as classification failures rather than
allowing a retained, low EMA to classify divergent telemetry as healthy.

**Effect on claims:** approximately 371× mean-signal separation and 10/12
classification describe the historical task. They do not establish performance
on correctly aligned fine-tuning. The classification labels are learning-rate
regimes, not an independently measured behavioral-safety endpoint. Historical
thresholds are retained only as diagnostic reference settings and require new
calibration. The 20-pair repeated dataset is still a confound; correcting the
alignment is not an implementation or execution of the proposed non-memorizing v3.

## 2. Alarm policy and available lead time

The historical zero-false-alarm result in Test 7 concerns six 300-step runs with
monitoring beginning at **step 60** and **three consecutive** φ values at or
above 0.10. The EMA is updated throughout the warmup. This does not validate a
single-threshold alarm beginning at step zero: that rule fires in four of the
six saved healthy runs during warmup.

The old sustained-event function returned the first step of an exceedance
sequence after seeing the whole sequence. That is a retrospective onset, not
the time the alarm becomes available. The revised offline accounting reports
both onset and confirmation, using actual step numbers and resetting across
gaps. For the four runs used in the old 1–2-step latency statement:

| Seed | Collapse onset | φ exceedance onset | φ alarm confirmed | Confirmed alarm minus collapse onset |
|---|---:|---:|---:|---:|
| 3 | 361 | 363 | 365 | 4 |
| 23 | 369 | 371 | 373 | 4 |
| 41 | 338 | 339 | 341 | 3 |
| 53 | 281 | 282 | 284 | 3 |

These collapse onsets are retrospective labels under the historical loss rule.
The new output also gives collapse confirmation time. Comparing confirmation
to a retrospective onset avoids mistaking confirmation delay for prediction.
The Test 7 predictive verdict remains **void** because of the task confounds.

The README now uses the shared `PhiJumpAlarm` with the historical warmup and
persistence. This example is not a validated default for new regimes. Non-finite
loss must be handled separately as invalid/divergent telemetry; it is never
treated as a successful healthy observation.

## 3. Controller telemetry and comparison limits

`OrbitalController.get_summary()` now exposes:

| Field | Meaning |
|---|---|
| `phi_raw` | Unrounded per-step value used by the FSM |
| `phi_ema` | Unrounded EMA of that value |
| `phi` | Historical compatibility alias: EMA rounded to six decimals |

Future trace collectors record both explicit quantities. The controller's
thresholds, state transitions, learning rates and ranks are unchanged. Old logs
have not been populated with inferred measurements.

Test 12's offline detector runs on the historical EMA trace at every step. The
FSM runs on raw φ and also has evaluation intervals and hysteresis. The reported
0.8-versus-4-step latencies therefore do not isolate an actuator-induced delay.
A causal contamination experiment needs the same sensor and detector in both
arms, with learning-rate and rank interventions separated. Task saturation,
fixed shock onset and missing AdaLoRA-control logs remain unresolved limitations.

## 4. Versioned ReViSQL baseline corrections

`revisql_prediction_analysis.py` now labels every output row with a derivation
version. The φ formulas, β, horizons and success threshold are unchanged.

| Baseline | `legacy-v1` | `corrected-v2` |
|---|---|---|
| Rolling MAD | Median of residuals computed against different rolling medians | Median absolute deviation from the single median of the current five-point window |
| Absolute z-score | Expanding statistics including the current loss | Statistics of the preceding five losses, excluding the current loss |

For losses 1 through 9, the final five-point window is [5, 6, 7, 8, 9]. Its MAD
is 1; the old implementation returned 2. The z-score's exact history length and
exclusion of the current observation are made explicit here as a correction and
clarification, not represented as an unchanged preregistration. Warmup and
zero-variance histories give an undefined z-score (`NaN`); downstream scoring
must disclose exclusions and compare predictors on a common eligible sample.

The original preregistration is not rewritten. Because seed 101 has already
been inspected, a comparison using `corrected-v2` is an amended analysis and
must be distinguished from the original frozen comparison. `legacy-v1` remains
available to inspect the prior code definitions; historical reported scores
are not silently regenerated or relabeled as corrected results.

Raw input checks reject empty files, duplicate or missing steps, non-finite
required telemetry, non-positive/non-integer token counts, and inconsistent
loss normalization. This prevents row shifts from masquerading as fixed-step
forecast horizons and prevents invalid losses from becoming zero innovations.
Output files must be new and distinct from the raw input.

The 300-step seed 101 summary remains preliminary and negative. Its raw JSONL is
not committed, so its AUROCs cannot be independently recomputed from the current
repository. The derivation script generates predictors and future targets; it
does not implement the full multi-seed calibration, metrics and uncertainty
pipeline required for a confirmatory result.

## 5. Reproducing the offline checks

`weight_control_demo.py` originally contained bare prose and could not be parsed
as Python. Its original text is now retained in a module docstring, with an
explicit notice that the controller claims are superseded. It remains a
historical narrative, not an executable training demonstration.

Two earlier prototypes, `experiments/stable_task_test.py` and
`experiments/stress_test_task_switch.py`, still contain malformed Python and
calls to obsolete controller interfaces. They are not used by the offline
checks below and have not been reconstructed or rerun in this correction.
The syntax check applies to the changed/new Python files, not every historical
prototype in the repository.

From the repository root, with NumPy and pandas installed:

```bash
python -m unittest discover -s tests -v
python validation/test7_reanalyze.py
python validation/test12_reanalyze.py
```

ReViSQL derivation, when an actual raw file is available:

```bash
python validation/revisql_prediction_analysis.py path/to/raw.jsonl --output-csv new-corrected.csv
python validation/revisql_prediction_analysis.py path/to/raw.jsonl --definition-version legacy-v1 --output-csv new-legacy.csv
```

These checks verify alignment, numerical definitions, historical φ accounting
and causal timestamps. They do not replace training replication, establish
behavioral safety, or improve the previously reported predictive performance.

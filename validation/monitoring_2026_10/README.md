# Measurement reliability and recovery monitoring

**Evidence update: 5 October 2026.** The original Unified-LoRA controller and
the proposed advantages of its φ signal remain unsupported in the reported
comparisons. This is a subsequent monitoring study, with the adaptive
controller disabled. It does not establish an improvement in fine-tuning,
validate a recovery predictor, or concern fluid dynamics.

The current question is narrower: **at the first observed performance drop,
does information about individual evaluation examples help predict recovery
within the next 50 updates better than simple alternatives?** The outcome is
defined from generated answers, independently of the loss features used by
the predictors. That separation does not by itself establish predictive value.

## Read this evidence in order

| Stage | Recorded outcome | What it supports |
|---|---|---|
| Per-example study, seed 102 | 400 valid steps; five forward evaluations at each of four checkpoints returned identical recorded losses | Repeatability in these checks and heterogeneous example dynamics |
| Initial generative endpoint pilot | Healthy arm stopped at 40 updates with `PROBE_FAILED`; stress arm not executed | The original requirement of identical generated text was not met |
| Healthy feasibility run, seed 105 | 400 updates; final score 31/32; spontaneous large score fluctuation | A non-saturated behavioral endpoint is available during part of this run |
| Recheck of saved exports at 280, 290 and 300 | Ten passes per export; score 14/32 on all ten passes at 290 | The low score was reproducible on that export |
| Calibration, seeds 103 and 106 | At D=2, four episodes per run: three recovered, one censored | Six complete recovery episodes for exploratory rule construction |
| Held-out stability check, seed 104 | Spearman 0.689; diagnostic loss-level baseline 0.702 | The prespecified stability-ranking criterion was met, without established added value over loss level |
| Prospective mechanics check, seed 107 | 450 updates; three episodes; two recoveries and one non-recovery within 50 updates | Decisions can be recorded prospectively and replayed exactly; predictive superiority remains unestablished |

The source archives, including incomplete attempts, are listed in
[`evidence_index.json`](evidence_index.json). They preserve the original
uploaded JSON bytes under lossless gzip compression and include 695 embedded
files with their original checksums. Selected analyses and protocols are also
available as readable, byte-identical copies in [`records/`](records/).

## Measurement corrections and limits

### Forward evaluation, generation and checkpoint restoration are different checks

In seed 102, five measurements at each of steps 50, 150, 250 and 350 had zero
recorded range, both for the mean and all 32 individual losses. This is an
observation about the tested states and recorded precision, not a proof of
universal GPU determinism or reproducible training trajectories.

Generated answers can change even on the same sampler. The seed-105 feasibility
probes had exact-match count ranges of zero or one answer, while the ten-pass
recheck at state 280 ranged from **26 to 29 correct answers**. Thus neither
one nor two answers is a global upper bound on scoring variation. Textual
agreement, per-example correctness and aggregate score agreement are distinct.
At the zero-score floor, identical aggregate scores provide little evidence
about useful endpoint repeatability.

Earlier checkpoint-restoration differences motivated these checks. Their
physical cause is unresolved: deterministic forward evaluations do not prove
that quantization or rounding during saving caused a mismatch. Nor do these
observations identify the source of divergence between separately trained runs.

The earlier paired-run failure endpoint could trigger between healthy runs.
Consequently, differences against a parallel trajectory cannot, under that
criterion, be taken as evidence of a stress-induced failure. The earlier
reported crossing times (64, 66, 74 and 87) are contextual reports from the
preceding campaign, not newly recomputed results of this release. The full raw
paired-run campaign is not among these seven bundles.

### Heterogeneous examples do not yet establish better sensors

The seed-102 analysis reports late-window residual MAD of about 0.000316 for
the mean loss and 0.0000289 for the median individual example, a ratio of about
10.9. The correlation between example loss level and its reported variability
is about 0.884. These describe a changing training trajectory, not independent
measurement noise on a fixed state. They do not demonstrate an elevenfold
increase in detector sensitivity. The original analysis and its rule are
preserved [here](records/per_example_102/analisi_peresempio.json).

The subsequent shared evaluation set separates the training seed from the
identity of the 32 control examples. Example indices must not be transferred
from an earlier seed-dependent evaluation set merely because the numbers match.
The seed-102 residual-MAD definition also differs from the consecutive-change
definition used for the seed-104 stability check.

### A reproducible score drop is not a demonstrated stress effect

In the healthy seed-105 trajectory, ordinary scores at states 280, 290 and
300 were 28, 14 and 25 out of 32. In ten rechecks of each saved export:

| State | Observed score range | Median | Passes |
|---|---:|---:|---:|
| 280 | 26–29 | 28 | 10 |
| 290 | 14–14 | 14 | 10 |
| 300 | 24–25 | 24 | 10 |

The state-290 score was separated from both neighboring observed ranges.
These are empirical ranges, not confidence intervals or global noise bounds.
The observation occurred without imposed stress and does not identify its
cause. It is retained in `evidence/recheck_290.json.gz`.

## Seed 104 — stability ranking

The rule used seeds **103 and 106** for calibration and the same 32 examples
for the held-out run. The frozen window was states **300–399**, giving 99
consecutive differences. For each example, stability was defined as the
median absolute difference in its loss. Calibration ranks were averaged
across the two runs; eight examples were selected before examining seed 104.

The primary Spearman correlation was **0.689238**, above the prespecified
practical criterion of **0.50**. Seven of the selected eight examples were
also among the holdout's eight least variable examples. This is a descriptive
replication on one new trajectory, not a significance test or population
performance estimate.

A diagnostic ranking based on calibration loss level correlated **0.702430**
with holdout variability. This comparable value does not prove equivalence,
but it prevents claiming an established advantage for the stability ranking.
The selected-to-all median variability ratio of 0.034 is a comparison of
per-example statistics, not the variance of an aggregated detector or its
signal-to-noise gain.

Sources: [result](records/stability_104/esito104.json),
[frozen rule](records/stability_104/rule104.json),
[original protocol](records/stability_104/LEGGIMI.md).
Seed 104 is now observed and cannot serve as a fresh blind test for a revised rule.

## Seed 107 — prospective recovery decisions

This was one healthy run using Qwen3-8B, fixed LoRA rank 8, learning rate
0.0001 and batch size 8. It completed **450 updates**, recorded loss at
451 states and made **46 regular generative evaluations**. Each regular
score is the median of three passes over 32 examples on the same export.
Extra loss-triggered diagnostic evaluations are excluded from episode
definitions and outcome labels. The archived 105 feasibility scores use a
different repetition schedule and are exploratory context, not interchangeable
calibration observations.

### Frozen episode and prediction rules

- A candidate episode begins at an eligible regular evaluation when the score
  is at least **D=2** below the median of the previous three regular scores.
  That reference must be at least eight correct answers.
- The decision is made using data through that state, before the optimizer
  update leading to the next state. No new episode is admitted through
  onset+50, inclusive; the last eligible onset is state 400.
- Recovery requires two consecutive future regular scores at least
  reference−1, both within 50 updates. Otherwise, with a complete observation
  window, the label is `NO_CONFIRMED_RECOVERY_WITHIN_50`.
- An incomplete window is `CENSORED`. Numerical failure is a separate outcome.
  Non-recovery within 50 updates is never labeled permanent damage.
- Five alternatives use the same decision time: selected stable examples,
  mean loss, mean-loss variability, selected low-loss examples, and always
  predicting recovery. The first four use the same 20-state feature window.
- Thresholds were the maxima of the respective features on the six complete
  recovered episodes in 103/106. There were no complete non-recoveries in that
  calibration set. These are exploratory one-class heuristics, not a fitted
  or validated two-class predictor. Stable and low-loss groups share seven
  of eight examples. The variability threshold is zero and was not revised
  after examining seed 107.

The exact definitions and numerical thresholds are in the
[original frozen protocol](records/mechanics_107/PROTOCOLLO_107.md) and
[rule](records/mechanics_107/rule107.json). Recorded timestamps and hashes
support audit of the supplied files; this later publication is not an
independent timestamped public preregistration of the earlier runs.

### Observed episodes

| Decision state | Reference | Current score | Outcome within 50 updates | Confirmation state |
|---|---:|---:|---|---:|
| 170 | 22 | 20 | Recovered | 220 |
| 230 | 25 | 21 | Recovered | 250 |
| 320 | 29 | 27 | No confirmed recovery | — |

| Rule | Correct recovery predictions | Correct non-recovery predictions | False non-recovery predictions | Missed non-recoveries |
|---|---:|---:|---:|---:|
| Stable examples | 2 | 0 | 0 | 1 |
| Mean loss | 2 | 0 | 0 | 1 |
| Mean-loss variability | 1 | 1 | 1 | 0 |
| Low-loss examples | 2 | 0 | 0 | 1 |
| Always predict recovery | 2 | 0 | 0 | 1 |

All five rules were correct on two of three episodes, with different errors
for mean-loss variability. This is not evidence of equal performance in a
population. Episodes within one run are not independent replications.
The observed non-recovery occurred **without stress**: the earlier expectation
that this category could only come from stress was too strong.

The online decisions, input-prefix hashes and decision chain reproduce
exactly offline. The recorded verdict is `VERIFICA_MECCANICA_COMPLETATA`;
`predictor_validated` remains false. The result is available
[in full](records/mechanics_107/esito107.json).

The bundle also retains an initial technical attempt under
`tentativo1_errore_setup/`, which stopped with zero optimizer updates before
collecting observations, and the note concerning its manual restart. The
cause of that setup error is not established by the retained error record.

## What remains to be done

The original seed-107 protocol records a proposed separate development pilot:
seeds 108/109, each with LR multipliers 3, 10 and 30 for updates 217–236,
followed by restoration of the healthy LR. It reserves seeds 200–209 for
subsequent verification. **This release does not execute that pilot or those
verification runs**, and does not claim that the proposed stress range will
produce both outcome classes.

After development, the comparative metric, minimum useful effect and number
of independent runs must be fixed before examining verification outcomes.
Useful evidence would be an incremental predictive advantage over the simple
baselines, with uncertainty assessed across independent runs. Independent
methodological review and replication are still needed.

A second possible research direction is a documented single-case study of
AI-assisted research without formal specialist training. Its potential value
lies in examining the recorded decisions, errors and corrections. No priority
claim, generalization to other users, or proof of a new scientific method is
made here. The author led experimental design and evaluation; implementation
was developed with AI assistance and would benefit from independent technical
review.

## Offline verification and access to the full records

From the repository root, using Python 3 and its standard library:

```bash
python validation/monitoring_2026_10/verify_evidence.py
python validation/monitoring_2026_10/verify_evidence.py --replay
```

The first command checks archive checksums, the original JSON checksums,
all embedded file checksums, and all exposed copies. The second additionally
extracts 104/107 to temporary directories and recomputes their results with
the archived analysis code; it makes no Tinker calls and consumes no API
credits. This is reproducibility of analysis on existing data, not independent
training replication.

To inspect all embedded protocols, datasets, generated answers and logs:

```bash
python validation/monitoring_2026_10/verify_evidence.py --extract /tmp/unified-lora-evidence
```

The destination must not already exist. Archives include checkpoint/export
references and metadata, not model-weight files or a promise that remote
checkpoints remain available. Original collection code is preserved for audit;
running live training requires separate API access and may incur costs.

## Preservation policy

No pre-existing experiment, log, figure, PDF, protocol or source file is
deleted or replaced by this evidence release. The two existing README files
receive dated notices above their previous content. Their earlier text remains
verbatim, and the complete pre-update repository is available at
[commit 26966fe](https://github.com/Sva76/Unified-LoRa/tree/26966fe9aad40d4f77e8b23ecf27510281b93acb).
The new evidence does not resolve all outstanding historical reruns or missing
ReViSQL logs described there.

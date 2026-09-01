# Black-box observability of PPO instability — ReViSQL / Tinker

## Research question

Can a black-box observer predict future training instability (PPO-KL divergence) using only telemetry exposed by a managed training API?

This is deliberately broader than asking whether phi wins. The experiment tests how much anticipatory information is observable from the external loss stream under ReViSQL/Tinker PPO training.

## Separation from the exploratory diagnostic

The earlier 50-step ReViSQL trajectory was an **exploratory diagnostic**. It showed that the original one-sided phi did not transfer convincingly to signed PPO loss and motivated a stricter independent test.

The experiment below is a **preregistered observational prediction challenge**. Predictor definitions, horizons and the success criterion are frozen rather than adapted to the result.

## Frozen design

- Model: `Qwen/Qwen3-8B`
- Workload: ReViSQL RLVR / text-to-SQL
- Backend: Tinker
- Optimisation loss: PPO
- Input stream: `loss_per_token`
- Primary target: future PPO-KL
- Horizons: `t+1`, `t+2`, `t+5`
- Success criterion: AUROC >= 0.70
- Candidate loss-only descriptors: `phi_jump`, `phi_abs`, rolling STD/MAD, causal z-score and simple temporal descriptors
- Measurement mode: passive observation only
- Orbital/NestedLoRA controller: OFF

Frozen original signal:

```text
phi_jump(t) = 0.8 * phi_jump(t-1) + 0.2 * max(0, delta_loss(t))
```

The controller is disabled because an active controller changes the trajectory read by its own sensor, creating sensor-actuator / closed-loop contamination.

## Preliminary independent result — seed 101

Run: `qwen3_8b_ppo_seed101`  
Data-order seed: 101  
Length: 300 steps (0–299)  
Run status: `completed_raw_recoverable`  
Analysis status: preliminary single seed; **not** the final confirmatory result.

| predictor | target | AUROC | Spearman with future PPO-KL | threshold met? |
|---|---|---:|---:|---|
| `phi_jump` | t+1 | **0.474** | **-0.023** | No |
| `phi_jump` | t+2 | **0.414** | **-0.075** | No |
| `phi_jump` | t+5 | **0.426** | **-0.091** | No |
| `phi_abs` | t+1 | 0.453 | — | No |
| `phi_abs` | t+2 | 0.444 | — | No |
| `phi_abs` | t+5 | 0.428 | — | No |

The preregistered AUROC threshold of 0.70 was not reached. Seed 101 therefore provides no preliminary evidence that `phi_jump` predicts future PPO-KL in this setting.

This is not a reason to repair or tune phi. The negative outcome is part of the test.

## Interpretation

The contribution under test is **observability**, not superiority over AdaLoRA and not a requirement that phi outperform elementary statistics.

Three outcomes are scientifically informative:

1. `phi_jump` predicts future PPO-KL robustly across independent runs;
2. another frozen elementary loss descriptor predicts better, in which case that descriptor is the result;
3. none predicts robustly, supporting the narrower empirical conclusion that this external telemetry is insufficient for anticipatory warning of this instability type in the tested regime.

The third outcome would be an observability-limit result for managed-training telemetry, not evidence that all possible PPO monitoring is impossible.

## Multi-seed continuation

The preregistered continuation uses independent data-order seeds with at least five runs planned and ten preferred. Predictor definitions, horizons and the success threshold are not to be changed in response to seed 101.

Current run-status table: [`revisql_runs/multiseed_status.md`](revisql_runs/multiseed_status.md).

Machine-readable seed-101 result: [`revisql_seed101_result.json`](revisql_seed101_result.json).

Frozen deterministic derivation code: [`revisql_prediction_analysis.py`](revisql_prediction_analysis.py).

The data-order seed controls the ReViSQL dataset permutation. It should not be described as full control of Tinker's remote internal RNG.

## Methodological rule

A negative result must remain negative. Do not change beta, sign convention, rolling window, horizon, event definition or success threshold after inspecting an independent run and then present the modified analysis as confirmatory. Any materially changed predictor is a new hypothesis requiring new independent data.

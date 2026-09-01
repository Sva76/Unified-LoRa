# ReViSQL/Tinker preregistered multi-seed status

This table is declared before completion of the remaining independent runs. The purpose is to make run inclusion visible in advance and reduce selection flexibility.

**Research question:** how much anticipatory information about future PPO-KL instability is observable from black-box, loss-only managed-training telemetry?

**Frozen primary horizons:** t+1, t+2, t+5  
**Preregistered success threshold:** AUROC >= 0.70  
**Measurement mode:** passive observation only; Orbital/NestedLoRA controller OFF  
**Model/workload:** Qwen3-8B, ReViSQL RLVR/PPO through Tinker

| run | data-order seed | steps | phi_jump AUC t+1 | t+2 | t+5 | best frozen baseline | >=0.70? | controller |
|---|---:|---:|---:|---:|---:|---|---|---|
| seed101 | 101 | 300 | 0.474 | 0.414 | 0.426 | pending full frozen-baseline comparison | No | OFF |
| seed202 | 202 | — | — | — | — | — | — | OFF |
| seed303 | 303 | — | — | — | — | — | — | OFF |
| seed404 | 404 | — | — | — | — | — | — | OFF |
| seed505 | 505 | — | — | — | — | — | — | OFF |

The labels above are **data-order seeds**, not claims of full control over Tinker's remote internal RNG.

Seed 101 is a preliminary single-seed result, not the final confirmatory result. No predictor, beta, window, sign convention, horizon, or success threshold should be changed in response to its outcome.

If phi_jump fails but a predefined elementary loss statistic performs better, that is the result. If none of the frozen loss-only descriptors predicts PPO-KL robustly across independent runs, the supported conclusion is an observability limit for this telemetry/regime, not a need to tune phi post hoc.

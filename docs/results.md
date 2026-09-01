# Results

**Status: August 2026.** Earlier versions of this page reported single-run
results (including a noise-sweep table) that were superseded by the multi-seed
falsification campaign. The current evidence now includes three distinct
validation regimes; detection and prediction claims are kept separate.

## Current results

- **Controller (Unified-LoRA vs AdaLoRA): falsified.** AdaLoRA is superior on
  quality and stability in the tested multi-seed regimes. See the summary
  table in the main [README](../README.md).

- **φ_jump detection on Qwen3-8B/Tinker: preliminary positive evidence in the
  original setting.** With thresholds fixed from Test 5 and reused in Tests
  6–7, φ_jump showed strong healthy/unstable separation, 10/12 blind
  classification (10/10 at the extremes), zero alarms across six 300-step
  healthy runs, and near-instantaneous detection around collapse. This is a
  **detector result, not a predictor result**. The original Test 7 lead-time
  verdict is void because the task saturated/memorised.

- **Local controller / φ_dev: sensor–actuator contamination observed.** The
  controller changes the training dynamics that its own sensor reads,
  degrading separation and latency. Detection is degraded rather than
  universally destroyed; valid operating points exist, but the shipped
  default `stable_window=30` failed in the tested setting. Several important
  confounds and missing logs remain documented in the validation bundle.

- **ReViSQL/RLVR/PPO transfer test: negative predictive result.** In an August
  2026 50-step Qwen3-8B ReViSQL PPO trajectory, the original one-sided φ_jump
  did not reliably anticipate future PPO-KL spikes. A symmetric loss-only
  follow-up, `φ_abs = EMA_0.8(|Δloss|)`, also failed to establish convincing
  predictive value or an advantage over simple loss-domain baselines. This
  rejects the strong setting-specific claim that the original φ formulation
  transfers directly as a general PPO instability predictor; it is not a
  universal impossibility result for loss-only monitoring.

## Evidence

The detailed evidence, limitations and reproducibility materials are in
[`validation/`](../validation/):

- [`validation/README.md`](../validation/README.md) — map of all three campaigns
- [`validation/phi_validation_note_EN.md`](../validation/phi_validation_note_EN.md) — original Tinker detector campaign
- [`validation/phi_contamination_note_EN.md`](../validation/phi_contamination_note_EN.md) — controller contamination analysis
- [`validation/revisql_ppo_phi_validation.md`](../validation/revisql_ppo_phi_validation.md) — ReViSQL/PPO prediction and φ_abs test

The repository's current practical conclusion remains: **use AdaLoRA for
adaptive-rank fine-tuning; treat Unified-LoRA and φ as research artifacts with
explicitly bounded claims.**

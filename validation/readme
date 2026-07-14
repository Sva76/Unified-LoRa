# validation/ — φ signal validation bundle

Preliminary empirical validation of φ, a gradient-free training-stress signal
(EMA of upward loss jumps), on Qwen3-8B via the Tinker API.

**Start here:** `phi_validation_note_EN.pdf` — 3-page technical note covering
the supported (preliminary) claims, the falsified claims, thresholds, seed
protocols, baseline detector comparison, and open questions.

## Contents

| File | What it is |
|---|---|
| `phi_validation_note_EN.pdf` | Technical note (read first) |
| `phi_natural_stress_log.json` | Test 5 raw results — healthy vs aggressive LR, 3 seeds, 371× ± 40 separation |
| `phi_blind_diagnostic.json` | Test 6 raw results — blind classification, pre-fixed thresholds, 10/12 (extremes 10/10) |
| `phi_lead_time_log.json` | Test 7 raw traces — per-step loss/φ/LR for all 14 runs (8 LR-ramp + 6 healthy) |
| `test5_natural_stress.py` | Test 5 script (Tinker) |
| `test6_blind_diagnostic.py` | Test 6 script (Tinker) |
| `test7_phi_lead_time_v2.py` | Test 7 script (Tinker) — pre-registered lead-time protocol |
| `test7_reanalyze.py` | Offline reanalysis of Test 7 logs (no credits needed) |

## Key facts

- **Threshold protocol:** φ alarm threshold 0.10, fixed from Test 5 data,
  reused unchanged in Tests 6–7. Verdict criteria (including failure
  conditions) are pre-registered inside each script.
- **Supported (preliminary):** φ separates healthy vs unstable regimes
  (371×), classifies blind at the extremes (10/10), produced 0 false alarms
  on 6 healthy runs, and fires within 1–2 steps of collapse.
- **Falsified:** the original adaptive controller does not outperform AdaLoRA
  (see note, §4.1).
- **Void (confound):** Test 7's predictive verdict — the task saturated to
  zero loss (full memorization), so no loss-derived signal could show
  precursors. Documented in the note (§4.2); corrected protocol (v3,
  non-memorizable stream) declared before execution.

Scripts require a Tinker API key (set `TINKER_API_KEY`); the detection and
reanalysis logic runs offline on the JSON logs.

# Bundle for `validation/` — local controller experiments (tests 8–13)

Drop these files into `validation/` alongside the existing Tinker bundle.
The note to read first is `phi_contamination_note_EN.md`.

## Scripts

| File | What it tests | Log it produces |
|---|---|---|
| `test8_full_controller.py` | Full Unified-LoRA controller on a real LLM, first run | `log_controller_full.json` |
| `test9_control_vs_shock.py` | Paired control vs shock; threshold sensitivity | `log_control_vs_shock.json` |
| `test10_fsm_latency.py` | Why LOW never fires: FSM hysteresis (`stable_window`) | `log_fsm_latency.json` |
| `test11_calibration_robustness.py` | Multi-seed replication of the discriminating cell | `log_calibration.json` |
| `test12_sensor_actuator_separation.py` | **Main result:** actuator off → clean φ traces + detector comparison | `traces_phi_clean.json` |
| `test12_reanalyze.py` | Offline reanalysis of the traces (no GPU, no rerun) | — |
| `test13_adalora_contamination.py` | **Counter-case:** does AdaLoRA contaminate its own signal? | `adalora_contamination.json` |
| `test13_verify_actuator.py` | Confirms AdaLoRA's actuator was actually active | — |

## Requirements

Tests 8–12 need a GPU runtime (the model trains locally). Test 13 also needs
`peft`. No API keys required — unlike the Tinker bundle, everything runs on
free Colab.

`test12_reanalyze.py` runs offline on saved traces: useful for evaluating new
detection rules without re-running any training.

## Reading order

1. `phi_contamination_note_EN.md` — claims, protocol, limitations
2. `test12_*` — the main result (sensor/actuator separation)
3. `test13_*` — the counter-case that bounds how far the claim generalises

## Note on ordering (PEFT)

In `test13`, `update_and_allocate` is called **after** `optimizer.step()`, per
PEFT's documented order. Reversing it leaves the actuator inert — AdamW
momentum restores the just-masked entries — and yields a null result that looks
like a genuine negative. `test13_verify_actuator.py` exists precisely to rule
this out: it confirms effective rank drops 576 → 384 (33%) when the actuator
is on, and stays at 576 when off.

#!/usr/bin/env python3
"""Versioned deterministic derivation for the ReViSQL/Tinker prediction challenge.

This script derives predefined loss-only predictors and future PPO-KL targets
from a raw revisql_phi_raw_v1 JSONL file. It does not tune parameters from the
observed outcome and does not modify the raw file.

Primary horizons: t+1, t+2, t+5
Primary success threshold: AUROC >= 0.70
phi_jump(t) = 0.8*phi_jump(t-1) + 0.2*max(0, delta_loss_t)
phi_abs(t)  = 0.8*phi_abs(t-1)  + 0.2*abs(delta_loss_t)
Corrected definitions: rolling windows contain 5 observations; the z-score
uses the PRECEDING 5 observations. Use legacy-v1 to inspect the old baseline
definitions. See corrections_2026_09.md; corrections are not new evidence.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

HORIZONS = (1, 2, 5)
BETA = 0.8
WINDOW = 5
SUCCESS_AUROC = 0.70
DEFINITION_VERSIONS = ("corrected-v2", "legacy-v1")
EXPECTED = [
    "step", "loss_sum", "action_tokens", "loss_per_token", "ppo_kl",
    "ppo_clip_fraction", "ppo_mean_ratio", "reward", "correct",
    "optim_entropy", "sample_train_kl_v1", "sample_train_kl_v2",
    "clock_cycle", "progress_done_frac",
]


def load_raw(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            obj = json.loads(line)
            missing = [k for k in EXPECTED if k not in obj]
            if missing:
                raise ValueError(f"line {line_no}: missing keys {missing}")
            rows.append(obj)
    if not rows:
        raise ValueError("raw file contains no observations")
    df = pd.DataFrame(rows)
    steps = pd.to_numeric(df["step"], errors="raise")
    if not np.isfinite(steps).all() or (steps < 0).any() or (steps % 1 != 0).any():
        raise ValueError("steps must be finite non-negative integers")
    df["step"] = steps.astype("int64")
    if df["step"].duplicated().any():
        raise ValueError("duplicate steps")
    df = df.sort_values("step").reset_index(drop=True)
    if (df["step"].diff().dropna() != 1).any():
        raise ValueError("missing steps: row shifts would not represent t+1/t+2/t+5")
    for name in ("loss_sum", "action_tokens", "loss_per_token", "ppo_kl"):
        df[name] = pd.to_numeric(df[name], errors="raise")
        if not np.isfinite(df[name]).all():
            raise ValueError(f"non-finite {name}: retain raw data but do not score silently")
    if (df["action_tokens"] <= 0).any() or (df["action_tokens"] % 1 != 0).any():
        raise ValueError("action_tokens must be positive integers")
    if not np.allclose(df.loss_sum / df.action_tokens, df.loss_per_token,
                       rtol=1e-6, atol=1e-9):
        raise ValueError("loss_per_token does not match loss_sum / action_tokens")
    return df


def ema_signal(values: np.ndarray, absolute: bool) -> np.ndarray:
    if not np.isfinite(values).all():
        raise ValueError("EMA input must contain only finite losses")
    out = np.zeros(len(values), dtype=float)
    state = 0.0
    prev = None
    for i, x in enumerate(values):
        delta = 0.0 if prev is None else float(x - prev)
        innovation = abs(delta) if absolute else max(0.0, delta)
        state = BETA * state + (1.0 - BETA) * innovation
        out[i] = state
        prev = x
    return out


def derive(df: pd.DataFrame, definition_version: str = "corrected-v2") -> pd.DataFrame:
    if definition_version not in DEFINITION_VERSIONS:
        raise ValueError(f"unknown definition version: {definition_version}")
    if df.empty or (df["step"].diff().dropna() != 1).any():
        raise ValueError("derive requires non-empty data in consecutive step order")
    x = pd.to_numeric(df["loss_per_token"], errors="raise")
    d = x.diff()
    df = df.copy()
    df["phi_jump"] = ema_signal(x.to_numpy(float), absolute=False)
    df["phi_abs"] = ema_signal(x.to_numpy(float), absolute=True)
    df["abs_delta_loss"] = d.abs()
    df["rolling_std5"] = x.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
    if definition_version == "legacy-v1":
        med = x.rolling(WINDOW, min_periods=WINDOW).median()
        df["rolling_mad5"] = (x - med).abs().rolling(WINDOW, min_periods=WINDOW).median()
        exp_mean = x.expanding(min_periods=2).mean()
        exp_std = x.expanding(min_periods=2).std(ddof=0).replace(0, np.nan)
        df["abs_causal_z"] = ((x - exp_mean) / exp_std).abs()
    else:
        # Every residual uses the SAME median of this five-observation window.
        df["rolling_mad5"] = x.rolling(WINDOW, min_periods=WINDOW).apply(
            lambda a: float(np.median(np.abs(a - np.median(a)))), raw=True)
        history = x.shift(1).rolling(WINDOW, min_periods=WINDOW)
        scale = history.std(ddof=0).replace(0, np.nan)
        # A zero-variance window has undefined z, recorded as NaN rather than
        # infinity or an invented high/low risk score. Warmup is likewise NaN.
        df["abs_causal_z"] = ((x - history.mean()) / scale).abs()
    for h in HORIZONS:
        df[f"future_ppo_kl_t{h}"] = pd.to_numeric(df["ppo_kl"], errors="coerce").shift(-h)
    df["derivation_version"] = definition_version
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_jsonl", type=Path)
    ap.add_argument("--output-csv", type=Path)
    ap.add_argument("--definition-version", choices=DEFINITION_VERSIONS,
                    default="corrected-v2")
    args = ap.parse_args()

    if args.output_csv and (args.output_csv.resolve() == args.raw_jsonl.resolve()
                            or args.output_csv.exists()):
        ap.error("output must be a new file, distinct from the raw input")
    df = derive(load_raw(args.raw_jsonl), args.definition_version)
    print(json.dumps({
        "rows": int(len(df)),
        "step_min": int(df.step.min()),
        "step_max": int(df.step.max()),
        "beta": BETA,
        "window": WINDOW,
        "horizons": HORIZONS,
        "success_auroc": SUCCESS_AUROC,
        "derivation_version": args.definition_version,
        "undefined_abs_causal_z": int(df.abs_causal_z.isna().sum()),
        "note": "Target labels/AUROC require the preregistered calibration rule and whole-seed split; this derivation script does not retune them."
    }, indent=2))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False, mode="x")
        print(f"derived CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Frozen deterministic derivation for the ReViSQL/Tinker prediction challenge.

This script derives predefined loss-only predictors and future PPO-KL targets
from a raw revisql_phi_raw_v1 JSONL file. It does not tune parameters from the
observed outcome and does not modify the raw file.

Primary horizons: t+1, t+2, t+5
Primary success threshold: AUROC >= 0.70
phi_jump(t) = 0.8*phi_jump(t-1) + 0.2*max(0, delta_loss_t)
phi_abs(t)  = 0.8*phi_abs(t-1)  + 0.2*abs(delta_loss_t)
Rolling windows are causal and fixed at 5 observations.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

HORIZONS = (1, 2, 5)
BETA = 0.8
WINDOW = 5
SUCCESS_AUROC = 0.70
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
    df = pd.DataFrame(rows)
    if df["step"].duplicated().any():
        raise ValueError("duplicate steps")
    df = df.sort_values("step").reset_index(drop=True)
    return df


def ema_signal(values: np.ndarray, absolute: bool) -> np.ndarray:
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


def derive(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.to_numeric(df["loss_per_token"], errors="coerce")
    d = x.diff()
    df = df.copy()
    df["phi_jump"] = ema_signal(x.to_numpy(float), absolute=False)
    df["phi_abs"] = ema_signal(x.to_numpy(float), absolute=True)
    df["abs_delta_loss"] = d.abs()
    df["rolling_std5"] = x.rolling(WINDOW, min_periods=WINDOW).std(ddof=0)
    med = x.rolling(WINDOW, min_periods=WINDOW).median()
    df["rolling_mad5"] = (x - med).abs().rolling(WINDOW, min_periods=WINDOW).median()
    exp_mean = x.expanding(min_periods=2).mean()
    exp_std = x.expanding(min_periods=2).std(ddof=0).replace(0, np.nan)
    df["abs_causal_z"] = ((x - exp_mean) / exp_std).abs()
    for h in HORIZONS:
        df[f"future_ppo_kl_t{h}"] = pd.to_numeric(df["ppo_kl"], errors="coerce").shift(-h)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_jsonl", type=Path)
    ap.add_argument("--output-csv", type=Path)
    args = ap.parse_args()

    df = derive(load_raw(args.raw_jsonl))
    print(json.dumps({
        "rows": int(len(df)),
        "step_min": int(df.step.min()),
        "step_max": int(df.step.max()),
        "beta": BETA,
        "window": WINDOW,
        "horizons": HORIZONS,
        "success_auroc": SUCCESS_AUROC,
        "note": "Target labels/AUROC require the preregistered calibration rule and whole-seed split; this derivation script does not retune them."
    }, indent=2))

    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output_csv, index=False)
        print(f"derived CSV: {args.output_csv}")


if __name__ == "__main__":
    main()

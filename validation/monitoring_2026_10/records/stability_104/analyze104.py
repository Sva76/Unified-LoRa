"""Offline analysis of the single prespecified seed-104 stability check."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics as S

ROOT = Path(__file__).resolve().parent


def ranks(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise ValueError("Finite, nonempty values required")
    order = sorted(range(len(values)), key=values.__getitem__)
    out = [0.0] * len(values)
    j = 0
    while j < len(values):
        k = j + 1
        while k < len(values) and values[order[k]] == values[order[j]]:
            k += 1
        for i in order[j:k]:
            out[i] = (j + 1 + k) / 2
        j = k
    return out


def pearson(a, b):
    if len(a) != len(b) or len(a) < 2:
        raise ValueError("Paired vectors required")
    a = [v - S.mean(a) for v in a]
    b = [v - S.mean(b) for v in b]
    den = math.sqrt(sum(v*v for v in a) * sum(v*v for v in b))
    return sum(v*w for v,w in zip(a,b)) / den if den > 0 else None


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def window_metrics(states, low=300, high=399):
    window = [r for r in states if low <= r["state"] <= high]
    if [r["state"] for r in window] != list(range(low, high+1)):
        raise ValueError("The frozen 100-state window is incomplete or duplicated")
    for row in window:
        vals = row.get("eval_per_example")
        if not row.get("valid") or not isinstance(vals, list) or len(vals) != 32:
            raise ValueError("Invalid per-example loss row")
        if not all(isinstance(v, (int,float)) and math.isfinite(v) and v >= 0 for v in vals):
            raise ValueError("Invalid per-example loss")
    variability, levels = [], []
    for i in range(32):
        v = [r["eval_per_example"][i] for r in window]
        variability.append(S.median(abs(b-a) for a,b in zip(v,v[1:])))
        levels.append(S.median(v))
    return variability, levels


def compare(frozen, variability, levels):
    rho = spearman(frozen["consensus_stability_ranks"], variability)
    selected = frozen["selected_indices_0based"]
    holdout_order = sorted(range(32), key=lambda i:(variability[i], i))
    selected_median = S.median(variability[i] for i in selected)
    all_median = S.median(variability)
    if rho is None:
        verdict = "NON_VALUTABILE"
    elif rho >= frozen["decision"]["minimum_spearman"]:
        verdict = "CRITERIO_DI_REPLICA_RAGGIUNTO"
    else:
        verdict = "CRITERIO_DI_REPLICA_NON_RAGGIUNTO"
    return {
        "primary_spearman": rho,
        "minimum_spearman": frozen["decision"]["minimum_spearman"],
        "verdict": verdict,
        "selected_indices_0based": selected,
        "selected_example_numbers_1based": [i+1 for i in selected],
        "selected_in_holdout_lowest8": len(set(selected) & set(holdout_order[:8])),
        "selected_median_abs_delta": selected_median,
        "all_examples_median_abs_delta": all_median,
        "selected_to_all_median_ratio": selected_median/all_median if all_median else None,
        "diagnostic_loss_level_baseline_spearman": spearman(frozen["consensus_loss_level_ranks"],variability),
        "per_example": [{"index_0based":i,"example_number_1based":i+1,
                         "median_abs_delta":variability[i],"level_median":levels[i]} for i in range(32)],
        "interpretation": "One healthy trajectory, descriptive replication only. Not a detector, recovery predictor, false-alarm estimate, controller benefit or fluid test. The loss-level baseline is diagnostic only and cannot change the verdict."
    }


def analyze(run_dir):
    import collector104 as R
    import healthy_core as C
    out = Path(run_dir)
    frozen = json.loads((ROOT/"rule104.json").read_text())["rule"]
    manifest = json.loads((out/"manifest.json").read_text())
    config, _ = R.verify_manifest(manifest)
    if config.seed != 104:
        raise ValueError("This analysis accepts only seed 104")
    completion, states, updates, generations, passes = R.verify_run(out,manifest)
    if completion["seed"] != 104 or any(r["seed"] != 104 for r in states+updates):
        raise ValueError("Wrong seed in result records")
    if any(r["protocol_sha256"] != manifest["protocol_sha256"] for r in states):
        raise ValueError("State protocol mismatch")
    result = {"schema":"blind-stability-result-v1", "seed":104,
              "rule_sha256":manifest["protocol"]["analysis_rule_sha256"],
              "protocol_sha256":manifest["protocol_sha256"],
              "run_status":completion["status"],
              "completed_updates":completion["completed_updates"],
              "test_executed":True, "detector_validated":False,
              "recovery_prediction_evaluated":False, "controller_efficacy_evaluated":False}
    if completion["status"] != "COMPLETE":
        result.update(verdict="NON_VALUTABILE",reason="The run did not complete all 400 updates and the original collection protocol")
        return result
    tokens = json.loads((out/"tokenization.json").read_text())
    if tokens["tokenizer_class"] != frozen["tokenizer_class"]:
        result.update(verdict="NON_VALUTABILE",reason="Tokenizer class differs from calibration")
        return result
    if C.digest(tokens["generation_prompt_ids"]) != frozen["generation_prompt_ids_sha256"]:
        result.update(verdict="NON_VALUTABILE",reason="Control prompt tokenization differs from calibration")
        return result
    variability, levels = window_metrics(states, *frozen["state_window"])
    result.update(compare(frozen,variability,levels))
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    a = p.parse_args()
    result = analyze(a.run_dir)
    text = json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+"\n"
    if a.out.exists() and a.out.read_text() != text:
        raise FileExistsError("Refusing to replace a different analysis")
    if not a.out.exists():
        a.out.write_text(text)
    print(result["verdict"])
    if "primary_spearman" in result:
        print("Spearman:",result["primary_spearman"],"; criterio congelato:",result["minimum_spearman"])

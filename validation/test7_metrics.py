"""Offline timing accounting for Test 7; no training or API access.

Retain the historical thresholds, but distinguish retrospective onset from
causal confirmation. This does not repair the historical task or its void
predictive experiment. See corrections_2026_09.md.
"""
import math
from statistics import median

try:
    from .phi_utils import first_sustained_event
except ImportError:
    from phi_utils import first_sustained_event

BASELINE_WIN = (20, 60)
DETECT_FROM = 60
PHI_THRESH, PHI_SUSTAIN = 0.10, 3
COLLAPSE_MULT, COLLAPSE_SUSTAIN = 3.0, 5
NAIVE_MULT, NAIVE_SUSTAIN = 1.5, 3
LEAD_USEFUL = 10


def _finite(value):
    return value is not None and math.isfinite(value)


def analyze_run(log):
    steps = [row[0] for row in log]
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("log steps must be strictly increasing")
    base = [loss for s, loss, _, _ in log
            if BASELINE_WIN[0] <= s < BASELINE_WIN[1] and _finite(loss)]
    med = median(base) if base else None
    phi = first_sustained_event(steps, [
        s >= DETECT_FROM and _finite(p) and p >= PHI_THRESH
        for s, _, p, _ in log], PHI_SUSTAIN)
    collapse, naive = None, None
    first_invalid = next((s for s, loss, _, _ in log
                          if s >= DETECT_FROM and not _finite(loss)), None)
    if med is not None:
        collapse = first_sustained_event(steps, [
            s >= DETECT_FROM and (not _finite(loss) or loss >= COLLAPSE_MULT * med)
            for s, loss, _, _ in log], COLLAPSE_SUSTAIN)
        naive = first_sustained_event(steps, [
            s >= DETECT_FROM and _finite(loss) and loss > NAIVE_MULT * med
            for s, loss, _, _ in log], NAIVE_SUSTAIN)
    onsets = [x for x in (collapse.onset if collapse else None, first_invalid) if x is not None]
    confirmations = [x for x in (collapse.confirmed if collapse else None, first_invalid)
                     if x is not None]
    onset = min(onsets) if onsets else None
    confirmation = min(confirmations) if confirmations else None

    def lead(alarm):
        return onset - alarm if onset is not None and alarm is not None else None

    return {
        "baseline_median": med,
        "t_c_onset": onset,
        "t_c_confirmed": confirmation,
        "t_a_phi_onset": phi.onset if phi else None,
        "t_a_phi_confirmed": phi.confirmed if phi else None,
        "t_a_naive_onset": naive.onset if naive else None,
        "t_a_naive_confirmed": naive.confirmed if naive else None,
        "legacy_lead_phi": lead(phi.onset if phi else None),
        "legacy_lead_naive": lead(naive.onset if naive else None),
        "lead_phi_to_onset": lead(phi.confirmed if phi else None),
        "lead_naive_to_onset": lead(naive.confirmed if naive else None),
    }


def summarize(data):
    collapse = [{"seed": int(seed), **analyze_run(log)}
                for seed, log in sorted(data["logs"]["collapse"].items(), key=lambda kv: int(kv[0]))]
    healthy = [{"seed": int(seed), **analyze_run(log)}
               for seed, log in sorted(data["logs"]["healthy"].items(), key=lambda kv: int(kv[0]))]
    return {
        "timing_version": "causal-confirmation-v2",
        "detect_from": DETECT_FROM,
        "phi_threshold": PHI_THRESH,
        "phi_sustain": PHI_SUSTAIN,
        "confirmatory": False,
        "interpretation": "Descriptive accounting only. Historical Test 7 is void; corrected toy reruns are not protocol v3.",
        "collapse": collapse,
        "healthy": healthy,
        "false_alarms": sum(r["t_a_phi_confirmed"] is not None for r in healthy),
    }

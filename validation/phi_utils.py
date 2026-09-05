"""Loss-only helpers shared by corrected Tinker experiments and offline checks.

These helpers do not import Tinker or start training. See corrections_2026_09.md
before comparing a corrected run with the committed historical logs.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math


def completion_example(prompt_tokens, completion_tokens):
    """Return next-token-aligned inputs, targets and completion-only weights.

    The last prompt position predicts the FIRST completion token. There is no
    synthetic target after the final completion token. A BOS token must be
    supplied by the caller if the prompt would otherwise be empty.
    """
    prompt_tokens, completion_tokens = list(prompt_tokens), list(completion_tokens)
    if not prompt_tokens or not completion_tokens:
        raise ValueError("prompt and completion must each contain at least one token")
    tokens = prompt_tokens + completion_tokens
    return (tokens[:-1], tokens[1:],
            [0.0] * (len(prompt_tokens) - 1) + [1.0] * len(completion_tokens))


def new_run_output(filename):
    """Reserve a new run directory so corrected experiments cannot overwrite evidence."""
    run_dir = Path(__file__).resolve().parent / "corrected_runs" / datetime.now(
        timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir / filename


def write_run_json(path, payload):
    """Write new telemetry once, retaining non-finite observations as JSON null."""
    def clean(value):
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [clean(v) for v in value]
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value
    with Path(path).open("x", encoding="utf-8") as f:
        json.dump(clean(payload), f, indent=2, allow_nan=False)


class PhiJumpMonitor:
    """EMA of positive loss jumps; no gradient input or controller action.

    Non-finite observations leave the EMA unchanged, matching the historical
    monitor. Callers must report them separately as invalid/divergent data;
    the retained EMA must not be interpreted as evidence of healthy training.
    """
    def __init__(self, beta=0.8):
        if not 0 <= beta < 1:
            raise ValueError("beta must be in [0, 1)")
        self.beta = beta
        self.ema_jump = 0.0
        self.prev_loss = None

    def update(self, loss):
        loss = float(loss)
        if not math.isfinite(loss):
            return self.ema_jump
        jump = 0.0 if self.prev_loss is None else max(0.0, loss - self.prev_loss)
        self.prev_loss = loss
        self.ema_jump = self.beta * self.ema_jump + (1 - self.beta) * jump
        return self.ema_jump


class PhiJumpAlarm:
    """Historical Test 7 alarm rule, with causal confirmation timing.

    Steps are zero-based. The EMA sees the whole stream, but consecutive
    exceedances only count from step 60. The alarm becomes available on the
    THIRD exceedance, not retrospectively on the first. The threshold is a
    historical setting, not a validated default for new training regimes.
    """
    def __init__(self, threshold=0.10, sustain=3, detect_from=60, beta=0.8):
        if not math.isfinite(threshold) or threshold < 0:
            raise ValueError("threshold must be finite and non-negative")
        if not isinstance(sustain, int) or sustain < 1:
            raise ValueError("sustain must be a positive integer")
        if not isinstance(detect_from, int) or detect_from < 0:
            raise ValueError("detect_from must be a non-negative integer")
        self.monitor = PhiJumpMonitor(beta)
        self.threshold, self.sustain, self.detect_from = threshold, sustain, detect_from
        self.step = -1
        self.count = 0

    def update(self, loss):
        if not math.isfinite(float(loss)):
            raise ValueError("non-finite loss: handle divergence separately from phi alarms")
        self.step += 1
        phi = self.monitor.update(loss)
        self.count = self.count + 1 if self.step >= self.detect_from and phi >= self.threshold else 0
        return self.count == self.sustain


@dataclass(frozen=True)
class SustainedEvent:
    onset: int
    confirmed: int


def first_sustained_event(steps, flags, sustain):
    """Return actual onset/confirmation steps for the first consecutive event.

    A gap in step numbers resets the streak; row position is not a timestamp.
    """
    if sustain < 1 or len(steps) != len(flags):
        raise ValueError("positive sustain and equal step/flag lengths required")
    count, onset, previous = 0, None, None
    for step, flag in zip(steps, flags):
        if previous is not None and step <= previous:
            raise ValueError("steps must be strictly increasing")
        if previous is not None and step != previous + 1:
            count = 0
        if flag:
            if count == 0:
                onset = step
            count += 1
            if count == sustain:
                return SustainedEvent(onset, step)
        else:
            count = 0
        previous = step
    return None

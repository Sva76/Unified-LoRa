"""
OrbitalController — Dual ΔW Control (Rank + Learning Rate)
============================================================

Closed-loop controller that stabilizes training by regulating
weight updates (ΔW) through TWO independent control channels:

    ΔW = LR × ∇L    (magnitude)
    ΔW = A_r @ B_r   (subspace)

Channel 1 — RANK (via NestedLoRA):
    Controls the dimensionality of the update subspace.
    Low rank → constrained ΔW direction → safer updates.
    High rank → expanded ΔW direction → more expressive but riskier.

Channel 2 — LEARNING RATE:
    Controls the magnitude of the update.
    Low LR → small ΔW steps → stable but slow.
    High LR → large ΔW steps → fast but unstable.

Together:
    Rank controls WHERE updates happen.
    LR controls HOW MUCH update happens.
    Full ΔW control = direction + magnitude.

------------------------------------------------------------------

INSTABILITY SIGNAL φ

    φ = |loss - EMA(loss)| + spike_weight × max(0, loss - prev_loss)

    deviation: sustained divergence from trend
    spike: sudden jumps (noisy batches, data corruption)

φ drives both control channels simultaneously:

    φ rising  → reduce rank + reduce LR  (double brake)
    φ falling → increase rank + increase LR (double accelerator)
    φ stable  → hold current state

------------------------------------------------------------------

ORBITAL STATES

    HIGH   — φ low,  stable conditions   → max rank, aggressive LR
    BASE   — φ mid,  normal training     → mid rank, base LR
    LOW    — φ high, instability detected → min rank, protective LR
                                            + weight snapshot for rollback

Transitions: HIGH ↔ BASE ↔ LOW (no skip, hysteresis enforced)

On entering LOW:  save weight snapshot (Mirror mechanism)
On leaving LOW:   if weights drifted < 5%, restore snapshot
                  (stress was transient noise)

------------------------------------------------------------------

BEHAVIOR UNDER STRESS

Without control:
    ΔW grows uncontrollably → weight drift → collapse

With Orbital control:
    Rank contracts (fewer update directions)
    + LR drops (smaller update magnitude)
    → ΔW is doubly bounded
    → training survives

------------------------------------------------------------------

Key Insight:

    Training stability = f(controlled ΔW)
    Full ΔW control = rank (subspace) + LR (magnitude)

------------------------------------------------------------------

Author: Simona Vargiu
License: Apache 2.0
"""

import math
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from nested_lora import NestedLoRALinear, inject_nested_lora


# ── Types ───────────────────────────────────────────────────────


class OrbitalState(IntEnum):
    HIGH = 0    # Stable → max rank, aggressive LR
    BASE = 1    # Normal → mid rank, base LR
    LOW = 2     # Stress → min rank, protective LR + snapshot


@dataclass
class PhiSnapshot:
    """Instability signal at a single step."""
    step: int
    loss: float
    loss_ema: float
    deviation: float
    spike: float
    phi: float


@dataclass
class Transition:
    """Record of a state change."""
    step: int
    old_state: OrbitalState
    new_state: OrbitalState
    old_rank: int
    new_rank: int
    old_lr: float
    new_lr: float
    phi: float
    trigger: str
    snapshot_saved: bool = False
    snapshot_restored: bool = False


# ── Controller ──────────────────────────────────────────────────


class OrbitalController:
    """
    Dual ΔW controller: adjusts rank AND learning rate based on φ(t).

    Simple API:
        rank = ctrl.step(loss)    # returns recommended rank
        set_rank(model, rank)     # apply to model

    If an optimizer is linked, LR is adjusted automatically.

    Args:
        rank_levels: [r_low, r_base, r_high] for [LOW, BASE, HIGH] states.
            Default: [4, 8, 16].
        base_lr: Base learning rate (BASE state).
        lr_scale_high: LR multiplier for HIGH state (default: 1.5).
        lr_scale_low: LR multiplier for LOW state (default: 0.3).
        ema_alpha: Smoothing factor for loss EMA.
        spike_weight: Weight of spike component in φ.
        stress_k: Std devs above mean φ → stress trigger.
        recovery_k: Std devs below mean φ → recovery trigger.
        warmup: Steps before any state changes.
        stable_window: Minimum steps in a state before transition.
        eval_interval: Steps between evaluations.
        snapshot_on_stress: Save weights when entering LOW.
        drift_threshold: Max relative drift for snapshot rollback.
        optimizer: Optional torch optimizer for LR control.
    """

    def __init__(
        self,
        rank_levels=None,
        base_lr=5e-4,
        lr_scale_high=1.5,
        lr_scale_low=0.3,
        ema_alpha=0.1,
        spike_weight=0.5,
        stress_k=1.5,
        recovery_k=1.0,
        warmup=50,
        stable_window=30,
        eval_interval=5,
        snapshot_on_stress=True,
        drift_threshold=0.05,
        optimizer=None,
    ):
        self.rank_levels = rank_levels or [4, 8, 16]
        assert len(self.rank_levels) == 3

        self.base_lr = base_lr
        self.lr_high = base_lr * lr_scale_high
        self.lr_low = base_lr * lr_scale_low
        self.ema_alpha = ema_alpha
        self.spike_weight = spike_weight
        self.stress_k = stress_k
        self.recovery_k = recovery_k
        self.warmup = warmup
        self.stable_window = stable_window
        self.eval_interval = eval_interval
        self.snapshot_on_stress = snapshot_on_stress
        self.drift_threshold = drift_threshold
        self.optimizer = optimizer

        # State → (rank, lr) mapping
        self._state_config = {
            OrbitalState.HIGH: (self.rank_levels[2], self.lr_high),
            OrbitalState.BASE: (self.rank_levels[1], self.base_lr),
            OrbitalState.LOW:  (self.rank_levels[0], self.lr_low),
        }

        # Internal state
        self.current_state = OrbitalState.BASE
        self.current_rank = self.rank_levels[1]
        self.current_lr = base_lr

        self.loss_ema = 0.0
        self.prev_loss = 0.0
        self.phi_history: deque = deque(maxlen=500)
        self.phi_ema = 0.0

        self.steps_in_state = 0
        self.global_step = 0

        self.weight_snapshot: Optional[dict] = None
        self._adapters: Optional[Dict[str, NestedLoRALinear]] = None

        self.transition_log: List[Transition] = []
        self.phi_log: List[PhiSnapshot] = []

    # ── Main API ────────────────────────────────────────────────

    def step(self, loss: float) -> int:
        """
        Process one training step. Returns recommended rank.

        Call AFTER loss.backward(), BEFORE optimizer.step().
        If optimizer is linked, LR is adjusted automatically.

        Args:
            loss: Current batch loss (scalar).

        Returns:
            Active rank to apply via set_rank(model, rank).
        """
        self.global_step += 1
        self.steps_in_state += 1

        # ── Compute φ ───────────────────────────────────────────
        phi_snap = self._compute_phi(loss)
        self.phi_log.append(phi_snap)
        self.phi_history.append(phi_snap.phi)
        self.phi_ema = (1 - self.ema_alpha) * self.phi_ema + self.ema_alpha * phi_snap.phi

        self.prev_loss = loss

        # ── Evaluate state ──────────────────────────────────────
        if (
            self.global_step >= self.warmup
            and self.global_step % self.eval_interval == 0
        ):
            self._evaluate(phi_snap)

        return self.current_rank

    def link_optimizer(self, optimizer: torch.optim.Optimizer):
        """Link optimizer for automatic LR control."""
        self.optimizer = optimizer
        self._apply_lr(self.current_lr)

    def link_adapters(self, adapters: Dict[str, NestedLoRALinear]):
        """Link adapters for weight snapshot mechanism."""
        self._adapters = adapters

    def get_summary(self) -> dict:
        return {
            "state": self.current_state.name,
            "rank": self.current_rank,
            "lr": round(self.current_lr, 8),
            "phi": round(self.phi_ema, 6),
            "steps_in_state": self.steps_in_state,
            "transitions": len(self.transition_log),
            "has_snapshot": self.weight_snapshot is not None,
            "step": self.global_step,
        }

    def orbit_history(self) -> List[dict]:
        return [
            {
                "step": t.step,
                "from": t.old_state.name,
                "to": t.new_state.name,
                "rank": t.new_rank,
                "lr": round(t.new_lr, 8),
                "phi": round(t.phi, 6),
                "trigger": t.trigger,
                "snapshot": "saved" if t.snapshot_saved else
                           ("restored" if t.snapshot_restored else "—"),
            }
            for t in self.transition_log
        ]

    # ── φ computation ───────────────────────────────────────────

    def _compute_phi(self, loss: float) -> PhiSnapshot:
        """
        φ = |loss - EMA(loss)| + spike_weight × max(0, Δloss)

        Deviation: sustained divergence from trend.
        Spike: sudden jumps (noisy batch, corruption).
        """
        if self.global_step == 1:
            self.loss_ema = loss
            self.prev_loss = loss

        self.loss_ema = (1 - self.ema_alpha) * self.loss_ema + self.ema_alpha * loss

        deviation = abs(loss - self.loss_ema)
        spike = max(0.0, loss - self.prev_loss)
        phi = deviation + self.spike_weight * spike

        return PhiSnapshot(
            step=self.global_step,
            loss=loss,
            loss_ema=self.loss_ema,
            deviation=deviation,
            spike=spike,
            phi=phi,
        )

    # ── State machine ───────────────────────────────────────────

    def _evaluate(self, phi_snap: PhiSnapshot):
        """FSM evaluation with adaptive thresholds and hysteresis."""
        if self.steps_in_state < self.stable_window:
            return

        if len(self.phi_history) < 10:
            return

        # Adaptive thresholds
        values = list(self.phi_history)
        mu = sum(values) / len(values)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))

        stress_thresh = mu + self.stress_k * sigma
        recovery_thresh = mu - self.recovery_k * sigma

        phi = phi_snap.phi
        old = self.current_state
        new = old

        # ── Toward LOW (stress) ─────────────────────────────────
        if old == OrbitalState.BASE and phi > stress_thresh:
            new = OrbitalState.LOW

        elif old == OrbitalState.HIGH and phi > recovery_thresh:
            new = OrbitalState.BASE

        # ── Toward HIGH (recovery) ──────────────────────────────
        elif old == OrbitalState.LOW and phi < stress_thresh:
            new = OrbitalState.BASE

        elif old == OrbitalState.BASE and phi < recovery_thresh:
            new = OrbitalState.HIGH

        if new != old:
            self._apply_transition(old, new, phi)

    def _apply_transition(self, old: OrbitalState, new: OrbitalState, phi: float):
        """Execute state change: update rank, LR, and manage snapshots."""
        old_rank = self.current_rank
        old_lr = self.current_lr
        new_rank, new_lr = self._state_config[new]

        snapshot_saved = False
        snapshot_restored = False

        # ── Entering LOW: save weight snapshot ──────────────────
        if new == OrbitalState.LOW and self.snapshot_on_stress:
            if self._adapters is not None:
                self.weight_snapshot = self._save_snapshot()
                snapshot_saved = True

        # ── Leaving LOW: evaluate rollback ──────────────────────
        if old == OrbitalState.LOW and self.weight_snapshot is not None:
            if self._adapters is not None and self._should_restore():
                self._restore_snapshot()
                snapshot_restored = True
            self.weight_snapshot = None

        # ── Apply rank + LR ─────────────────────────────────────
        self.current_state = new
        self.current_rank = new_rank
        self.current_lr = new_lr
        self.steps_in_state = 0

        if self.optimizer is not None:
            self._apply_lr(new_lr)

        self.transition_log.append(Transition(
            step=self.global_step,
            old_state=old,
            new_state=new,
            old_rank=old_rank,
            new_rank=new_rank,
            old_lr=old_lr,
            new_lr=new_lr,
            phi=phi,
            trigger="stress" if new.value > old.value else "recovery",
            snapshot_saved=snapshot_saved,
            snapshot_restored=snapshot_restored,
        ))

    def _apply_lr(self, lr: float):
        """Set LR on all optimizer param groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ── Weight snapshot (Mirror) ────────────────────────────────

    def _save_snapshot(self) -> dict:
        snap = {}
        for name, adapter in self._adapters.items():
            snap[name] = {
                "lora_A": adapter.lora_A.data.clone(),
                "lora_B": adapter.lora_B.data.clone(),
            }
        snap["_step"] = self.global_step
        return snap

    def _should_restore(self) -> bool:
        """Restore if weights drifted < threshold (transient noise)."""
        if self.weight_snapshot is None or self._adapters is None:
            return False

        total_drift = 0.0
        total_baseline = 0.0

        for name, adapter in self._adapters.items():
            if name not in self.weight_snapshot:
                continue
            s = self.weight_snapshot[name]
            total_drift += (adapter.lora_A.data - s["lora_A"]).norm().item()
            total_drift += (adapter.lora_B.data - s["lora_B"]).norm().item()
            total_baseline += s["lora_A"].norm().item()
            total_baseline += s["lora_B"].norm().item()

        if total_baseline < 1e-8:
            return False

        return (total_drift / total_baseline) < self.drift_threshold

    def _restore_snapshot(self):
        for name, adapter in self._adapters.items():
            if name not in self.weight_snapshot:
                continue
            s = self.weight_snapshot[name]
            adapter.lora_A.data.copy_(s["lora_A"])
            adapter.lora_B.data.copy_(s["lora_B"])


# ── Convenience setup ───────────────────────────────────────────


def setup_unified_lora(
    model: nn.Module,
    target_modules: list = None,
    max_rank: int = 16,
    alpha: float = 16.0,
    base_lr: float = 5e-4,
    **controller_kwargs,
) -> Tuple[Dict[str, NestedLoRALinear], OrbitalController, torch.optim.Optimizer]:
    """
    One-call setup: NestedLoRA + OrbitalController + Optimizer.

    Creates a full ΔW control system:
        rank → ΔW subspace (direction)
        LR → ΔW magnitude (amplitude)

    Returns:
        (adapters, controller, optimizer)

    Usage:
        adapters, ctrl, opt = setup_unified_lora(model)
        for batch in dataloader:
            loss = model(**batch).loss
            loss.backward()
            rank = ctrl.step(loss.item())
            set_rank(model, rank)
            opt.step()
            opt.zero_grad()
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Freeze base
    for p in model.parameters():
        p.requires_grad = False

    # Inject NestedLoRA
    adapters = inject_nested_lora(model, target_modules, max_rank=max_rank, alpha=alpha)

    # Optimizer over LoRA params only
    from nested_lora import get_lora_params
    lora_params = list(get_lora_params(model))
    optimizer = torch.optim.AdamW(lora_params, lr=base_lr)

    # Controller with dual control
    ctrl = OrbitalController(base_lr=base_lr, **controller_kwargs)
    ctrl.link_optimizer(optimizer)
    ctrl.link_adapters(adapters)

    return adapters, ctrl, optimizer

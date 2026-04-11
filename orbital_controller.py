"""
OrbitalLRController — Weight Dynamics Stabilizer
================================================

Adaptive learning rate controller designed to regulate weight updates (ΔW)
during training by reacting to instability signals in real time.

Unlike static schedulers (cosine, linear), this controller operates as a
closed-loop system:

    loss → instability signal (φ) → LR adjustment → ΔW modulation

------------------------------------------------------------------

⚠️ CORE IDEA: CONTROL OF ΔW (WEIGHT UPDATES)

Learning rate directly scales weight updates:

    ΔW ∝ LR × ∇L

This means:

- High LR → large ΔW → fast learning but unstable
- Low LR  → small ΔW → stable but slow

OrbitalLRController dynamically adjusts LR to keep ΔW within a stable regime.

------------------------------------------------------------------

CONTROL MECHANISM

We define an instability signal φ based on:

- deviation from EMA(loss)
- loss spikes (Δloss)

    φ = |loss - EMA(loss)| + spike_component

The controller maintains two adaptive thresholds:

- stress threshold → instability detected → reduce LR
- stable threshold → recovery detected → increase LR

This creates an orbital behavior:

    HIGH → BASE → LOW → BASE → HIGH

Each transition corresponds to regulating ΔW magnitude.

------------------------------------------------------------------

BEHAVIOR UNDER STRESS

Without control:
    ΔW grows uncontrollably → weight drift → collapse (NaN / divergence)

With Orbital control:
    ΔW is reduced during instability
    → drift is bounded
    → training survives

------------------------------------------------------------------

KEY PROPERTIES

- Closed-loop (feedback-driven)
- No predefined schedule
- Reacts to real training dynamics
- Prevents catastrophic updates
- Enables aggressive training regimes

------------------------------------------------------------------

SYSTEM VIEW

OrbitalLRController is one component of a broader weight control system:

    controller → adjusts LR → modulates ΔW → stabilizes weights

Combined with NestedLoRA:

    rank → defines ΔW subspace
    LR   → defines ΔW magnitude

Together:

    control over direction + control over magnitude = full ΔW control

------------------------------------------------------------------

Key Insight:

    Training stability is a function of controlled ΔW,
    not just optimized loss.

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


class LRState(IntEnum):
    """Orbital LR states — each represents a qualitative regime."""
    HIGH = 0     # Aggressive learning — stable conditions
    BASE = 1     # Normal learning — default operating point
    LOW = 2      # Protective mode — instability detected


@dataclass
class PhiSnapshot:
    """Instability signal at a single step."""
    step: int
    loss: float
    loss_ema: float
    deviation: float       # |loss - EMA(loss)|
    spike: float           # max(0, loss - prev_loss)
    phi: float             # Combined instability signal
    phi_normalized: float  # φ mapped to [0, 1]


@dataclass
class LRTransition:
    """Record of a learning rate change."""
    step: int
    old_state: LRState
    new_state: LRState
    old_lr: float
    new_lr: float
    phi: float
    trigger: str           # "stress" | "recovery" | "warmup"


@dataclass
class ControllerState:
    """Internal state of the controller."""
    current_lr_state: LRState = LRState.BASE
    current_lr: float = 0.0

    # Loss tracking
    loss_ema: float = 0.0
    prev_loss: float = 0.0
    loss_history: deque = field(default_factory=lambda: deque(maxlen=200))

    # φ tracking
    phi_history: deque = field(default_factory=lambda: deque(maxlen=500))
    phi_ema: float = 0.0

    # Transition log
    orbit_stack: list = field(default_factory=list)

    # Hysteresis
    steps_in_state: int = 0

    # Mirror snapshot (weights saved on entering LOW state)
    weight_snapshot: Optional[dict] = None


# ── Controller ──────────────────────────────────────────────────


class OrbitalController:
    """
    Closed-loop learning rate controller that stabilizes ΔW.

    Monitors an instability signal φ derived from loss dynamics and
    adjusts learning rate across three orbital states (HIGH / BASE / LOW).

    When combined with NestedLoRA:
        - Rank controls the SUBSPACE of ΔW (direction)
        - LR controls the MAGNITUDE of ΔW (amplitude)
        - Together: full control over weight updates

    Args:
        optimizer: The torch optimizer whose LR will be controlled.
        base_lr: Default learning rate (BASE state).
        high_lr: Aggressive learning rate (HIGH state). Default: 1.5× base.
        low_lr: Protective learning rate (LOW state). Default: 0.3× base.
        ema_alpha: Smoothing factor for loss EMA (0 < α ≤ 1).
        spike_weight: Weight of the spike component in φ.
        stress_k: Std deviations above mean φ to trigger stress (→ LOW).
        recovery_k: Std deviations below mean φ to trigger recovery (→ HIGH).
        hysteresis_steps: Minimum steps in a state before transition.
        eval_interval: Steps between state evaluations.
        warmup_steps: Steps before any state changes.
        snapshot_on_stress: Save weight snapshot when entering LOW state.
        snapshot_drift_threshold: Max relative drift to trigger rollback (0.05 = 5%).
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_lr: float = 5e-4,
        high_lr: float = None,
        low_lr: float = None,
        ema_alpha: float = 0.1,
        spike_weight: float = 0.5,
        stress_k: float = 1.5,
        recovery_k: float = 1.0,
        hysteresis_steps: int = 30,
        eval_interval: int = 10,
        warmup_steps: int = 50,
        snapshot_on_stress: bool = True,
        snapshot_drift_threshold: float = 0.05,
    ):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.high_lr = high_lr or base_lr * 1.5
        self.low_lr = low_lr or base_lr * 0.3
        self.ema_alpha = ema_alpha
        self.spike_weight = spike_weight
        self.stress_k = stress_k
        self.recovery_k = recovery_k
        self.hysteresis_steps = hysteresis_steps
        self.eval_interval = eval_interval
        self.warmup_steps = warmup_steps
        self.snapshot_on_stress = snapshot_on_stress
        self.snapshot_drift_threshold = snapshot_drift_threshold

        # State → LR mapping
        self.state_lr = {
            LRState.HIGH: self.high_lr,
            LRState.BASE: self.base_lr,
            LRState.LOW:  self.low_lr,
        }

        # Internal state
        self.state = ControllerState(
            current_lr=base_lr,
            current_lr_state=LRState.BASE,
        )

        self.transition_log: List[LRTransition] = []
        self.phi_log: List[PhiSnapshot] = []
        self.global_step = 0

        # Reference to adapters (set via setup_unified_lora)
        self._adapters: Optional[Dict[str, NestedLoRALinear]] = None

        # Apply initial LR
        self._set_lr(base_lr)

    # ── Public API ──────────────────────────────────────────────

    def step(self, loss: float) -> Optional[LRTransition]:
        """
        Call once per training step (after backward, before optimizer.step).

        This is the main control loop:
            loss → φ → state evaluation → LR adjustment → ΔW regulation

        Args:
            loss: Current batch loss value (required).

        Returns:
            LRTransition if a state change occurred, None otherwise.
        """
        self.global_step += 1
        s = self.state
        s.steps_in_state += 1

        # ── Compute φ ───────────────────────────────────────────
        phi_snap = self._compute_phi(loss)
        self.phi_log.append(phi_snap)

        s.phi_history.append(phi_snap.phi)
        s.phi_ema = (1 - self.ema_alpha) * s.phi_ema + self.ema_alpha * phi_snap.phi_normalized

        # Update loss tracking
        s.prev_loss = loss

        # ── Evaluate state transition ───────────────────────────
        if (
            self.global_step < self.warmup_steps
            or self.global_step % self.eval_interval != 0
        ):
            return None

        transition = self._evaluate_transition(phi_snap)
        if transition is not None:
            self.transition_log.append(transition)
            s.orbit_stack.append(transition)

        return transition

    def get_lr(self) -> float:
        """Current learning rate."""
        return self.state.current_lr

    def get_lr_state(self) -> LRState:
        """Current orbital state."""
        return self.state.current_lr_state

    def get_phi(self) -> float:
        """Current smoothed φ value."""
        return self.state.phi_ema

    def get_summary(self) -> dict:
        """Current controller state."""
        return {
            "lr_state": self.state.current_lr_state.name,
            "lr": round(self.state.current_lr, 8),
            "phi": round(self.state.phi_ema, 6),
            "steps_in_state": self.state.steps_in_state,
            "transitions": len(self.state.orbit_stack),
            "has_snapshot": self.state.weight_snapshot is not None,
            "step": self.global_step,
        }

    def orbit_history(self) -> List[dict]:
        """Condensed transition history for visualization."""
        return [
            {
                "step": t.step,
                "from": t.old_state.name,
                "to": t.new_state.name,
                "lr": round(t.new_lr, 8),
                "phi": round(t.phi, 6),
                "trigger": t.trigger,
            }
            for t in self.transition_log
        ]

    # ── φ computation ───────────────────────────────────────────

    def _compute_phi(self, loss: float) -> PhiSnapshot:
        """
        Compute instability signal:

            φ = |loss - EMA(loss)| + spike_weight × max(0, loss - prev_loss)

        Deviation captures sustained divergence from trend.
        Spike captures sudden jumps (noisy batches, data corruption).
        """
        s = self.state

        # Update loss EMA
        if self.global_step == 1:
            s.loss_ema = loss
            s.prev_loss = loss

        s.loss_ema = (1 - self.ema_alpha) * s.loss_ema + self.ema_alpha * loss
        s.loss_history.append(loss)

        # Deviation from trend
        deviation = abs(loss - s.loss_ema)

        # Spike detection
        spike = max(0.0, loss - s.prev_loss)

        # Combined signal
        phi = deviation + self.spike_weight * spike

        # Normalize to [0, 1]
        phi_normalized = self._normalize_phi(phi)

        return PhiSnapshot(
            step=self.global_step,
            loss=loss,
            loss_ema=s.loss_ema,
            deviation=deviation,
            spike=spike,
            phi=phi,
            phi_normalized=phi_normalized,
        )

    def _normalize_phi(self, phi_raw: float) -> float:
        """Normalize φ to [0, 1] via running z-score."""
        history = self.state.phi_history
        if len(history) < 10:
            return 0.5

        values = list(history)
        mu = sum(values) / len(values)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))

        if sigma < 1e-8:
            return 0.5

        z = (phi_raw - mu) / sigma
        z_clamped = max(-3.0, min(3.0, z))
        return (z_clamped + 3.0) / 6.0

    # ── State machine ───────────────────────────────────────────

    def _evaluate_transition(self, phi_snap: PhiSnapshot) -> Optional[LRTransition]:
        """
        Orbital state machine with hysteresis.

        Transitions:
            BASE → LOW:   φ > stress_threshold   (instability detected)
            LOW  → BASE:  φ < stress_threshold   (recovery)
            BASE → HIGH:  φ < recovery_threshold (sustained stability)
            HIGH → BASE:  φ > recovery_threshold (stability lost)

        No skip transitions: HIGH ↔ BASE ↔ LOW
        """
        s = self.state

        if s.steps_in_state < self.hysteresis_steps:
            return None

        if len(s.phi_history) < 20:
            return None

        # Adaptive thresholds from running statistics
        values = list(s.phi_history)
        mu = sum(values) / len(values)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in values) / len(values))

        stress_threshold = mu + self.stress_k * sigma
        recovery_threshold = mu - self.recovery_k * sigma

        phi = phi_snap.phi
        old_state = s.current_lr_state
        new_state = old_state
        trigger = ""

        # ── Upward stress (toward LOW / protective) ─────────────
        if old_state == LRState.BASE and phi > stress_threshold:
            new_state = LRState.LOW
            trigger = "stress"

        elif old_state == LRState.HIGH and phi > recovery_threshold:
            new_state = LRState.BASE
            trigger = "stress"

        # ── Downward recovery (toward HIGH / aggressive) ────────
        elif old_state == LRState.LOW and phi < stress_threshold:
            new_state = LRState.BASE
            trigger = "recovery"

        elif old_state == LRState.BASE and phi < recovery_threshold:
            new_state = LRState.HIGH
            trigger = "recovery"

        if new_state == old_state:
            return None

        return self._apply_transition(old_state, new_state, phi_snap.phi, trigger)

    def _apply_transition(
        self,
        old_state: LRState,
        new_state: LRState,
        phi: float,
        trigger: str,
    ) -> LRTransition:
        """Execute a state transition with side effects."""
        s = self.state
        old_lr = s.current_lr
        new_lr = self.state_lr[new_state]

        # ── Entering LOW (stress): save weight snapshot ─────────
        if new_state == LRState.LOW and self.snapshot_on_stress:
            if self._adapters is not None:
                s.weight_snapshot = self._save_snapshot()

        # ── Leaving LOW (recovery): evaluate rollback ───────────
        if old_state == LRState.LOW and s.weight_snapshot is not None:
            if self._adapters is not None:
                if self._should_restore(s.weight_snapshot):
                    self._restore_snapshot(s.weight_snapshot)
            s.weight_snapshot = None

        # ── Apply new LR ────────────────────────────────────────
        self._set_lr(new_lr)
        s.current_lr = new_lr
        s.current_lr_state = new_state
        s.steps_in_state = 0

        return LRTransition(
            step=self.global_step,
            old_state=old_state,
            new_state=new_state,
            old_lr=old_lr,
            new_lr=new_lr,
            phi=phi,
            trigger=trigger,
        )

    # ── LR application ──────────────────────────────────────────

    def _set_lr(self, lr: float):
        """Apply learning rate to all parameter groups."""
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    # ── Weight snapshot (Mirror mechanism) ──────────────────────

    def _save_snapshot(self) -> dict:
        """Save current LoRA weights before entering protective mode."""
        snapshot = {}
        for name, adapter in self._adapters.items():
            snapshot[name] = {
                "lora_A": adapter.lora_A.data.clone(),
                "lora_B": adapter.lora_B.data.clone(),
            }
        snapshot["_step"] = self.global_step
        return snapshot

    def _should_restore(self, snapshot: dict) -> bool:
        """
        Decide whether to restore pre-stress weights.

        If weights barely moved during LOW state (< threshold relative drift),
        the stress was transient noise → restore stable weights.
        If weights moved significantly, the stress was real → keep new weights.
        """
        total_drift = 0.0
        total_baseline = 0.0

        for name, adapter in self._adapters.items():
            if name not in snapshot:
                continue
            snap = snapshot[name]
            drift_A = (adapter.lora_A.data - snap["lora_A"]).norm().item()
            drift_B = (adapter.lora_B.data - snap["lora_B"]).norm().item()
            base_A = snap["lora_A"].norm().item()
            base_B = snap["lora_B"].norm().item()

            total_drift += drift_A + drift_B
            total_baseline += base_A + base_B

        if total_baseline < 1e-8:
            return False

        relative_drift = total_drift / total_baseline
        return relative_drift < self.snapshot_drift_threshold

    def _restore_snapshot(self, snapshot: dict):
        """Restore pre-stress weights to all adapters."""
        for name, adapter in self._adapters.items():
            if name not in snapshot:
                continue
            snap = snapshot[name]
            adapter.lora_A.data.copy_(snap["lora_A"])
            adapter.lora_B.data.copy_(snap["lora_B"])


# ── Convenience setup ───────────────────────────────────────────


def setup_unified_lora(
    model: nn.Module,
    optimizer: torch.optim.Optimizer = None,
    target_modules: list = None,
    max_rank: int = 16,
    alpha: float = 16.0,
    base_lr: float = 5e-4,
    **controller_kwargs,
) -> Tuple[Dict[str, NestedLoRALinear], OrbitalController]:
    """
    One-call setup: inject NestedLoRA + create OrbitalController.

    This creates a full ΔW control system:
        - NestedLoRA: controls ΔW subspace (via rank)
        - OrbitalController: controls ΔW magnitude (via LR)

    Args:
        model: Base model to adapt.
        optimizer: Torch optimizer. If None, creates AdamW with base_lr.
        target_modules: Layer name patterns (default: ["q_proj", "v_proj"]).
        max_rank: Maximum rank per adapter.
        alpha: LoRA alpha.
        base_lr: Base learning rate for the controller.
        **controller_kwargs: Passed to OrbitalController.

    Returns:
        (adapters_dict, controller)

    Usage:
        adapters, ctrl = setup_unified_lora(model)
        # optimizer is created internally — access via ctrl.optimizer
        for batch in dataloader:
            loss = model(**batch).loss
            loss.backward()
            ctrl.step(loss=loss.item())
            ctrl.optimizer.step()
            ctrl.optimizer.zero_grad()
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    # Inject NestedLoRA
    adapters = inject_nested_lora(
        model, target_modules, max_rank=max_rank, alpha=alpha
    )

    # Create optimizer if not provided
    if optimizer is None:
        from nested_lora import get_lora_params
        lora_params = list(get_lora_params(model))
        optimizer = torch.optim.AdamW(lora_params, lr=base_lr)

    # Create controller
    controller = OrbitalController(
        optimizer=optimizer,
        base_lr=base_lr,
        **controller_kwargs,
    )

    # Link adapters for snapshot mechanism
    controller._adapters = adapters

    return adapters, controller

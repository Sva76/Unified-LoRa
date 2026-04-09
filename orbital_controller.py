"""
OrbitalController — Synaptic Stress FSM + Nested Rank Slicing
===============================================================
Unifies the two core ideas of Unified-LoRA:

1. **NestedLoRA** (execution): rank via matrix slicing, zero cold-start
2. **φ(t) FSM** (control): qualitatively different operational modes
   driven by a synaptic stress signal inspired by neurobiological
   plasticity (potentiation → stress → recovery).

Modes:
    Mode 0 — SINGLE  (φ < φ_low):   Low capacity, efficient cruise.
             Rank slice → r_min. Minimal parameters active.

    Mode 1 — MULTI   (φ_low ≤ φ < φ_high): Intermediate capacity.
             Rank slice → r_mid. Standard fine-tuning regime.

    Mode 2 — MIRROR  (φ ≥ φ_high):  Maximum capacity + stability snapshot.
             Rank slice → r_max. Weight snapshot saved for rollback.
             This is NOT just "more rank" — it adds redundancy and
             the ability to revert if the stress was transient.

Stress signal:
    φ(t) = f(C, E, S)
        C = convergence signal  (loss EMA trend — are we improving?)
        E = entropy signal      (gradient direction diversity — are
                                 gradients aligned or chaotic?)
        S = stress magnitude    (gradient norm EMA — raw force)

    φ is normalized to [0, 1] via adaptive scaling (running μ ± σ).

Key difference from a PID/threshold controller:
    The FSM modes are qualitatively different responses, not just
    quantitative rank adjustments. Mirror mode saves snapshots.
    Recovery from Mirror → Multi → Single follows hysteresis
    (requires sustained low stress, not a single good step).

Author: Simona Vargiu
License: Apache 2.0
"""

import copy
import math
import torch
import torch.nn as nn
from collections import deque
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Tuple

from nested_lora import NestedLoRALinear


# ── Types ───────────────────────────────────────────────────────


class Mode(IntEnum):
    SINGLE = 0   # Low stress — efficient cruise
    MULTI = 1    # Moderate stress — working regime
    MIRROR = 2   # High stress — max capacity + snapshot


@dataclass
class PhiComponents:
    """The three components of the synaptic stress signal."""
    convergence: float = 0.0   # C: loss trend (positive = diverging)
    entropy: float = 0.0       # E: gradient direction diversity
    stress: float = 0.0        # S: gradient magnitude
    phi: float = 0.0           # Combined φ(t)


@dataclass
class ModeTransition:
    """Record of a mode change."""
    step: int
    layer_name: str
    old_mode: Mode
    new_mode: Mode
    old_rank: int
    new_rank: int
    phi: float
    components: PhiComponents
    snapshot_saved: bool = False
    snapshot_restored: bool = False


@dataclass
class AdapterState:
    """Per-adapter tracking."""
    current_mode: Mode = Mode.MULTI
    current_rank: int = 8

    # Stress signal components
    loss_ema: float = 0.0
    loss_ema_slow: float = 0.0       # Slower EMA for trend detection
    grad_norm_ema: float = 0.0
    grad_direction_ema: float = 0.0  # Cosine similarity of consecutive grads
    prev_grad: Optional[torch.Tensor] = None

    # φ history
    phi_history: deque = field(default_factory=lambda: deque(maxlen=200))
    phi_ema: float = 0.0

    # Adaptive normalization for φ
    phi_raw_history: deque = field(default_factory=lambda: deque(maxlen=500))

    # Mode transition log
    orbit_stack: list = field(default_factory=list)

    # Mirror snapshot (saved when entering Mirror mode)
    mirror_snapshot: Optional[dict] = None

    # Hysteresis: steps in current mode (prevents rapid oscillation)
    steps_in_mode: int = 0


# ── Controller ──────────────────────────────────────────────────


class OrbitalController:
    """
    FSM-driven rank controller for NestedLoRA adapters.

    Args:
        adapters: Dict of name → NestedLoRALinear.
        rank_levels: [r_min, r_mid, r_max] mapping to [SINGLE, MULTI, MIRROR].
        phi_low: φ threshold for SINGLE↔MULTI boundary (default: 0.3).
        phi_high: φ threshold for MULTI↔MIRROR boundary (default: 0.7).
        ema_fast: Fast EMA alpha for stress/loss (default: 0.1).
        ema_slow: Slow EMA alpha for trend detection (default: 0.02).
        hysteresis_steps: Minimum steps in a mode before allowing transition.
        eval_interval: Steps between mode evaluations.
        warmup_steps: Steps before any mode changes.
        phi_weights: (w_C, w_E, w_S) weights for combining φ components.
    """

    def __init__(
        self,
        adapters: Dict[str, NestedLoRALinear],
        rank_levels: List[int] = None,
        phi_low: float = 0.3,
        phi_high: float = 0.7,
        ema_fast: float = 0.1,
        ema_slow: float = 0.02,
        hysteresis_steps: int = 30,
        eval_interval: int = 10,
        warmup_steps: int = 50,
        phi_weights: Tuple[float, float, float] = (0.3, 0.3, 0.4),
    ):
        self.adapters = adapters
        self.rank_levels = rank_levels or [4, 8, 16]
        assert len(self.rank_levels) == 3, "Need exactly 3 rank levels: [single, multi, mirror]"

        self.phi_low = phi_low
        self.phi_high = phi_high
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.hysteresis_steps = hysteresis_steps
        self.eval_interval = eval_interval
        self.warmup_steps = warmup_steps
        self.w_C, self.w_E, self.w_S = phi_weights

        # Mode → rank mapping
        self.mode_rank = {
            Mode.SINGLE: self.rank_levels[0],
            Mode.MULTI:  self.rank_levels[1],
            Mode.MIRROR: self.rank_levels[2],
        }

        # Per-adapter state
        self.states: Dict[str, AdapterState] = {}
        for name, adapter in adapters.items():
            adapter.active_rank = self.rank_levels[1]  # Start in MULTI
            self.states[name] = AdapterState(
                current_mode=Mode.MULTI,
                current_rank=self.rank_levels[1],
            )

        self.transition_log: List[ModeTransition] = []
        self.global_step = 0

    # ── Public API ──────────────────────────────────────────────

    def step(self, loss: Optional[float] = None) -> List[ModeTransition]:
        """
        Call once per training step (after backward, before optimizer.step).

        Args:
            loss: Current batch loss value. If provided, enables the
                  convergence component of φ. Optional but recommended.

        Returns:
            List of mode transitions that occurred (empty if no changes).
        """
        self.global_step += 1
        transitions = []

        for name, adapter in self.adapters.items():
            state = self.states[name]
            state.steps_in_mode += 1

            # ── Update φ components ─────────────────────────────
            phi_comp = self._compute_phi(adapter, state, loss)
            state.phi_history.append(phi_comp.phi)
            state.phi_ema = (
                (1 - self.ema_fast) * state.phi_ema + self.ema_fast * phi_comp.phi
            )

            # ── Evaluate mode transition ────────────────────────
            if (
                self.global_step < self.warmup_steps
                or self.global_step % self.eval_interval != 0
            ):
                continue

            transition = self._evaluate_fsm(name, adapter, state, phi_comp)
            if transition is not None:
                transitions.append(transition)
                self.transition_log.append(transition)

        return transitions

    def get_phi(self, layer_name: str) -> float:
        """Current φ value for a specific adapter."""
        return self.states[layer_name].phi_ema

    def get_mode(self, layer_name: str) -> Mode:
        """Current mode for a specific adapter."""
        return self.states[layer_name].current_mode

    def get_summary(self) -> Dict[str, dict]:
        """Current state of all adapters."""
        summary = {}
        for name, state in self.states.items():
            summary[name] = {
                "mode": state.current_mode.name,
                "rank": state.current_rank,
                "phi": round(state.phi_ema, 4),
                "steps_in_mode": state.steps_in_mode,
                "transitions": len(state.orbit_stack),
                "has_snapshot": state.mirror_snapshot is not None,
            }
        return summary

    def avg_rank(self) -> float:
        """Average active rank across all adapters."""
        ranks = [s.current_rank for s in self.states.values()]
        return sum(ranks) / len(ranks) if ranks else 0.0

    def rank_saving_pct(self) -> float:
        """Rank reduction vs max allocation."""
        max_total = self.rank_levels[-1] * len(self.adapters)
        active_total = sum(s.current_rank for s in self.states.values())
        return (1 - active_total / max_total) * 100 if max_total > 0 else 0.0

    def mode_distribution(self) -> Dict[str, int]:
        """Count of adapters in each mode."""
        dist = {m.name: 0 for m in Mode}
        for state in self.states.values():
            dist[state.current_mode.name] += 1
        return dist

    # ── φ(t) computation ────────────────────────────────────────

    def _compute_phi(
        self,
        adapter: NestedLoRALinear,
        state: AdapterState,
        loss: Optional[float],
    ) -> PhiComponents:
        """
        Compute the synaptic stress signal φ(t) = f(C, E, S).

        C (Convergence): Trend of loss — positive means diverging.
            Uses ratio of fast EMA to slow EMA. If fast > slow,
            loss is increasing → stress.

        E (Entropy): Gradient direction diversity.
            Cosine similarity between consecutive gradient snapshots.
            Low similarity = chaotic gradients = high entropy = stress.

        S (Stress magnitude): Raw gradient norm, EMA-smoothed.
        """
        comp = PhiComponents()

        # ── S: Gradient magnitude ───────────────────────────────
        grad_norm = self._grad_norm(adapter)
        state.grad_norm_ema = (
            (1 - self.ema_fast) * state.grad_norm_ema + self.ema_fast * grad_norm
        )
        comp.stress = state.grad_norm_ema

        # ── E: Gradient direction entropy ───────────────────────
        current_grad = self._flat_grad(adapter)
        if current_grad is not None and state.prev_grad is not None:
            cos_sim = torch.nn.functional.cosine_similarity(
                current_grad.unsqueeze(0),
                state.prev_grad.unsqueeze(0),
            ).item()
            # Convert: high similarity (1.0) = low entropy = low stress
            entropy = 1.0 - max(0.0, cos_sim)
            state.grad_direction_ema = (
                (1 - self.ema_fast) * state.grad_direction_ema
                + self.ema_fast * entropy
            )
        comp.entropy = state.grad_direction_ema

        # Save current grad for next step
        if current_grad is not None:
            state.prev_grad = current_grad.detach().clone()

        # ── C: Convergence signal ───────────────────────────────
        if loss is not None:
            state.loss_ema = (
                (1 - self.ema_fast) * state.loss_ema + self.ema_fast * loss
            )
            state.loss_ema_slow = (
                (1 - self.ema_slow) * state.loss_ema_slow + self.ema_slow * loss
            )
            # Positive when diverging (fast > slow means loss increasing)
            if state.loss_ema_slow > 1e-8:
                comp.convergence = max(0.0,
                    (state.loss_ema - state.loss_ema_slow) / state.loss_ema_slow
                )

        # ── Combine into φ ──────────────────────────────────────
        phi_raw = (
            self.w_C * comp.convergence
            + self.w_E * comp.entropy
            + self.w_S * comp.stress
        )

        # Adaptive normalization to [0, 1]
        state.phi_raw_history.append(phi_raw)
        comp.phi = self._normalize_phi(phi_raw, state)

        return comp

    def _normalize_phi(self, phi_raw: float, state: AdapterState) -> float:
        """Normalize φ_raw to [0, 1] using running statistics."""
        if len(state.phi_raw_history) < 10:
            return 0.5  # Default during warmup

        history = list(state.phi_raw_history)
        mu = sum(history) / len(history)
        sigma = math.sqrt(
            sum((x - mu) ** 2 for x in history) / len(history)
        )

        if sigma < 1e-8:
            return 0.5

        # Z-score → clamped linear mapping to [0, 1]
        z = (phi_raw - mu) / sigma
        z_clamped = max(-3.0, min(3.0, z))
        return (z_clamped + 3.0) / 6.0

    # ── FSM evaluation ──────────────────────────────────────────

    def _evaluate_fsm(
        self,
        name: str,
        adapter: NestedLoRALinear,
        state: AdapterState,
        phi_comp: PhiComponents,
    ) -> Optional[ModeTransition]:
        """
        Finite State Machine for mode transitions.

        Transitions follow hysteresis rules:
            SINGLE → MULTI:  φ > φ_low  (sustained)
            MULTI  → MIRROR: φ > φ_high (sustained)
            MIRROR → MULTI:  φ < φ_high (sustained) — may restore snapshot
            MULTI  → SINGLE: φ < φ_low  (sustained)

        "Sustained" means hysteresis_steps must have passed in current mode.
        No skip transitions (SINGLE cannot jump directly to MIRROR).
        """
        if state.steps_in_mode < self.hysteresis_steps:
            return None

        phi = state.phi_ema
        old_mode = state.current_mode
        new_mode = old_mode

        # ── Upward transitions (increasing stress) ──────────────
        if old_mode == Mode.SINGLE and phi > self.phi_low:
            new_mode = Mode.MULTI

        elif old_mode == Mode.MULTI and phi > self.phi_high:
            new_mode = Mode.MIRROR

        # ── Downward transitions (recovery) ─────────────────────
        elif old_mode == Mode.MIRROR and phi < self.phi_high:
            new_mode = Mode.MULTI

        elif old_mode == Mode.MULTI and phi < self.phi_low:
            new_mode = Mode.SINGLE

        if new_mode == old_mode:
            return None

        return self._apply_transition(
            name, adapter, state, old_mode, new_mode, phi_comp
        )

    def _apply_transition(
        self,
        name: str,
        adapter: NestedLoRALinear,
        state: AdapterState,
        old_mode: Mode,
        new_mode: Mode,
        phi_comp: PhiComponents,
    ) -> ModeTransition:
        """Execute a mode transition with side effects."""
        old_rank = state.current_rank
        new_rank = self.mode_rank[new_mode]
        snapshot_saved = False
        snapshot_restored = False

        # ── Mirror entry: save stability snapshot ───────────────
        if new_mode == Mode.MIRROR:
            state.mirror_snapshot = {
                "lora_A": adapter.lora_A.data.clone(),
                "lora_B": adapter.lora_B.data.clone(),
                "step": self.global_step,
            }
            snapshot_saved = True

        # ── Mirror exit: optionally restore if stress was transient
        if old_mode == Mode.MIRROR and state.mirror_snapshot is not None:
            if self._should_restore(adapter, state.mirror_snapshot):
                adapter.lora_A.data.copy_(state.mirror_snapshot["lora_A"])
                adapter.lora_B.data.copy_(state.mirror_snapshot["lora_B"])
                snapshot_restored = True
            state.mirror_snapshot = None

        # ── Apply rank change ───────────────────────────────────
        adapter.active_rank = new_rank
        state.current_mode = new_mode
        state.current_rank = new_rank
        state.steps_in_mode = 0

        transition = ModeTransition(
            step=self.global_step,
            layer_name=name,
            old_mode=old_mode,
            new_mode=new_mode,
            old_rank=old_rank,
            new_rank=new_rank,
            phi=phi_comp.phi,
            components=PhiComponents(
                convergence=phi_comp.convergence,
                entropy=phi_comp.entropy,
                stress=phi_comp.stress,
                phi=phi_comp.phi,
            ),
            snapshot_saved=snapshot_saved,
            snapshot_restored=snapshot_restored,
        )

        state.orbit_stack.append(transition)
        return transition

    def _should_restore(
        self,
        adapter: NestedLoRALinear,
        snapshot: dict,
    ) -> bool:
        """
        Decide whether to restore Mirror snapshot on exit.

        If weights drifted significantly during Mirror mode (indicating
        the stress caused real parameter movement), keep the new weights.
        If weights barely changed (transient noise spike), restore the
        stable pre-Mirror state.
        """
        drift_A = (adapter.lora_A.data - snapshot["lora_A"]).norm().item()
        drift_B = (adapter.lora_B.data - snapshot["lora_B"]).norm().item()
        baseline_A = snapshot["lora_A"].norm().item()
        baseline_B = snapshot["lora_B"].norm().item()

        baseline = max(baseline_A + baseline_B, 1e-8)
        relative_drift = (drift_A + drift_B) / baseline

        # If less than 5% relative drift, the stress was transient → restore
        return relative_drift < 0.05

    # ── Gradient utilities ──────────────────────────────────────

    def _grad_norm(self, adapter: NestedLoRALinear) -> float:
        """L2 norm of active LoRA gradients."""
        total = 0.0
        for p in [adapter.lora_A, adapter.lora_B]:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total)

    def _flat_grad(self, adapter: NestedLoRALinear) -> Optional[torch.Tensor]:
        """Concatenated flat gradient vector for direction comparison."""
        grads = []
        for p in [adapter.lora_A, adapter.lora_B]:
            if p.grad is not None:
                grads.append(p.grad.data.flatten())
        if grads:
            return torch.cat(grads)
        return None


# ── Convenience setup ───────────────────────────────────────────


def setup_unified_lora(
    model: nn.Module,
    target_modules: list[str] = None,
    max_rank: int = 16,
    alpha: float = 16.0,
    rank_levels: list[int] = None,
    **controller_kwargs,
) -> Tuple[Dict[str, NestedLoRALinear], OrbitalController]:
    """
    One-call setup: inject NestedLoRA + create OrbitalController with FSM.

    Args:
        model: Base model to adapt.
        target_modules: Layer name patterns (default: ["q_proj", "v_proj"]).
        max_rank: Maximum rank per adapter.
        alpha: LoRA alpha.
        rank_levels: [r_single, r_multi, r_mirror] (default: [4, 8, 16]).
        **controller_kwargs: Passed to OrbitalController (phi_low, phi_high, etc).

    Returns:
        (adapters_dict, controller)

    Usage:
        adapters, ctrl = setup_unified_lora(model)
        for batch in dataloader:
            loss = model(batch).loss
            loss.backward()
            ctrl.step(loss=loss.item())   # Pass loss for convergence signal
            optimizer.step()
    """
    from nested_lora import inject_nested_lora

    if target_modules is None:
        target_modules = ["q_proj", "v_proj"]
    if rank_levels is None:
        rank_levels = [4, 8, 16]

    adapters = inject_nested_lora(
        model, target_modules, max_rank=max_rank, alpha=alpha
    )

    controller = OrbitalController(
        adapters, rank_levels=rank_levels, **controller_kwargs
    )

    return adapters, controller

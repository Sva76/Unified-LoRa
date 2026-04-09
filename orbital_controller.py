"""
OrbitalController — Stress-Driven Rank Adaptation
===================================================
Controls NestedLoRA rank using a physics-inspired orbital model.

Core idea:
    The adapter occupies nested "energy orbitals" (r4 ⊂ r8 ⊂ r16).
    Gradient stress determines promotion/demotion between orbitals.
    Transitions are governed by adaptive thresholds (μ ± kσ) with
    hysteresis, preventing oscillation.

Key features:
    - orbit_stack: history of orbital transitions for diagnostics
    - Adaptive thresholds: mean ± k*std of recent stress window
    - Symmetric return logic: promotes and demotes with equal criteria
    - Per-layer independence: each adapter has its own stress profile

Author: Simona Vargiu
License: Apache 2.0
"""

import torch
import torch.nn as nn
import math
from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from nested_lora import NestedLoRALinear


# ── Data structures ─────────────────────────────────────────────


@dataclass
class OrbitalState:
    """Per-adapter orbital tracking."""
    current_rank: int
    stress_ema: float = 0.0
    stress_history: deque = field(default_factory=lambda: deque(maxlen=100))
    orbit_stack: list = field(default_factory=list)  # [(step, rank, direction)]


@dataclass
class OrbitalTransition:
    """Record of a single orbital change."""
    step: int
    layer_name: str
    old_rank: int
    new_rank: int
    stress: float
    threshold: float
    direction: str  # "promote" or "demote"


# ── Controller ──────────────────────────────────────────────────


class OrbitalController:
    """
    Drives rank adaptation across all NestedLoRA adapters.

    Args:
        adapters: Dict of name → NestedLoRALinear (from inject_nested_lora).
        rank_levels: Allowed rank values, ascending (e.g. [4, 8, 16]).
        ema_alpha: Smoothing factor for stress EMA (0 < α ≤ 1).
        threshold_k: Number of std deviations for promote/demote thresholds.
        window_size: Number of recent stress values for adaptive thresholds.
        eval_interval: Steps between rank evaluations.
        warmup_steps: Steps before any rank changes are allowed.
    """

    def __init__(
        self,
        adapters: Dict[str, NestedLoRALinear],
        rank_levels: List[int] = None,
        ema_alpha: float = 0.1,
        threshold_k: float = 1.5,
        window_size: int = 50,
        eval_interval: int = 10,
        warmup_steps: int = 50,
    ):
        self.adapters = adapters
        self.rank_levels = rank_levels or [4, 8, 16]
        self.ema_alpha = ema_alpha
        self.threshold_k = threshold_k
        self.window_size = window_size
        self.eval_interval = eval_interval
        self.warmup_steps = warmup_steps

        # Per-adapter state
        self.states: Dict[str, OrbitalState] = {}
        for name, adapter in adapters.items():
            initial_rank = self._nearest_level(adapter.active_rank)
            adapter.active_rank = initial_rank
            self.states[name] = OrbitalState(
                current_rank=initial_rank,
                stress_history=deque(maxlen=window_size),
            )

        # Global log
        self.transition_log: List[OrbitalTransition] = []
        self.global_step = 0

    # ── Public API ──────────────────────────────────────────────

    def step(self) -> List[OrbitalTransition]:
        """
        Call once per training step (after backward, before optimizer.step).
        Returns list of transitions that occurred (empty if no changes).
        """
        self.global_step += 1
        transitions = []

        for name, adapter in self.adapters.items():
            state = self.states[name]

            # Compute gradient stress
            stress = self._compute_stress(adapter)
            state.stress_ema = (
                (1 - self.ema_alpha) * state.stress_ema + self.ema_alpha * stress
            )
            state.stress_history.append(state.stress_ema)

            # Only evaluate at intervals and after warmup
            if (
                self.global_step < self.warmup_steps
                or self.global_step % self.eval_interval != 0
            ):
                continue

            transition = self._evaluate_transition(name, adapter, state)
            if transition is not None:
                transitions.append(transition)
                self.transition_log.append(transition)

        return transitions

    def get_summary(self) -> Dict[str, dict]:
        """Return current state of all adapters."""
        summary = {}
        for name, state in self.states.items():
            summary[name] = {
                "rank": state.current_rank,
                "stress_ema": round(state.stress_ema, 6),
                "transitions": len(state.orbit_stack),
            }
        return summary

    def avg_rank(self) -> float:
        """Average active rank across all adapters."""
        ranks = [s.current_rank for s in self.states.values()]
        return sum(ranks) / len(ranks) if ranks else 0.0

    def rank_saving_pct(self) -> float:
        """Percentage rank reduction vs max allocation."""
        max_total = sum(
            self.rank_levels[-1] for _ in self.adapters
        )
        active_total = sum(s.current_rank for s in self.states.values())
        return (1 - active_total / max_total) * 100 if max_total > 0 else 0.0

    # ── Internal logic ──────────────────────────────────────────

    def _compute_stress(self, adapter: NestedLoRALinear) -> float:
        """Gradient L2 norm of active LoRA parameters."""
        total = 0.0
        for p in [adapter.lora_A, adapter.lora_B]:
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total)

    def _evaluate_transition(
        self,
        name: str,
        adapter: NestedLoRALinear,
        state: OrbitalState,
    ) -> Optional[OrbitalTransition]:
        """Check if stress warrants an orbital transition."""
        if len(state.stress_history) < 10:
            return None

        history = list(state.stress_history)
        mu = sum(history) / len(history)
        sigma = math.sqrt(sum((x - mu) ** 2 for x in history) / len(history))

        promote_thresh = mu + self.threshold_k * sigma
        demote_thresh = mu - self.threshold_k * sigma

        current_idx = self._level_index(state.current_rank)

        # Promote: stress above upper threshold → need more capacity
        if state.stress_ema > promote_thresh and current_idx < len(self.rank_levels) - 1:
            new_rank = self.rank_levels[current_idx + 1]
            return self._apply_transition(
                name, adapter, state, new_rank, state.stress_ema, promote_thresh, "promote"
            )

        # Demote: stress below lower threshold → can reduce capacity
        if state.stress_ema < demote_thresh and current_idx > 0:
            new_rank = self.rank_levels[current_idx - 1]
            return self._apply_transition(
                name, adapter, state, new_rank, state.stress_ema, demote_thresh, "demote"
            )

        return None

    def _apply_transition(
        self,
        name: str,
        adapter: NestedLoRALinear,
        state: OrbitalState,
        new_rank: int,
        stress: float,
        threshold: float,
        direction: str,
    ) -> OrbitalTransition:
        """Execute a rank change."""
        old_rank = state.current_rank

        # Apply the slice change
        adapter.active_rank = new_rank
        state.current_rank = new_rank
        state.orbit_stack.append((self.global_step, new_rank, direction))

        return OrbitalTransition(
            step=self.global_step,
            layer_name=name,
            old_rank=old_rank,
            new_rank=new_rank,
            stress=stress,
            threshold=threshold,
            direction=direction,
        )

    def _nearest_level(self, rank: int) -> int:
        """Snap a rank value to the nearest allowed level."""
        return min(self.rank_levels, key=lambda r: abs(r - rank))

    def _level_index(self, rank: int) -> int:
        """Index of rank in rank_levels."""
        try:
            return self.rank_levels.index(rank)
        except ValueError:
            return self.rank_levels.index(self._nearest_level(rank))


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
    One-call setup: inject NestedLoRA + create OrbitalController.

    Args:
        model: Base model to adapt.
        target_modules: Layer name patterns (default: ["q_proj", "v_proj"]).
        max_rank: Maximum rank per adapter.
        alpha: LoRA alpha.
        rank_levels: Allowed orbital levels (default: [4, 8, 16]).
        **controller_kwargs: Passed to OrbitalController.

    Returns:
        (adapters_dict, controller)

    Usage:
        adapters, ctrl = setup_unified_lora(model)
        for batch in dataloader:
            loss = model(batch)
            loss.backward()
            ctrl.step()
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

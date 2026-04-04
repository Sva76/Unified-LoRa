"""
Orbital Controller — Trajectory Control with Memory
=====================================================

Closed-loop rank controller that adapts model capacity based on
observed training stress. Works with any rank-adjustable system
(NestedLoRA, adaptive LR, or API-based training).

This module is the "intelligence" — pure control logic, no model code.
Pair with NestedLoRA for the complete Unified-LoRA system.

Author: Simona Vargiu
License: Apache 2.0
"""

import numpy as np
from typing import Dict, List, Optional


class OrbitalController:
    """
    Closed-loop trajectory controller for dynamic capacity adaptation.

    Unlike threshold-based controllers that map stress to rank statically,
    this implements orbital dynamics with memory:

        Ascend:  stress detected  → jump to higher orbital, push delta
        Hold:    oscillating      → stay, don't move
        Descend: confirmed stable → pop delta, symmetric return

    Each capacity increase is tracked on a stack and reversed only under
    confirmed stability. This prevents premature compression (returning
    too early) and oscillatory collapse (bouncing between ranks).

    The stress signal and thresholds are adaptive — they auto-calibrate
    to any model/task/loss scale without manual tuning.

    Args:
        ranks: Available capacity levels (default: [4, 8, 16])
        warmup: Steps at max capacity to build EMA baseline
        stable_window: Consecutive stable steps required for descent

    Example:
        >>> from nested_lora import inject_nested_lora, set_rank
        >>> from orbital_controller import OrbitalController
        >>>
        >>> model = inject_nested_lora(model, max_rank=16)
        >>> ctrl = OrbitalController()
        >>>
        >>> for step, batch in enumerate(loader):
        ...     loss = model(**batch).loss
        ...     new_rank = ctrl.step(loss.item())
        ...     set_rank(model, new_rank)
        ...     loss.backward()
        ...     optimizer.step()
    """

    def __init__(
        self,
        ranks: Optional[List[int]] = None,
        warmup: int = 10,
        stable_window: int = 6,
    ):
        self.RANKS = ranks or [4, 8, 16]
        self.warmup = warmup
        self.stable_window = stable_window
        self.reset()

    def reset(self):
        """Reset controller to initial state."""
        self.rank = self.RANKS[-1]
        self.orbit_stack = []
        self.loss_ema = 0.0
        self.prev_loss = None
        self.phi_hist = []
        self.stable_count = 0
        self.step_count = 0
        self.post_warmup = False

        self.history = {
            "rank": [],
            "phi": [],
            "stable_count": [],
        }

    # ── Stress signal ───────────────────────────────

    def _compute_phi(self, loss: float) -> float:
        """
        Stress signal from loss trajectory.

        φ = |loss - EMA| + 2.0 × max(0, loss - prev_loss)

        Combines deviation from trend (general instability)
        with spike detection (sudden deterioration).
        """
        self.loss_ema = 0.9 * self.loss_ema + 0.1 * loss
        delta = abs(loss - self.loss_ema)
        spike = max(0.0, loss - self.prev_loss) if self.prev_loss is not None else 0.0
        self.prev_loss = loss
        return delta + 2.0 * spike

    def _thresholds(self):
        """
        Adaptive thresholds from running statistics.

        t_stress = μ + 0.7σ  (above this → ascend)
        t_stable = μ - 0.3σ  (below this → stability confirmed)

        Auto-calibrates to loss scale. No manual tuning.
        """
        if len(self.phi_hist) < 10:
            return 0.15, 0.04
        recent = self.phi_hist[-40:]
        mu = np.mean(recent)
        sigma = np.std(recent) + 1e-8
        t_stress = mu + 0.7 * sigma
        t_stable = max(mu - 0.3 * sigma, 0.0)
        return t_stress, t_stable

    # ── Core logic ──────────────────────────────────

    def _rank_index(self) -> int:
        return self.RANKS.index(self.rank)

    def step(self, loss: float) -> int:
        """
        Called once per training step. Returns the capacity level to use.

        Args:
            loss: Current step loss value

        Returns:
            int: Active rank (or capacity level) for next step
        """
        self.step_count += 1

        # First step: initialize EMA
        if self.prev_loss is None:
            self.loss_ema = loss
            self.prev_loss = loss
            self._log(0.0)
            return self.rank

        phi = self._compute_phi(loss)
        self.phi_hist.append(phi)

        # Warmup: build baseline at max capacity
        if self.step_count <= self.warmup:
            self._log(phi)
            return self.rank

        # Transition: warmup → ground state
        if not self.post_warmup:
            self.post_warmup = True
            self.rank = self.RANKS[0]
            self.orbit_stack = []
            self.stable_count = 0
            self._log(phi)
            return self.rank

        t_stress, t_stable = self._thresholds()

        # Stability counter
        if phi <= t_stable:
            self.stable_count += 1
        elif phi > t_stress:
            self.stable_count = 0
        else:
            self.stable_count = max(0, self.stable_count - 1)

        # ASCEND: stress → jump to higher orbital
        if phi > t_stress and self.rank < self.RANKS[-1]:
            idx = self._rank_index()
            new_idx = min(idx + 1, len(self.RANKS) - 1)
            new_rank = self.RANKS[new_idx]
            if new_rank != self.rank:
                self.orbit_stack.append(new_rank - self.rank)
                self.rank = new_rank
                self.stable_count = 0
            self._log(phi)
            return self.rank

        # DESCEND: confirmed stability → symmetric return
        if self.stable_count >= self.stable_window and self.orbit_stack:
            delta = self.orbit_stack.pop()
            target = self.rank - delta
            self.rank = min(self.RANKS, key=lambda r: abs(r - target))
            self.rank = max(self.rank, self.RANKS[0])
            self.stable_count = 0
            self._log(phi)
            return self.rank

        # HOLD: neutral → don't move
        self._log(phi)
        return self.rank

    # ── Introspection ───────────────────────────────

    def _log(self, phi: float):
        self.history["rank"].append(self.rank)
        self.history["phi"].append(phi)
        self.history["stable_count"].append(self.stable_count)

    def get_state(self) -> Dict:
        """Current controller state."""
        return {
            "rank": self.rank,
            "step": self.step_count,
            "orbit_stack": list(self.orbit_stack),
            "stable_count": self.stable_count,
            "phi": self.phi_hist[-1] if self.phi_hist else 0.0,
        }

    def get_history(self) -> Dict[str, list]:
        """Complete training history."""
        return self.history

    def __repr__(self) -> str:
        return (
            f"OrbitalController(step={self.step_count}, rank={self.rank}, "
            f"stack={self.orbit_stack}, stable={self.stable_count})"
        )


# ============================================================
# CONVENIENCE: setup helper
# ============================================================

def setup_unified_lora(model, max_rank=16, ranks=None, warmup=10, stable_window=6):
    """
    One-call setup: inject NestedLoRA + create OrbitalController.

    Args:
        model: PyTorch model
        max_rank: Maximum LoRA rank
        ranks: Available rank levels
        warmup: Controller warmup steps
        stable_window: Steps of stability before descent

    Returns:
        (model, controller) tuple

    Example:
        >>> from orbital_controller import setup_unified_lora
        >>> from nested_lora import set_rank
        >>>
        >>> model, ctrl = setup_unified_lora(model)
        >>> for step, batch in enumerate(loader):
        ...     loss = model(**batch).loss
        ...     set_rank(model, ctrl.step(loss.item()))
        ...     loss.backward(); optimizer.step(); optimizer.zero_grad()
    """
    from nested_lora import inject_nested_lora

    model = inject_nested_lora(model, max_rank)
    controller = OrbitalController(
        ranks=ranks or [4, 8, 16],
        warmup=warmup,
        stable_window=stable_window,
    )
    return model, controller


# ============================================================
# DEMO
# ============================================================

if __name__ == "__main__":
    print("Orbital Controller — Demo")
    print("=" * 50)
    print("Simulating: 30 stable → 10 shock → 30 recovery\n")

    ctrl = OrbitalController(warmup=8, stable_window=5)

    for step in range(70):
        if step < 30:
            loss = np.random.uniform(0.4, 0.6)
        elif step < 40:
            loss = np.random.uniform(1.5, 3.0)
        else:
            loss = np.random.uniform(0.3, 0.5)

        rank = ctrl.step(loss)

        if step % 5 == 0 or step == 30:
            s = ctrl.get_state()
            tag = " <<<SHOCK" if step == 30 else ""
            print(f"  [{step:3d}] rank={rank:2d}  phi={s['phi']:.3f}  stack={s['orbit_stack']}{tag}")

    print(f"\nFinal: {ctrl}")

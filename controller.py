"""
Unified LoRA — Nested Orbital Controller
==========================================

Adaptive parameter-efficient fine-tuning with dynamic rank control.

Architecture: Single LoRA adapter pair (A, B) with rank controlled via slicing.
    r4 ⊂ r8 ⊂ r16 — one particle, multiple orbitals.
    Descending = pausing dimensions, not destroying them. Zero cold start.

Controller: Closed-loop trajectory controller with orbital memory.
    Stress  → ascend to higher orbital, push delta to stack
    Stable  → pop delta, symmetric return to lower orbital
    Neutral → hold position

Author: Simona Vargiu
License: Apache 2.0
"""

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


# ============================================================
# NESTED LoRA — ONE PARTICLE, MULTIPLE ORBITALS
# ============================================================

class NestedLoRALinear(nn.Module):
    """
    Single LoRA adapter with dynamic rank via slicing.

    Instead of separate adapters for each rank (which causes cold start
    on transitions), a single pair of matrices A and B is shared.
    The active rank is controlled by slicing:

        r=4  → A[:4, :], B[:, :4]
        r=8  → A[:8, :], B[:, :8]
        r=16 → A[:16,:], B[:, :16]

    When descending from r=16 to r=4, dimensions 0-3 retain all
    learned weights. Dimensions 4-15 are paused, not destroyed.
    When ascending back, they resume exactly where they left off.

    Args:
        linear: Original nn.Linear layer to wrap
        max_rank: Maximum LoRA rank (default: 16)
    """

    def __init__(self, linear: nn.Linear, max_rank: int = 16):
        super().__init__()
        self.linear = linear
        self.max_rank = max_rank
        self.active_rank = max_rank

        # Freeze original weights
        for p in self.linear.parameters():
            p.requires_grad = False

        # One particle: single A and B
        self.lora_A = nn.Parameter(torch.empty(max_rank, linear.in_features))
        self.lora_B = nn.Parameter(torch.zeros(linear.out_features, max_rank))

        # Standard LoRA init: A = kaiming, B = zeros → initial delta = 0
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def set_rank(self, r: int):
        """Set the active orbital (rank). Must be <= max_rank."""
        self.active_rank = min(r, self.max_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.linear(x)
        r = self.active_rank

        # Slice = same particle, smaller orbital
        h = F.linear(x, self.lora_A[:r, :])         # (batch, r)
        delta = F.linear(h, self.lora_B[:, :r])      # (batch, out)

        # Scale: maintain output magnitude across ranks
        scale = self.max_rank / r

        return base + delta * scale


def inject_nested_lora(model: nn.Module, max_rank: int = 16) -> nn.Module:
    """
    Replace attention Linear layers with NestedLoRALinear.

    Args:
        model: PyTorch model
        max_rank: Maximum LoRA rank

    Returns:
        Model with NestedLoRA injected into attention layers
    """
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and "attention" in name:
            parent = model
            *path, last = name.split(".")
            for p in path:
                parent = getattr(parent, p)
            setattr(parent, last, NestedLoRALinear(module, max_rank))
    return model


def set_rank(model: nn.Module, r: int):
    """Set active rank on all NestedLoRALinear modules."""
    for m in model.modules():
        if isinstance(m, NestedLoRALinear):
            m.set_rank(r)


# ============================================================
# ORBITAL CONTROLLER — TRAJECTORY WITH MEMORY
# ============================================================

class OrbitalController:
    """
    Closed-loop trajectory controller for dynamic rank adaptation.

    Unlike threshold-based controllers (AdaLoRA, schedule-based),
    this implements a state machine with orbital memory:

        Ascend:  stress detected  → jump to higher orbital, push delta
        Hold:    oscillating      → stay, don't move
        Descend: confirmed stable → pop delta, symmetric return

    The key insight: each capacity increase is tracked and reversed
    only under confirmed stability, preventing premature compression
    and oscillatory collapse.

    "I climb → I remember. I stabilize → I return exactly.
     I oscillate → I don't move."

    Args:
        ranks: Available rank levels (default: [4, 8, 16])
        warmup: Steps at max rank before controller activates
        stable_window: Consecutive stable steps required for descent

    Example:
        >>> ctrl = OrbitalController()
        >>> for step in range(num_steps):
        ...     loss = train_step(model, batch)
        ...     new_rank = ctrl.step(loss)
        ...     set_rank(model, new_rank)
    """

    def __init__(
        self,
        ranks: List[int] = None,
        warmup: int = 10,
        stable_window: int = 6,
    ):
        self.RANKS = ranks or [4, 8, 16]
        self.warmup = warmup
        self.stable_window = stable_window
        self.reset()

    def reset(self):
        """Reset controller to initial state."""
        self.rank = self.RANKS[-1]    # start at max during warmup
        self.orbit_stack = []          # stack of deltas (orbital memory)
        self.loss_ema = 0.0
        self.prev_loss = None
        self.phi_hist = []
        self.stable_count = 0
        self.step_count = 0
        self.post_warmup = False

        # History tracking
        self.history = {
            "rank": [],
            "phi": [],
            "lr_label": [],
            "stable_count": [],
        }

    def _compute_phi(self, loss: float) -> float:
        """
        Compute stress signal from loss trajectory.

        phi = |loss - EMA| + 2.0 * max(0, loss - prev_loss)

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
        Adaptive thresholds that auto-calibrate to loss scale.

        Uses running statistics (mu, sigma) of phi history.
        No manual tuning needed across different models/tasks.
        """
        if len(self.phi_hist) < 10:
            return 0.15, 0.04          # conservative defaults
        recent = self.phi_hist[-40:]
        mu = np.mean(recent)
        sigma = np.std(recent) + 1e-8
        t_stress = mu + 0.7 * sigma
        t_stable = max(mu - 0.3 * sigma, 0.0)
        return t_stress, t_stable

    def _rank_index(self) -> int:
        return self.RANKS.index(self.rank)

    def step(self, loss: float) -> int:
        """
        Called once per training step. Returns the rank to use.

        Args:
            loss: Current step loss value

        Returns:
            int: Active rank for next step
        """
        self.step_count += 1

        # --- First step: initialize ---
        if self.prev_loss is None:
            self.loss_ema = loss
            self.prev_loss = loss
            self._log(0.0)
            return self.rank

        phi = self._compute_phi(loss)
        self.phi_hist.append(phi)

        # --- Warmup: build EMA baseline at max rank ---
        if self.step_count <= self.warmup:
            self._log(phi)
            return self.rank

        # --- Transition: warmup → ground state ---
        if not self.post_warmup:
            self.post_warmup = True
            self.rank = self.RANKS[0]   # drop to ground state
            self.orbit_stack = []
            self.stable_count = 0
            self._log(phi)
            return self.rank

        t_stress, t_stable = self._thresholds()

        # --- Stability counter ---
        if phi <= t_stable:
            self.stable_count += 1
        elif phi > t_stress:
            self.stable_count = 0
        else:
            self.stable_count = max(0, self.stable_count - 1)

        # --- ASCEND: stress → orbital jump ---
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

        # --- DESCEND: confirmed stability → symmetric return ---
        if self.stable_count >= self.stable_window and self.orbit_stack:
            delta = self.orbit_stack.pop()
            target = self.rank - delta
            self.rank = min(self.RANKS, key=lambda r: abs(r - target))
            self.rank = max(self.rank, self.RANKS[0])
            self.stable_count = 0
            self._log(phi)
            return self.rank

        # --- HOLD: oscillating or neutral → don't move ---
        self._log(phi)
        return self.rank

    def _log(self, phi: float):
        """Record step in history."""
        self.history["rank"].append(self.rank)
        self.history["phi"].append(phi)
        self.history["stable_count"].append(self.stable_count)

    def get_state(self) -> Dict:
        """Get current controller state."""
        return {
            "rank": self.rank,
            "step": self.step_count,
            "orbit_stack": list(self.orbit_stack),
            "stable_count": self.stable_count,
            "phi": self.phi_hist[-1] if self.phi_hist else 0.0,
        }

    def get_history(self) -> Dict[str, list]:
        """Get complete training history."""
        return self.history

    def __repr__(self) -> str:
        return (
            f"OrbitalController(step={self.step_count}, rank={self.rank}, "
            f"stack={self.orbit_stack}, stable={self.stable_count})"
        )


# ============================================================
# CONVENIENCE: COMBINED USAGE
# ============================================================

def setup_unified_lora(
    model: nn.Module,
    max_rank: int = 16,
    ranks: List[int] = None,
    warmup: int = 10,
    stable_window: int = 6,
):
    """
    One-call setup: inject NestedLoRA and create OrbitalController.

    Args:
        model: PyTorch model to adapt
        max_rank: Maximum LoRA rank
        ranks: Available rank levels (default: [4, 8, 16])
        warmup: Controller warmup steps
        stable_window: Steps of stability before descent

    Returns:
        (model, controller) tuple

    Example:
        >>> model, ctrl = setup_unified_lora(model)
        >>> optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        >>> for step, batch in enumerate(loader):
        ...     loss = model(**batch).loss
        ...     new_rank = ctrl.step(loss.item())
        ...     set_rank(model, new_rank)
        ...     loss.backward()
        ...     optimizer.step()
        ...     optimizer.zero_grad()
    """
    model = inject_nested_lora(model, max_rank)
    controller = OrbitalController(
        ranks=ranks or [4, 8, 16],
        warmup=warmup,
        stable_window=stable_window,
    )
    return model, controller


# ============================================================
# EXAMPLE
# ============================================================

if __name__ == "__main__":
    print("Unified LoRA — Nested Orbital Controller")
    print("=" * 50)

    ctrl = OrbitalController(warmup=10, stable_window=6)

    # Simulate: stable training → shock → recovery
    print("\nSimulating: 40 steps stable → SHOCK → 40 steps recovery\n")

    for step in range(80):
        if step < 40:
            loss = np.random.uniform(0.4, 0.6)
        elif step < 50:
            loss = np.random.uniform(1.5, 3.0)   # SHOCK
        else:
            loss = np.random.uniform(0.3, 0.5)    # recovery

        rank = ctrl.step(loss)

        if step % 5 == 0 or step == 40:
            state = ctrl.get_state()
            marker = " <<<SHOCK" if step == 40 else ""
            print(
                f"  [{step:3d}] rank={rank:2d}  "
                f"phi={state['phi']:.3f}  "
                f"stack={state['orbit_stack']}"
                f"{marker}"
            )

    print(f"\nFinal: {ctrl}")

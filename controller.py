"""
Unified LoRA Controller
========================

Adaptive parameter-efficient fine-tuning controller with automatic 
Single/Multi/Mirror mode switching based on synaptic stress signals.

Author: Simona Vargiu
License: Apache 2.0
"""

import torch
from typing import Dict, Optional, Tuple


class UnifiedController:
    """
    Unified LoRA adaptive controller.
    
    Monitors training stress via synaptic signal φ(t) and automatically
    switches between three operational modes:
    - Mode 0 (Single): Shared adapter for low conflict
    - Mode 1 (Multi): Task-specific adapters for moderate stress
    - Mode 2 (Mirror): Stability snapshots for catastrophic forgetting
    
    Args:
        alpha (float): Learning rate for φ(t) updates (default: 0.1)
        beta (float): EMA smoothing factor for loss (default: 0.9)
        theta0 (float): Single/Multi threshold (default: 0.3)
        theta1 (float): Multi/Mirror threshold (default: 0.7)
        lr_single (float): Learning rate for Single mode (default: 5e-5)
        lr_multi (float): Learning rate for Multi mode (default: 3e-5)
        lr_mirror (float): Learning rate for Mirror mode (default: 1e-5)
        
    Example:
        >>> controller = UnifiedController()
        >>> for step, batch in enumerate(train_loader):
        ...     outputs = model(**batch)
        ...     new_lr = controller.update(outputs.loss.item())
        ...     # Apply new_lr to optimizer
    """
    
    def __init__(
        self,
        alpha: float = 0.1,
        beta: float = 0.9,
        theta0: float = 0.3,
        theta1: float = 0.7,
        lr_single: float = 5e-5,
        lr_multi: float = 3e-5,
        lr_mirror: float = 1e-5,
    ):
        self.alpha = alpha
        self.beta = beta
        self.theta0 = theta0
        self.theta1 = theta1
        
        # Learning rates per mode
        self.lr_map = {
            0: lr_single,
            1: lr_multi,
            2: lr_mirror,
        }
        
        # State variables
        self.phi = 0.5  # Synaptic stress signal
        self.E_smooth = 1.0  # Smoothed loss
        self.mode = 1  # Current mode (start with Multi)
        self.step = 0
        
        # History tracking
        self.history = {
            "phi": [],
            "E_smooth": [],
            "mode": [],
            "step": [],
        }
    
    def update(self, loss: float) -> float:
        """
        Update controller state and return new learning rate.
        
        Args:
            loss (float): Current training loss
            
        Returns:
            float: New learning rate based on current mode
        """
        self.step += 1
        
        # Update smoothed loss (EMA)
        E = float(loss)
        self.E_smooth = self.beta * self.E_smooth + (1 - self.beta) * E
        
        # Compute normalized stress signal
        D = self.E_smooth / (1 + self.E_smooth)  # Normalize to [0,1]
        
        # Update synaptic signal φ(t) with EMA
        self.phi = (1 - self.alpha) * self.phi + self.alpha * D
        
        # FSM: Determine mode based on φ(t)
        if self.phi < self.theta0:
            self.mode = 0  # Single
        elif self.phi < self.theta1:
            self.mode = 1  # Multi
        else:
            self.mode = 2  # Mirror
        
        # Log history
        self.history["phi"].append(self.phi)
        self.history["E_smooth"].append(self.E_smooth)
        self.history["mode"].append(self.mode)
        self.history["step"].append(self.step)
        
        # Return learning rate for current mode
        return self.lr_map[self.mode]
    
    def get_state(self) -> Dict[str, float]:
        """
        Get current controller state.
        
        Returns:
            dict: Current values of phi, E_smooth, mode, step
        """
        return {
            "phi": self.phi,
            "E_smooth": self.E_smooth,
            "mode": self.mode,
            "step": self.step,
        }
    
    def get_history(self) -> Dict[str, list]:
        """
        Get complete training history.
        
        Returns:
            dict: History of phi, E_smooth, mode, step
        """
        return self.history
    
    def reset(self):
        """Reset controller to initial state."""
        self.phi = 0.5
        self.E_smooth = 1.0
        self.mode = 1
        self.step = 0
        self.history = {
            "phi": [],
            "E_smooth": [],
            "mode": [],
            "step": [],
        }
    
    @staticmethod
    def mode_name(mode: int) -> str:
        """
        Get human-readable mode name.
        
        Args:
            mode (int): Mode number (0, 1, or 2)
            
        Returns:
            str: Mode name
        """
        names = {0: "Single", 1: "Multi", 2: "Mirror"}
        return names.get(mode, "Unknown")
    
    def __repr__(self) -> str:
        """String representation of controller state."""
        return (
            f"UnifiedController(step={self.step}, phi={self.phi:.3f}, "
            f"mode={self.mode} ({self.mode_name(self.mode)}), "
            f"E_smooth={self.E_smooth:.3f})"
        )


# Example usage
if __name__ == "__main__":
    import numpy as np
    
    print("Unified LoRA Controller - Example")
    print("=" * 50)
    
    controller = UnifiedController()
    
    # Simulate training with stress events
    print("\nSimulating training with SHOCK at step 150...")
    print()
    
    for step in range(300):
        # Simulate loss
        if step < 150:
            loss = np.random.uniform(0.4, 0.6)  # Normal training
        else:
            loss = np.random.uniform(2.0, 4.0)  # SHOCK
        
        # Update controller
        new_lr = controller.update(loss)
        
        # Log every 50 steps
        if step % 50 == 0:
            state = controller.get_state()
            print(
                f"[{step:3d}] phi={state['phi']:.3f} | "
                f"mode={state['mode']} ({controller.mode_name(state['mode'])}) | "
                f"lr={new_lr:.1e}"
            )
    
    print("\n" + "=" * 50)
    print("Simulation complete!")
    print(f"\nFinal state: {controller}")

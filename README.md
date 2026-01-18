# Unified LoRA

**Adaptive Single/Multi/Mirror LoRA framework with automatic mode switching driven by synaptic stress signals.**

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

## Overview

Unified LoRA is a dynamic parameter-efficient fine-tuning system that automatically switches between three operational modes based on training stress:

- **Mode 0 (Single)**: Shared adapter for low-conflict scenarios
- **Mode 1 (Multi)**: Task-specific LoRA adapters for moderate stress
- **Mode 2 (Mirror)**: Stability snapshots for catastrophic forgetting prevention

The system uses a synaptic control parameter **φ(t)** derived from:
- **C**: Task conflict (weight space variance)
- **E**: Multi-task error
- **S**: Memory stability

φ(t) = f(C, E, S, ΔC, ΔE, ΔS)

## Validation Results

### 🧪 Tinker run (Llama-3.2-1B, cloud GPU)

A controlled run demonstrating a full **stress → recovery cycle** and adaptive mode switching.  
This experiment illustrates the behavior of **φ(t)** under induced instability; it is **not a production deployment claim**.

[250] Mode=1 φ=0.333 E_s=0.500 lr=5e-4  (Multi stable)  
>>> SHOCK @ step 300  
[350] Mode=2 φ=0.827 E_s=4.979 lr=1e-4  (Mirror activated)  
>>> RECOVERY @ step 500  
[550] Mode=1 φ=0.371 E_s=0.521 lr=5e-4  (Multi return)  
[700] Mode=1 φ=0.333 E_s=0.500 lr=5e-4  (baseline restored)

**Key observation (this run):**  
φ returns close to its pre-shock regime after recovery (example: 0.33 → 0.83 → 0.33), indicating reversible stress handling within this setup.

### 📊 Standard Benchmark (GLUE MRPC + DistilBERT)

Observed comparable performance to baseline LoRA on GLUE MRPC under the reported configuration:

| Method | F1 | Accuracy | φ final | Mode |
|--------|-----|----------|---------|------|
| Baseline LoRA | 0.785 | 0.646 | - | - |
| Unified LoRA | 0.785 | 0.646 | 0.367 | 1 |

Key observation (this run):  
No performance degradation observed with adaptive control active in the reported setup.

## Quick Start

from unified_lora import UnifiedController

# Initialize controller
controller = UnifiedController(
    alpha=0.1,      # φ(t) learning rate
    beta=0.9,       # EMA smoothing
    theta0=0.3,     # Single/Multi threshold
    theta1=0.7      # Multi/Mirror threshold
)

# Training loop
for step, batch in enumerate(train_loader):
    outputs = model(**batch)
    loss = outputs.loss

    # Update controller and get adaptive LR
    new_lr = controller.update(loss.item())

    # Apply new learning rate
    for g in optimizer.param_groups:
        g['lr'] = new_lr

    # Standard backprop
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

## Technical Details

### FSM State Transitions

φ < 0.3  → Mode 0 (Single)   LR = 5e-5  
φ < 0.7  → Mode 1 (Multi)    LR = 3e-5  
φ ≥ 0.7  → Mode 2 (Mirror)   LR = 1e-5  

### Stress Signal Computation

The metrics C (conflict), E (error), and S (stability) are normalized to [0,1] to keep φ(t) well-conditioned:

E_smooth = β * E_smooth + (1 - β) * loss  
D = E_smooth / (1 + E_smooth)  
φ = (1 - α) * φ + α * D  

This normalization supports stable FSM transitions and reduces numerical sensitivity during training.

## Installation

pip install transformers peft torch

## Citation

@software{unified_lora_2025,
  author = {Simona Vargiu},
  title = {Unified LoRA: Adaptive Parameter-Efficient Fine-Tuning},
  year = {2025},
  url = {https://github.com/Sva76/Unified-LoRA}
}

## License

Apache License 2.0 — see LICENSE for details.

## Contact

Simona Vargiu (Independent Researcher)  
For collaboration inquiries: simona.vargiu.malta@gmail.com

Status: Demonstrated on (1) a Tinker Llama-3.2-1B run showing adaptive stress → recovery behavior and (2) the GLUE MRPC benchmark (DistilBERT) with comparable F1/Accuracy to baseline LoRA under the reported configuration. Broader benchmarks and statistical evaluation across seeds and tasks are ongoing. Open to research collaboration and extended evaluation.

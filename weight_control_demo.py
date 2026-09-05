"""Historical weight-control narrative; this file contains no executable demo.

The original prose below is retained for provenance. Its claims of collapse
prevention and bounded weights are not established by these descriptive
measurements and are superseded by the controller falsification documented in
README.md and validation/corrections_2026_09.md.

Unified-LoRA does not only improve performance metrics — it directly controls weight updates (ΔW).

We measure this using three signals:
- Weight Drift: L2 distance from initial weights
- Weight Velocity: step-to-step change magnitude
- Weight Norm Growth: overall magnitude expansion

Under aggressive training (Qwen 7B, LR=3e-4):

Final Weight Drift:
- Baseline: 11.39
- Unified: 6.01  → 1.9× lower

Mean Weight Velocity:
- Baseline: 0.144
- Unified: 0.047 → 3× slower updates

Weight Norm Growth:
- Baseline: +3.40
- Unified: +1.02 → controlled expansion

Drift over time:

Step    Baseline    Unified
0       0.39        0.39
100     8.83        5.75
200     10.65       5.94
300     11.39       6.01

Interpretation:

Baseline weights grow continuously and drift away from their initial state.
Unified-LoRA actively constrains weight updates, keeping them bounded.

Reduced weight drift → prevents collapse  
Reduced velocity → avoids instability spikes

This demonstrates that Unified-LoRA operates as a weight control system,
not just a parameter-efficient fine-tuning method.

Without control: weights diverge.  
With Unified-LoRA: weights remain bounded.
"""

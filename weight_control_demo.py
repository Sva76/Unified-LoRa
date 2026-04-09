# ============================================================
# UNIFIED-LoRA — WEIGHT CONTROL DEMO
# Mostra: controllo reale dei pesi (ΔW)
# ============================================================

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(0)

# ============================================================
# MODEL SEMPLICE
# ============================================================
class SimpleModel(nn.Module):
    def __init__(self, d=128):
        super().__init__()
        self.linear = nn.Linear(d, d)

    def forward(self, x):
        return self.linear(x)

# ============================================================
# UNIFIED CONTROLLER (semplice)
# ============================================================
class WeightController:
    def __init__(self, threshold=1.0, damping=0.5):
        self.threshold = threshold
        self.damping = damping

    def apply(self, grad):
        norm = grad.norm().item()

        if norm > self.threshold:
            return grad * self.damping
        return grad

# ============================================================
# TRAIN LOOP
# ============================================================
def run(control=False):
    model = SimpleModel().cuda()
    opt = torch.optim.SGD(model.parameters(), lr=1.0)

    controller = WeightController()

    weight_norms = []

    for step in range(100):
        x = torch.randn(32, 128).cuda()

        out = model(x)
        loss = (out ** 2).mean()

        loss.backward()

        # CONTROLLO PESI
        if control:
            for p in model.parameters():
                if p.grad is not None:
                    p.grad = controller.apply(p.grad)

        opt.step()
        opt.zero_grad()

        # misura ΔW
        total_norm = 0
        for p in model.parameters():
            total_norm += p.data.norm().item()

        weight_norms.append(total_norm)

    return weight_norms

# ============================================================
# RUN
# ============================================================
baseline = run(control=False)
controlled = run(control=True)

# ============================================================
# PLOT
# ============================================================
plt.plot(baseline, label="Baseline")
plt.plot(controlled, label="Unified-Controlled")
plt.legend()
plt.title("Weight Growth (ΔW)")
plt.xlabel("Step")
plt.ylabel("Weight Norm")
plt.show()

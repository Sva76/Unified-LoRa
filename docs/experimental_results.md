## 📊 Experimental Evidence — Rank Dynamics under Disturbance

This section summarizes the **qualitative experimental evidence** supporting the design of **Unified-LoRA**, focusing on *rank dynamics* rather than downstream accuracy.

The goal is **not** to compete on SOTA benchmarks, but to demonstrate a **structural difference** in how model capacity is controlled during fine-tuning.

---

## Experimental Setting

All methods were evaluated under **identical conditions**:

- **Model:** `Qwen/Qwen3-4B-Instruct-2507`
- **Task:** GLUE CoLA (classification, autoregressive formulation)
- **Environment:** Tinker (black-box setting — loss not directly observable)
- **Hardware:** Standard cloud GPU (T4-class)
- **Training length:** ~60 steps per method

This setup reflects realistic **API-based / enterprise fine-tuning**, where internal loss signals are not exposed.

---

## Methods Compared

| Method | Category | Control Logic |
|------|---------|---------------|
| Standard LoRA | Baseline | Fixed rank |
| Schedule-free / Fixed Rank | Baseline+ | Fixed rank, optimized LR |
| AdaLoRA-like | Open-loop adaptive | Rank = function of time |
| **Unified-LoRA (proposed)** | **Closed-loop continuous** | **Rank = function of stress** |

---

## Rank Dynamics — Comparative Analysis

### Axes
- **X-axis:** training step (0 → ~60)
- **Y-axis:** effective LoRA rank

### 1️⃣ AdaLoRA-like (budget-based)

- Stepwise, monotonic decreasing trajectory  
- Starts at **rank = 32**
- Slowly decays according to a predefined schedule
- At step ~60 remains around **rank ≈ 23–24**
- **No reaction** to shocks or dynamic changes

**Interpretation:**  
Adaptive *offline*, but **blind to the real training state**. Rank allocation follows a schedule, not feedback.

---

### 2️⃣ Schedule-free / Standard LoRA

- Flat trajectory
- **Fixed rank = 16**
- No dynamics, no feedback, no adaptation

**Interpretation:**  
A stable but **capacity-blind baseline**. Learning rate optimization cannot compensate for lack of structural flexibility.

---

### 3️⃣ Unified-LoRA (loss-proxy + injected shocks)

- Continuous, **non-monotonic** trajectory
- Starts from **rank = 6** (minimum capacity)
- Progressively grows up to **rank ≈ 31**
- **Immediate reaction** to injected disturbances (e.g. steps ~20, ~30, ~45)
- No unstable oscillations observed

**Interpretation:**  
True **closed-loop control** over model capacity. Rank adapts to *observed stress*, not to a predefined schedule.

---

## 📌 Key Observation — Disturbance Rejection

| Method | Shock Reaction | Stability | Recovery |
|------|----------------|----------|----------|
| Standard / Schedule-free | ❌ None | Passive | — |
| AdaLoRA-like | ⚠️ Indirect, delayed | Partial | Limited |
| **Unified-LoRA** | ✅ Immediate | Stable | Immediate |

👉 **Only Unified-LoRA exhibits disturbance rejection**, a property expected from closed-loop control systems and absent in open-loop approaches.

---

## Control-Theoretic Interpretation

- **Standard / Schedule-free / AdaLoRA:** open-loop control  
- **Unified-LoRA:** closed-loop continuous control

Formally:

Standard / AdaLoRA: rank = f(step)
Unified-LoRA: rank = f(stress(step, history))


Where **stress** is a continuous, smoothed, normalized signal derived from observable training dynamics.

---

## Why Black-Box Matters

Unified-LoRA operates **without direct access to the loss**.

In Tinker-like environments, the system observes *trajectory-level signals*, not internal optimization variables.

> “I observe the missile trajectory, not the engine — yet I can still control it.”

This capability is critical for:
- API-based fine-tuning
- enterprise training pipelines
- safety- or cost-constrained environments

---

## Computational Overhead

Unified-LoRA introduces:

- **O(1)** computation per step
- No SVD
- No matrix decomposition
- Negligible overhead relative to the training step

---

## Takeaway

Unified-LoRA is:
- **not** a scheduler
- **not** a rank budget
- **not** a learning-rate trick

It implements a **dynamic controller over model capacity**.

At equal training conditions:
- higher stability
- better resource utilization

Under disturbances:
- **it is the only method that reacts correctly**

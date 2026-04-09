Unified-LoRA


Unified-LoRA is not a LoRA optimization — it is a weight control system.


Instead of only reducing parameters, it actively regulates weight updates (ΔW), controlling both their magnitude (via adaptive learning rate) and their subspace (via nested rank).

This transforms fine-tuning from an unstable process into a controlled dynamic system.



The Idea


Standard LoRA treats rank as capacity.


Unified-LoRA treats rank as control over ΔW.


Biological systems don’t just scale activity — they switch modes depending on stress.

Unified-LoRA brings the same principle to fine-tuning:


φ(t) low    →  SINGLE mode  →  constrained ΔW (rank 4)
φ(t) medium →  MULTI mode   →  adaptive ΔW    (rank 8)
φ(t) high   →  MIRROR mode  →  maximum ΔW + snapshot (rank 16)
                                 ↳ rollback if instability was transient



The system doesn’t just change rank — it changes how weights evolve.



φ(t): The Synaptic Stress Signal


A composite instability signal:


φ(t) = w_C · C(t) + w_E · E(t) + w_S · S(t)





C (Convergence): fast vs slow loss EMA


E (Entropy): gradient direction alignment


S (Stress): gradient magnitude




φ is normalized dynamically → works across models and scales.



Weight Control (ΔW)


Unified-LoRA explicitly controls:


ΔW = A_r @ B_r



Two levers:




Rank (NestedLoRA) → controls update subspace


LR (Orbital controller) → controls update magnitude




Result:


direction control + magnitude control = ΔW control




NestedLoRA — Control of ΔW Subspace


Single allocation, nested slicing:


A ∈ ℝ^(d × 16), B ∈ ℝ^(16 × d)

r=4   → A[:, :4]  @ B[:4, :]
r=8   → A[:, :8]  @ B[:8, :]
r=16  → A[:, :16] @ B[:16, :]



Properties:




Zero cold-start


Instant transitions


Progressive expansion of ΔW space




👉 Rank is not capacity — it is control over ΔW.



Orbital Controller — Control of ΔW Magnitude


Learning rate controls update size:


ΔW ∝ LR × ∇L



Orbital controller:




detects instability (φ)


reduces LR → shrinks ΔW


restores LR → expands ΔW




Behavior:


HIGH → BASE → LOW → BASE → HIGH



👉 prevents uncontrolled weight growth



Weight Control Evidence


Under aggressive training (Qwen 7B, LR=3e-4):


Final Drift:
Baseline: 11.39
Unified:  6.01   (1.9× lower)

Velocity:
Baseline: 0.144
Unified:  0.047  (3× slower)

Norm growth:
Baseline: +3.40
Unified:  +1.02



Trajectory:


Step   Baseline   Unified
0      0.39       0.39
100    8.83       5.75
200    10.65      5.94
300    11.39      6.01



Interpretation:




Baseline → weights drift uncontrollably


Unified → weights remain bounded




Without control: weights diverge
With Unified-LoRA: weights remain bounded




Collapse Recovery (Safety Mechanism)


Under extreme instability:


Baseline: collapse at step 1
Unified:  survives 33 steps



Mechanism:




detects instability


rolls back to stable weights


skips unstable updates




👉 acts as a safety brake for training



Why This Matters




Controls weight dynamics, not just parameters


Prevents collapse under aggressive training


Handles noisy / multi-domain data


Stabilizes training without slowing it down





Results


GLUE (DistilBERT)




Task
Metric
Baseline
Unified
Rank ↓




MRPC
F1
0.882
0.886
42%


SST-2
Acc
0.898
0.885
56%


CoLA
MCC
0.488
0.491
56%


RTE
Acc
0.556
0.592
33%





Noisy Data (key use case)




Noise
Baseline
Unified
Δ




50%
0.42
0.73
+31




👉 stability advantage appears only under stress



Quick Start


from controller import setup_unified_lora

adapters, ctrl = setup_unified_lora(model)

for batch in dataloader:
    loss = model(**batch).loss
    loss.backward()

    ctrl.step(loss=loss.item())

    optimizer.step()
    optimizer.zero_grad()




Structure


nested_lora.py        → ΔW subspace control
orbital_controller.py → ΔW magnitude control
controller.py         → orchestration




Key Insight


Training stability is not a loss problem.
It is a weight dynamics problem.




Limitations




No multi-seed yet


Limited >3B validation


No generation benchmarks





License


Apache 2.0



Contact


Simona Vargiu

Independent Researcher


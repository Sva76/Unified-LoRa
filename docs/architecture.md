Architecture — Nested Orbital LoRA


Core idea: dynamic rank control via stress-driven orbital transitions with weight persistence (no cold start).



Problem: cold start on rank transitions


Standard multi-rank LoRA keeps separate adapters per rank:


r=4, r=8, r=16 → independent weights


Switching rank causes partial cold restarts → performance drop.



Solution: Nested LoRA (one adapter, multiple ranks)


Single adapter at max rank:


A(16, d), B(d, 16)


Active rank is obtained by slicing:




r=4  → A[:4, :],  B[:, :4]


r=8  → A[:8, :],  B[:, :8]


r=16 → full matrix




r4 ⊂ r8 ⊂ r16



Lower ranks reuse trained weights → no cold start.



Scaling


To keep output magnitude consistent:


scale = max_rank / max(r, 1)
scale = min(scale, 4.0)  # optional clamp




Orbital Controller (no thresholds)


Dynamic trajectory instead of static FSM:




Ascend → stress detected → increase rank


Hold → oscillation → stay


Descend → stable → decrease rank




Uses a stack to ensure symmetric return.



Stress signal


φ(t) = |loss - EMA(loss)| + 2.0 × max(0, loss - prev_loss)


Auto-calibrated thresholds:


t_stress = μ + 0.7σ

t_stable = max(μ - 0.3σ, 0)


Robust stats can be used to reduce noise.



Why it matters




avoids cold starts across rank changes


adapts capacity in real-time


works in black-box settings


O(1) overhead





Comparison




Property
Standard LoRA
AdaLoRA
Orbital LoRA




Rank control
Fixed
SVD
Stress


Control type
None
Open
Closed-loop


Transition cost
N/A
High
O(1)


Architecture
Single
Pruned
Nested


Black-box
Yes
No
Yes





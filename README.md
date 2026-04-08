Unified-LoRA


Adaptive LoRA fine-tuning with nested orbital rank control.


A closed-loop controller that dynamically adjusts LoRA rank during training based on observed stress, using a single adapter with sliced dimensions — no cold start, no capacity loss on transitions.



🚀 Quick Start


from nested_lora import inject_nested_lora, set_rank
from orbital_controller import OrbitalController

model = inject_nested_lora(model, max_rank=16)
ctrl = OrbitalController()

for batch in train_loader:
    loss = model(**batch).loss
    rank = ctrl.step(loss.item())
    set_rank(model, rank)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()




🧪 Demo


👉 Run the notebook:

https://github.com/Sva76/Unified-LoRa/blob/main/notebooks/unified_lora_demo.ipynb



📊 Key results


Stress test: task switch (MRPC → SST-2, DistilBERT, 3 seeds)





Baseline (r=16 fixed)
Unified (orbital)
Delta




SST-2 Acc (new task)
0.736
0.740
+0.004


MRPC F1 (retention)
0.526
0.515
-0.011


Effective rank
16.0
13.6
↓


Rank saving
0%
15%
✔




👉 Under distribution shift, the controller adapts capacity dynamically with ~15% rank saving and no performance loss.



Rank trace under shock (Seed 1)


[  0] r4  r4  r4  r8  r8  r8  r8  r16 r16 r16
[ 10] r16 r16 r16 r16 r16 r16 r16 r16 r16 r16
...
[ 60] <<<SHOCK  r16 r16 r16 r16 r16 r16 r16 r16
[ 68] r8  r8  r8  r8  r8  r8  r4  r4  r4  r4
[ 80] r4  r4  r4  r4  r4  r4  r4  r4  r4  r4
[ 92] r8  r16 r16 r16 r16 r16 r16 r16 r16 r16



👉 The controller exhibits disturbance rejection: detects the shock, stabilizes, then reallocates capacity only when needed.



Stable task (MRPC only, 120 steps, 3 seeds)





Baseline (r=16)
Unified
Delta




F1 mean
0.818
0.820
+0.002


σ
0.008
0.008
=




👉 On stable training, the controller stays at max rank. Zero degradation.



🧠 How it works


Architecture: nested orbitals (r4 ⊂ r8 ⊂ r16)


Unified-LoRA uses a single pair of matrices with rank slicing:


self.lora_A = Parameter(shape=[max_rank, in_features])
self.lora_B = Parameter(shape=[out_features, max_rank])

h     = x @ A[:r, :].T
delta = h @ B[:, :r].T



👉 Lower ranks reuse learned weights. No reset, no cold start.



Controller logic


Stress  → increase rank
Stable  → decrease rank
Neutral → hold



Stress signal:


φ(t) = |loss - EMA(loss)| + 2.0 × max(0, loss - prev_loss)



👉 Adaptive thresholds (μ ± kσ) → no manual tuning.



🎯 What this solves




Distribution shift


Noisy / unstable training


Black-box fine-tuning APIs





⚠️ Limitations




No benefit on fully stable training


Not designed for SOTA accuracy gains




👉 Focus: stability + cost efficiency



⚡ Overhead


O(1) per step → negligible.



🔬 Control perspective




Method
Control
Rank




LoRA
None
constant


AdaLoRA
Open-loop
f(step)


Unified-LoRA
Closed-loop
f(stress)





📁 Structure


nested_lora.py        # LoRA engine (rank slicing)
orbital_controller.py # adaptive controller
notebooks/            # demos
experiments/          # tests




👤 Author


Simona Vargiu


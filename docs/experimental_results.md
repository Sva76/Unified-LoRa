Experimental Results


Core result: parity with baseline performance with ~15% rank reduction and dynamic shock response.



1. Stress Test — Task Switch


Setup




Model: DistilBERT-base-uncased + NestedLoRALinear (max_rank=16)


Protocol: MRPC x 60 steps → SST-2 x 60 steps (shock at step 60)


Seeds: 0, 1, 2


Baseline: same architecture, fixed rank=16


Hardware: Colab T4




Results





Baseline (r=16)
Orbital LoRA




SST-2 Accuracy
0.736
0.740


MRPC F1 (retention)
0.526
0.515


Effective rank
16.0
13.6




Parity with ~15% rank saving


Behavior


Post-shock:




detect → descend (r16 → r4)


stabilize


re-ascend (r4 → r16)




Baseline: no reaction (fixed r=16)



2. Stable Task — Parity


Setup




Task: MRPC only (120 steps)


Seeds: 0, 1, 2


Baseline: fixed r=16




Results




Seed
Baseline F1
Orbital F1




0
0.806
0.808


1
0.822
0.826


2
0.824
0.824


Mean
0.818
0.820




No degradation on stable training



3. Rank Dynamics (Black-box — Tinker)


Methods




Method
Control




Standard LoRA
Fixed rank


AdaLoRA-like
Open-loop


Orbital LoRA
Closed-loop




Disturbance response




Method
Reaction
Stability
Recovery




Standard
None
Passive
—


AdaLoRA-like
Indirect
Partial
Limited


Orbital LoRA
Immediate
Stable
Immediate





4. Architecture Insight


Root cause: cold start from separate adapters.


Fix: nested slicing → no cold start → parity restored.



5. Black-box compatibility


Uses only loss signal.

No gradients required.

O(1) overhead.



Next




7B+ validation (ongoing)


LR controller integration




    # Unified-LoRA

**Adaptive LoRA fine-tuning with nested orbital rank control.**

A closed-loop controller that dynamically adjusts LoRA rank during training based on observed stress, using a single adapter with sliced dimensions — no cold start, no capacity loss on transitions.

---

## Key results

### Stress test: task switch (MRPC → SST-2, DistilBERT, 3 seeds)

|                        | Baseline (r=16 fixed) | Unified (orbital) | Delta     |
|------------------------|-----------------------|-------------------|-----------|
| SST-2 Acc (new task)   | 0.736                 | 0.740             | **+0.004** |
| MRPC F1 (retention)    | 0.526                 | 0.515             | -0.011    |
| Effective rank         | 16.0                  | 13.6              |           |
| Rank saving            | 0%                    | **15%**           |           |

Under distribution shift, the controller adapts capacity dynamically with 15% rank saving and no performance loss.

---

### Rank trace under shock (Seed 1)

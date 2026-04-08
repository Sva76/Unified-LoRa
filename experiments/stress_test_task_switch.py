    """
Orbital LoRA — Stress Test: Task Switch

MRPC (60 steps) → SST-2 (60 steps)
Baseline (r=16 fixed) vs Orbital Controller
"""

import time, random, math, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(file))))

from nested_lora import NestedLoRALinear, inject_nested_lora
from orbital_controller import OrbitalController
from controller import set_rank

── CONFIG ──────────────────────────────────────────

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = "distilbert-base-uncased"
BATCH  = 8
LR     = 5e-5
SEEDS  = [0, 1, 2]

MAX_RANK      = 16
WARMUP        = 10
STABLE_WINDOW = 6

STEPS_TASK1   = 60
STEPS_TASK2   = 60
TOTAL_STEPS   = STEPS_TASK1 + STEPS_TASK2

── DATA ────────────────────────────────────────────

print("Loading data...")
tok = AutoTokenizer.from_pretrained(MODEL)

ds_mrpc = load_dataset("glue", "mrpc")
def tok_mrpc(x):
return tok(x["sentence1"], x["sentence2"],
truncation=True, padding="max_length", max_length=128)
ds_mrpc = ds_mrpc.map(tok_mrpc, batched=True)
ds_mrpc.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_mrpc = DataLoader(ds_mrpc["train"], batch_size=BATCH, shuffle=True)
val_mrpc   = DataLoader(ds_mrpc["validation"], batch_size=BATCH)

ds_sst2 = load_dataset("glue", "sst2")
def tok_sst2(x):
return tok(x["sentence"], truncation=True, padding="max_length", max_length=128)
ds_sst2 = ds_sst2.map(tok_sst2, batched=True)
ds_sst2.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_sst2 = DataLoader(ds_sst2["train"], batch_size=BATCH, shuffle=True)
val_sst2   = DataLoader(ds_sst2["validation"], batch_size=BATCH)

metric_mrpc = evaluate.load("glue", "mrpc")
metric_sst2 = evaluate.load("glue", "sst2")

── HELPERS ─────────────────────────────────────────

def make_iter(loader):
while True:
for batch in loader:
yield batch

def get_batch(it):
batch = next(it)
return (batch["input_ids"].to(DEVICE),
batch["attention_mask"].to(DEVICE),
batch["label"].to(DEVICE))

def build_model():
base = AutoModelForSequenceClassification.from_pretrained(
MODEL, num_labels=2, ignore_mismatched_sizes=True
)
return inject_nested_lora(base, MAX_RANK).to(DEVICE)

def eval_f1(model, loader, metric_fn):
model.eval()
preds, labels = [], []
with torch.no_grad():
for batch in loader:
x = batch["input_ids"].to(DEVICE)
m = batch["attention_mask"].to(DEVICE)
y = batch["label"].to(DEVICE)
logits = model(input_ids=x, attention_mask=m).logits
preds.extend(logits.argmax(dim=-1).cpu().numpy())
labels.extend(y.cpu().numpy())
model.train()
result = metric_fn.compute(predictions=preds, references=labels)
return result.get("f1", result.get("accuracy", 0.0))

def eff_rank(usage):
tot = sum(usage.values())
return sum(k * v for k, v in usage.items()) / tot if tot > 0 else 0

── TRAIN BASELINE ──────────────────────────────────

def train_baseline(model):
opt = torch.optim.AdamW(model.parameters(), lr=LR)
set_rank(model, 16)
it_mrpc = make_iter(train_mrpc)
it_sst2 = make_iter(train_sst2)

for step in range(TOTAL_STEPS):
    x, m, y = get_batch(it_mrpc if step < STEPS_TASK1 else it_sst2)

    loss = model(input_ids=x, attention_mask=m, labels=y).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

return model

── TRAIN ORBITAL ───────────────────────────────────

def train_orbital(model):
ctrl = OrbitalController(warmup=WARMUP, stable_window=STABLE_WINDOW)
ctrl.rank = 4
set_rank(model, 4)

opt = torch.optim.AdamW(model.parameters(), lr=LR)
usage = {4: 0, 8: 0, 16: 0}
rank_trace = []
it_mrpc = make_iter(train_mrpc)
it_sst2 = make_iter(train_sst2)

for step in range(TOTAL_STEPS):
    x, m, y = get_batch(it_mrpc if step < STEPS_TASK1 else it_sst2)

    loss = model(input_ids=x, attention_mask=m, labels=y).loss
    loss.backward()

    new_rank = ctrl.step(loss.item())
    new_rank = max(4, min(16, new_rank))
    set_rank(model, new_rank)

    usage[new_rank] += 1
    rank_trace.append(new_rank)

    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()

return model, usage, rank_trace

── RUN ─────────────────────────────────────────────

print(f"\nDevice: {DEVICE}")
print(f"Plan: MRPC × {STEPS_TASK1} → SST-2 × {STEPS_TASK2}")
print(f"Shock at step {STEPS_TASK1}")
print("=" * 55)

results = []

for seed in SEEDS:
print(f"\n{'─' * 55}\n  SEED {seed}\n{'─' * 55}")

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

base_model = build_model()
base_model = train_baseline(base_model)
f1_mrpc_base = eval_f1(base_model, val_mrpc, metric_mrpc)
f1_sst2_base = eval_f1(base_model, val_sst2, metric_sst2)
del base_model; torch.cuda.empty_cache()

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

uni_model = build_model()
uni_model, usage, rank_trace = train_orbital(uni_model)
f1_mrpc_uni = eval_f1(uni_model, val_mrpc, metric_mrpc)
f1_sst2_uni = eval_f1(uni_model, val_sst2, metric_sst2)

er = eff_rank(usage)
saving = 1 - er / 16
transitions = sum(1 for i in range(1, len(rank_trace)) if rank_trace[i] != rank_trace[i-1])

print(f"\n  {'':30s} {'BASELINE':>10s}  {'ORBITAL':>10s}")
print(f"  {'─' * 55}")
print(f"  {'MRPC F1 (retention)':30s} {f1_mrpc_base:10.3f}  {f1_mrpc_uni:10.3f}")
print(f"  {'SST-2 Acc (new task)':30s} {f1_sst2_base:10.3f}  {f1_sst2_uni:10.3f}")
print(f"\n  Orbital: eff_rank={er:.1f}  saving={saving*100:.0f}%  transitions={transitions}")

results.append({
    'f1_mrpc_base': f1_mrpc_base, 'f1_sst2_base': f1_sst2_base,
    'f1_mrpc_uni': f1_mrpc_uni, 'f1_sst2_uni': f1_sst2_uni,
    'eff_rank': er, 'saving': saving
})
del uni_model; torch.cuda.empty_cache()

── SUMMARY ─────────────────────────────────────────

print(f"\n{'=' * 55}\n  SUMMARY\n{'=' * 55}")
mrpc_b = np.mean([r['f1_mrpc_base'] for r in results])
mrpc_u = np.mean([r['f1_mrpc_uni']  for r in results])
sst2_b = np.mean([r['f1_sst2_base'] for r in results])
sst2_u = np.mean([r['f1_sst2_uni']  for r in results])
er_avg = np.mean([r['eff_rank']     for r in results])
sv_avg = np.mean([r['saving']       for r in results])

print(f"\n  {'MRPC F1':20s} {mrpc_b:.3f} → {mrpc_u:.3f}")
print(f"  {'SST-2 Acc':20s} {sst2_b:.3f} → {sst2_u:.3f}")
print(f"  {'Eff rank':20s} 16.0 → {er_avg:.1f}")
print(f"  {'Saving':20s} 0% → {sv_avg*100:.0f}%")

"""
Unified-LoRA — Stable Task Parity Test
========================================

MRPC only, 120 steps, 3 seeds.
Validates that the controller causes zero degradation on stable training.

Usage:
    pip install transformers datasets evaluate
    python stable_task_test.py
"""

import time, random, math, numpy as np, torch, torch.nn as nn
import torch.nn.functional as F, evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from controller import NestedLoRALinear, OrbitalController, inject_nested_lora, set_rank

# ── CONFIG ──────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL  = "distilbert-base-uncased"
BATCH  = 8
STEPS  = 120
LR     = 5e-5
SEEDS  = [0, 1, 2]

MAX_RANK      = 16
WARMUP        = 15
STABLE_WINDOW = 8

# ── DATA ────────────────────────────────────────────
print("Loading data...")
tok = AutoTokenizer.from_pretrained(MODEL)
ds  = load_dataset("glue", "mrpc")

def tok_fn(x):
    return tok(x["sentence1"], x["sentence2"],
               truncation=True, padding="max_length", max_length=128)

ds = ds.map(tok_fn, batched=True)
ds.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])
train_loader = DataLoader(ds["train"], batch_size=BATCH, shuffle=True)
val_loader   = DataLoader(ds["validation"], batch_size=BATCH)
metric = evaluate.load("glue", "mrpc")

# ── HELPERS ─────────────────────────────────────────
def build_model():
    base = AutoModelForSequenceClassification.from_pretrained(
        MODEL, num_labels=2, ignore_mismatched_sizes=True
    )
    return inject_nested_lora(base, MAX_RANK).to(DEVICE)

def eval_model(model):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            x = batch["input_ids"].to(DEVICE)
            m = batch["attention_mask"].to(DEVICE)
            y = batch["label"].to(DEVICE)
            logits = model(input_ids=x, attention_mask=m).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy())
            labels.extend(y.cpu().numpy())
    return metric.compute(predictions=preds, references=labels)["f1"]

def eff_rank(usage):
    tot = sum(usage.values())
    return sum(k * v for k, v in usage.items()) / tot if tot > 0 else 0

# ── TRAIN BASELINE ──────────────────────────────────
def train_baseline(model):
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    set_rank(model, 16)
    it = iter(train_loader)

    for step in range(STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)

        x = batch["input_ids"].to(DEVICE)
        m = batch["attention_mask"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        loss = model(input_ids=x, attention_mask=m, labels=y).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

    return model

# ── TRAIN UNIFIED ───────────────────────────────────
def train_unified(model):
    ctrl = OrbitalController(warmup=WARMUP, stable_window=STABLE_WINDOW)
    opt  = torch.optim.AdamW(model.parameters(), lr=LR)
    usage = {4: 0, 8: 0, 16: 0}
    rank_trace = []
    it = iter(train_loader)

    for step in range(STEPS):
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)

        x = batch["input_ids"].to(DEVICE)
        m = batch["attention_mask"].to(DEVICE)
        y = batch["label"].to(DEVICE)

        loss = model(input_ids=x, attention_mask=m, labels=y).loss
        new_rank = ctrl.step(loss.item())
        set_rank(model, new_rank)

        usage[new_rank] += 1
        rank_trace.append(new_rank)

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

    return model, usage, rank_trace, ctrl

# ── RUN ─────────────────────────────────────────────
print(f"\nDevice: {DEVICE}")
print(f"Task: MRPC, {STEPS} steps")
print("=" * 55)

results = []

for seed in SEEDS:
    print(f"\n{'─' * 50}\n  SEED {seed}\n{'─' * 50}")

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    base_model = build_model()
    base_model = train_baseline(base_model)
    f1_base = eval_model(base_model)
    del base_model; torch.cuda.empty_cache()

    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    uni_model = build_model()
    uni_model, usage, trace, ctrl = train_unified(uni_model)
    f1_uni = eval_model(uni_model)

    er = eff_rank(usage)
    saving = 1 - er / 16
    transitions = sum(1 for i in range(1, len(trace)) if trace[i] != trace[i-1])

    print(f"\n  BASELINE   F1 = {f1_base:.3f}   (rank=16 fixed)")
    print(f"  UNIFIED    F1 = {f1_uni:.3f}   (eff_rank={er:.1f}, saving={saving*100:.0f}%)")
    print(f"  delta F1 = {f1_uni - f1_base:+.3f}")
    print(f"  Usage: r4={usage[4]}  r8={usage[8]}  r16={usage[16]}  transitions={transitions}")

    results.append({
        'seed': seed, 'f1_base': f1_base, 'f1_uni': f1_uni,
        'delta': f1_uni - f1_base, 'eff_rank': er,
    })
    del uni_model; torch.cuda.empty_cache()

# ── SUMMARY ─────────────────────────────────────────
print(f"\n{'=' * 55}\n  SUMMARY\n{'=' * 55}")
f1b = [r['f1_base'] for r in results]
f1u = [r['f1_uni']  for r in results]
print(f"\n  Baseline F1:  {np.mean(f1b):.3f} +/- {np.std(f1b):.3f}")
print(f"  Unified  F1:  {np.mean(f1u):.3f} +/- {np.std(f1u):.3f}")
print(f"  delta F1:     {np.mean([r['delta'] for r in results]):+.3f}")

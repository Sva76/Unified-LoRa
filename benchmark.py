"""
Unified-LoRA Benchmark
Adaptive per-layer rank controller for LoRA fine-tuning.

Runs 4 GLUE tasks (MRPC, SST-2, CoLA, RTE) comparing:
  - Baseline: fixed rank=16
  - Adaptive: per-layer gradient-stress rank controller

Requirements:
  pip install transformers datasets evaluate accelerate scikit-learn

Hardware: GPU recommended (tested on T4, ~30 min total)
"""

import copy, torch, time, gc
import torch.nn as nn
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
import evaluate

# ================================================================
# CONFIG
# ================================================================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "distilbert-base-uncased"

BATCH_SIZE = 16
EPOCHS = 3
LR = 5e-4
MAX_RANK = 16
MIN_RANK = 4
ALPHA = 16
GRAD_CLIP = 1.0

TASKS = {
    "mrpc": {"num_labels": 2, "metric_key": "f1",
             "paired": True, "keys": ("sentence1", "sentence2")},
    "sst2": {"num_labels": 2, "metric_key": "accuracy",
             "paired": False, "keys": ("sentence",)},
    "cola": {"num_labels": 2, "metric_key": "matthews_correlation",
             "paired": False, "keys": ("sentence",)},
    "rte":  {"num_labels": 2, "metric_key": "accuracy",
             "paired": True, "keys": ("sentence1", "sentence2")},
}

# ================================================================
# DATA
# ================================================================
def load_task(task_name):
    cfg = TASKS[task_name]
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    ds = load_dataset("glue", task_name)

    if cfg["paired"]:
        def preprocess(x):
            return tokenizer(x[cfg["keys"][0]], x[cfg["keys"][1]], truncation=True)
    else:
        def preprocess(x):
            return tokenizer(x[cfg["keys"][0]], truncation=True)

    ds = ds.map(preprocess, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer)
    train_loader = DataLoader(
        ds["train"], batch_size=BATCH_SIZE, shuffle=True, collate_fn=collator
    )
    val_loader = DataLoader(
        ds["validation"], batch_size=32, collate_fn=collator
    )
    metric = evaluate.load("glue", task_name)

    return train_loader, val_loader, metric, cfg

# ================================================================
# LoRA MODULE — per-layer adaptive rank
# ================================================================
class LoRALinear(nn.Module):
    """
    LoRA adapter with:
    - Per-layer gradient stress tracking (EMA)
    - Dynamic rank adjustment based on stress trend
    - Standard alpha/r scaling
    """

    def __init__(self, base, max_r=16, layer_name=""):
        super().__init__()
        self.base = copy.deepcopy(base)
        for p in self.base.parameters():
            p.requires_grad = False

        self.max_r = max_r
        self.layer_name = layer_name
        self.A = nn.Parameter(torch.randn(max_r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, max_r))
        self.active_r = MIN_RANK

        # Stress tracking
        self.grad_ema = None
        self.prev_grad_ema = None

    def set_rank(self, r):
        self.active_r = max(MIN_RANK, min(r, self.max_r))

    def update_rank(self):
        """Adapt rank based on gradient stress trend."""
        if self.A.grad is None:
            return

        grad_norm = self.A.grad[:self.active_r].norm().item()

        if self.grad_ema is None:
            self.grad_ema = grad_norm
            self.prev_grad_ema = grad_norm
            return

        self.prev_grad_ema = self.grad_ema
        self.grad_ema = 0.9 * self.grad_ema + 0.1 * grad_norm

        delta = self.grad_ema - self.prev_grad_ema
        threshold = 0.01 * self.grad_ema if self.grad_ema > 0 else 0.01

        if delta > threshold:      # stress increasing -> more capacity
            self.active_r = min(self.max_r, self.active_r + 2)
        elif delta < -threshold:   # stress decreasing -> reduce
            self.active_r = max(MIN_RANK, self.active_r - 2)

    def forward(self, x):
        base_out = self.base(x)
        A = self.A[:self.active_r]
        B = self.B[:, :self.active_r]
        lora_out = x @ A.t() @ B.t()
        scale = ALPHA / self.active_r
        return base_out + scale * lora_out

# ================================================================
# HELPERS
# ================================================================
def inject_lora(model):
    for i, layer in enumerate(model.distilbert.transformer.layer):
        layer.attention.q_lin = LoRALinear(
            layer.attention.q_lin, MAX_RANK, layer_name=f"layer{i}.q"
        )
        layer.attention.v_lin = LoRALinear(
            layer.attention.v_lin, MAX_RANK, layer_name=f"layer{i}.v"
        )
    return model


def get_lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]


def setup_trainable(model):
    for p in model.parameters():
        p.requires_grad = False
    for m in get_lora_modules(model):
        m.A.requires_grad = True
        m.B.requires_grad = True
    for n, p in model.named_parameters():
        if "classifier" in n or "pre_classifier" in n:
            p.requires_grad = True
    return model


def evaluate_model(model, val_loader, metric):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.argmax(logits, dim=1)
            preds += p.cpu().tolist()
            labels += batch["labels"].cpu().tolist()
    return metric.compute(predictions=preds, references=labels)

# ================================================================
# TRAINING
# ================================================================
def train(task_name, adaptive=True):
    train_loader, val_loader, metric, cfg = load_task(task_name)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=cfg["num_labels"]
    )
    model = inject_lora(model)

    if not adaptive:
        for m in get_lora_modules(model):
            m.set_rank(MAX_RANK)

    model = setup_trainable(model).to(DEVICE)

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    rank_history = {m.layer_name: [] for m in get_lora_modules(model)}

    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            loss = model(**batch).loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

            if adaptive:
                for m in get_lora_modules(model):
                    m.update_rank()
                    rank_history[m.layer_name].append(m.active_r)

            opt.step()
            opt.zero_grad()

    elapsed = time.time() - t0
    res = evaluate_model(model, val_loader, metric)

    # Stats
    all_ranks = []
    layer_avg = {}
    for name, ranks in rank_history.items():
        if ranks:
            layer_avg[name] = sum(ranks) / len(ranks)
            all_ranks.extend(ranks)

    global_avg_rank = sum(all_ranks) / len(all_ranks) if all_ranks else MAX_RANK

    if adaptive:
        print(f"\n  Per-layer rank ({task_name}):")
        for name in sorted(layer_avg.keys()):
            print(f"    {name}: {layer_avg[name]:.1f}")

    del model, opt
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {**res, "avg_rank": global_avg_rank, "time": elapsed}

# ================================================================
# RUN
# ================================================================
def main():
    results = {}

    for task_name in TASKS:
        print(f"\n{'='*50}")
        print(f" {task_name.upper()}")
        print(f"{'='*50}")

        results[task_name] = {}

        print(f"\n  Baseline (fixed rank=16)...")
        results[task_name]["baseline"] = train(task_name, adaptive=False)

        print(f"\n  Adaptive (per-layer controller)...")
        results[task_name]["adaptive"] = train(task_name, adaptive=True)

    # Results table
    print("\n" + "=" * 65)
    print(" RESULTS")
    print("=" * 65)

    print(f"\n{'Task':<8} {'Method':<12} {'Metric':>10} {'Avg Rank':>10} {'Time':>8}")
    print("-" * 50)

    for task_name in TASKS:
        metric_key = TASKS[task_name]["metric_key"]

        for method in ["baseline", "adaptive"]:
            r = results[task_name][method]
            val = r.get(metric_key, r.get("accuracy", -1))
            rank = r.get("avg_rank", -1)
            t = r.get("time", -1)
            print(f"{task_name:<8} {method:<12} {val:>10.4f} {rank:>10.1f} {t:>7.1f}s")
        print()

    # Summary
    print("=" * 65)
    print(" SUMMARY")
    print("=" * 65)

    for task_name in TASKS:
        metric_key = TASKS[task_name]["metric_key"]
        b = results[task_name]["baseline"]
        a = results[task_name]["adaptive"]

        b_val = b.get(metric_key, b.get("accuracy", 0))
        a_val = a.get(metric_key, a.get("accuracy", 0))
        a_rank = a.get("avg_rank", 16)

        rank_red = 100 * (1 - a_rank / 16)

        print(f"  {task_name:<8} delta: {a_val - b_val:+.4f}  rank: {a_rank:.1f}/16  reduction: {rank_red:.0f}%")


if __name__ == "__main__":
    main()

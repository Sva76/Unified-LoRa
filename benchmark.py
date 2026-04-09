"""
Unified-LoRA GLUE Benchmark
============================
Runs adaptive vs fixed-rank LoRA on GLUE tasks with DistilBERT.

Usage:
    python benchmark.py                      # All tasks
    python benchmark.py --tasks mrpc cola     # Specific tasks
    python benchmark.py --noise 0.3           # With 30% label noise

Author: Simona Vargiu
License: Apache 2.0
"""

import argparse
import random
import torch
import numpy as np
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from datasets import load_dataset
import evaluate

from nested_lora import inject_nested_lora, get_lora_params, count_params
from orbital_controller import OrbitalController


# ── Config ──────────────────────────────────────────────────────

TASK_CONFIG = {
    "mrpc":  {"metric": "f1",       "num_labels": 2},
    "sst2":  {"metric": "accuracy", "num_labels": 2},
    "cola":  {"metric": "matthews_correlation", "num_labels": 2},
    "rte":   {"metric": "accuracy", "num_labels": 2},
}

MODEL_NAME = "distilbert-base-uncased"
EPOCHS = 3
LR = 5e-4
BATCH_SIZE = 32
MAX_RANK = 16
ALPHA = 16.0
RANK_LEVELS = [4, 8, 16]
SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def inject_noise(dataset, noise_rate: float, num_labels: int):
    """Flip labels randomly at the given rate."""
    if noise_rate <= 0:
        return dataset

    def flip(example):
        if random.random() < noise_rate:
            example["label"] = random.randint(0, num_labels - 1)
        return example

    return dataset.map(flip)


def run_task(task: str, adaptive: bool, noise_rate: float = 0.0):
    """Run a single GLUE task. Returns (metric_value, avg_rank)."""
    set_seed(SEED)
    config = TASK_CONFIG[task]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model + tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=config["num_labels"]
    ).to(device)

    # Freeze base model
    for p in model.parameters():
        p.requires_grad = False

    # Inject NestedLoRA
    adapters = inject_nested_lora(
        model,
        target_modules=["q_lin", "v_lin"],  # DistilBERT naming
        max_rank=MAX_RANK,
        alpha=ALPHA,
    )

    controller = None
    if adaptive:
        controller = OrbitalController(
            adapters,
            rank_levels=RANK_LEVELS,
            ema_alpha=0.1,
            threshold_k=1.5,
            eval_interval=10,
            warmup_steps=50,
        )
    else:
        # Fixed rank at MAX_RANK
        for adapter in adapters.values():
            adapter.active_rank = MAX_RANK

    # Load data
    glue_key = "sst2" if task == "sst2" else task
    dataset = load_dataset("glue", glue_key)

    if task in ("mrpc", "rte"):
        text_keys = ("sentence1", "sentence2")
    elif task == "sst2":
        text_keys = ("sentence",)
    elif task == "cola":
        text_keys = ("sentence",)
    else:
        text_keys = ("sentence1", "sentence2")

    def tokenize(batch):
        if len(text_keys) == 2:
            return tokenizer(batch[text_keys[0]], batch[text_keys[1]],
                             truncation=True, padding="max_length", max_length=128)
        return tokenizer(batch[text_keys[0]],
                         truncation=True, padding="max_length", max_length=128)

    train_ds = dataset["train"].map(tokenize, batched=True)
    train_ds = inject_noise(train_ds, noise_rate, config["num_labels"])
    train_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    val_split = "validation_matched" if task == "mnli" else "validation"
    val_ds = dataset[val_split].map(tokenize, batched=True)
    val_ds.set_format("torch", columns=["input_ids", "attention_mask", "label"])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(get_lora_params(adapters), lr=LR)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, 0, total_steps)

    # Train
    model.train()
    for epoch in range(EPOCHS):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                labels=batch["label"],
            )
            outputs.loss.backward()

            if controller is not None:
                controller.step()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    # Evaluate
    metric = evaluate.load("glue", glue_key)
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = outputs.logits.argmax(dim=-1)
            metric.add_batch(predictions=preds, references=batch["label"])

    results = metric.compute()
    metric_val = results[config["metric"]]
    avg_rank = controller.avg_rank() if controller else MAX_RANK

    return metric_val, avg_rank


def main():
    parser = argparse.ArgumentParser(description="Unified-LoRA GLUE Benchmark")
    parser.add_argument("--tasks", nargs="+", default=list(TASK_CONFIG.keys()))
    parser.add_argument("--noise", type=float, default=0.0,
                        help="Label noise rate (0.0–1.0)")
    args = parser.parse_args()

    print(f"{'Task':<8} {'Mode':<10} {'Metric':>8} {'Avg Rank':>10}")
    print("-" * 40)

    for task in args.tasks:
        if task not in TASK_CONFIG:
            print(f"Unknown task: {task}")
            continue

        # Baseline
        baseline_val, _ = run_task(task, adaptive=False, noise_rate=args.noise)
        print(f"{task:<8} {'fixed':10} {baseline_val:8.3f} {MAX_RANK:10.1f}")

        # Adaptive
        adaptive_val, avg_rank = run_task(task, adaptive=True, noise_rate=args.noise)
        print(f"{task:<8} {'adaptive':10} {adaptive_val:8.3f} {avg_rank:10.1f}")

        reduction = (1 - avg_rank / MAX_RANK) * 100
        delta = adaptive_val - baseline_val
        print(f"  → Δ={delta:+.3f}, rank reduction={reduction:.0f}%\n")


if __name__ == "__main__":
    main()

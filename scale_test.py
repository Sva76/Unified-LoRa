"""
SCALE TEST: Does rank matter on larger models?
===============================================
Qwen2.5-3B in 4-bit, MRPC, 3 seeds
r=8 vs r=16 vs r=32 vs Adaptive

Colab Pro: select A100 in Runtime → Change runtime type
Estimated time: ~30-45 min on A100
"""

!pip install -q transformers datasets evaluate accelerate scikit-learn bitsandbytes

import copy, torch, time, gc
import torch.nn as nn
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
    DataCollatorWithPadding,
)
from torch.utils.data import DataLoader
import evaluate

DEVICE = "cuda"
MODEL_NAME = "Qwen/Qwen2.5-3B"

BATCH_SIZE = 4
EPOCHS = 2
LR = 1e-4
MAX_RANK = 32
MIN_RANK = 4
ALPHA = 16
GRAD_CLIP = 0.5
MAX_LENGTH = 128

SEEDS = [0, 1, 2]

# ================================================================
# SEED
# ================================================================
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ================================================================
# DATA
# ================================================================
def load_data():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ds = load_dataset("glue", "mrpc")

    def preprocess(batch):
        return tokenizer(
            batch["sentence1"], batch["sentence2"],
            truncation=True, padding="max_length", max_length=MAX_LENGTH,
        )

    ds = ds.map(preprocess, batched=True)
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_loader = DataLoader(
        ds["train"], batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=collator, generator=torch.Generator().manual_seed(0),
    )
    val_loader = DataLoader(
        ds["validation"], batch_size=8, collate_fn=collator,
    )
    metric = evaluate.load("glue", "mrpc")

    return train_loader, val_loader, metric, tokenizer

# ================================================================
# LoRA MODULE
# ================================================================
class LoRALinear(nn.Module):
    def __init__(self, base, max_r=32, layer_name=""):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.max_r = max_r
        self.layer_name = layer_name
        self.A = nn.Parameter(torch.randn(max_r, base.in_features, dtype=torch.float32) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, max_r, dtype=torch.float32))
        self.active_r = MIN_RANK

        self.grad_ema = None
        self.prev_grad_ema = None

    def set_rank(self, r):
        self.active_r = max(MIN_RANK, min(r, self.max_r))

    def update_rank(self):
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

        if delta > threshold:
            self.active_r = min(self.max_r, self.active_r + 2)
        elif delta < -threshold:
            self.active_r = max(MIN_RANK, self.active_r - 2)

    def forward(self, x):
        base_out = self.base(x)
        A = self.A[:self.active_r].to(device=x.device, dtype=x.dtype)
        B = self.B[:, :self.active_r].to(device=x.device, dtype=x.dtype)
        lora_out = x @ A.t() @ B.t()
        lora_out = torch.clamp(lora_out, -5, 5)
        scale = ALPHA / self.active_r
        return base_out + scale * lora_out

# ================================================================
# INJECT
# ================================================================
def inject_lora(model, fixed_rank=None):
    replace_names = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and (
            name.endswith("q_proj") or name.endswith("v_proj")
        ):
            replace_names.append(name)

    for name in replace_names:
        parts = name.split(".")
        parent = model
        for p in parts[:-1]:
            parent = getattr(parent, p)
        original = getattr(parent, parts[-1])
        lora = LoRALinear(original, MAX_RANK, layer_name=name)
        if fixed_rank is not None:
            lora.set_rank(fixed_rank)
        setattr(parent, parts[-1], lora)

    print(f"  Injected LoRA into {len(replace_names)} layers")
    return model

def get_lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]

# ================================================================
# TRAIN
# ================================================================
def train(mode="r16", seed=0):
    set_seed(seed)
    train_loader, val_loader, metric, tokenizer = load_data()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    if mode == "adaptive":
        model = inject_lora(model, fixed_rank=None)
    else:
        rank = int(mode.replace("r", ""))
        model = inject_lora(model, fixed_rank=rank)

    for p in model.parameters():
        p.requires_grad = False

    for m in get_lora_modules(model):
        m.A.requires_grad = True
        m.B.requires_grad = True

    for n, p in model.named_parameters():
        if "score" in n or "classifier" in n:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable params: {trainable:,}")

    opt = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LR
    )

    rank_history = []
    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss

            if torch.isnan(loss) or torch.isinf(loss):
                opt.zero_grad()
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )

            if mode == "adaptive":
                for m in get_lora_modules(model):
                    m.update_rank()
                    rank_history.append(m.active_r)

            opt.step()
            opt.zero_grad()

            if step % 50 == 0:
                r_str = f" rank={np.mean([m.active_r for m in get_lora_modules(model)]):.1f}" if mode == "adaptive" else ""
                print(f"    e={epoch} s={step} loss={loss.item():.4f}{r_str}")

    elapsed = time.time() - t0

    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.argmax(logits, dim=1)
            preds += p.cpu().tolist()
            labels += batch["labels"].cpu().tolist()

    res = metric.compute(predictions=preds, references=labels)
    avg_rank = np.mean(rank_history) if rank_history else int(mode.replace("r", "")) if mode != "adaptive" else MIN_RANK

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    return {**res, "avg_rank": float(avg_rank), "time": elapsed, "mode": mode, "seed": seed}

# ================================================================
# RUN
# ================================================================
print("=" * 60)
print(f" SCALE TEST: {MODEL_NAME}")
print(f" Does rank matter at this scale?")
print("=" * 60)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODES = ["r8", "r16", "r32", "adaptive"]
results = {m: [] for m in MODES}

for seed in SEEDS:
    for mode in MODES:
        label = f"{mode}/seed={seed}"
        print(f"\n  {label}...")

        try:
            res = train(mode=mode, seed=seed)
            results[mode].append(res)
            print(f"  → acc={res['accuracy']:.4f} f1={res['f1']:.4f} rank={res['avg_rank']:.1f} ({res['time']:.0f}s)")
        except Exception as e:
            print(f"  → FAILED: {e}")
            import traceback
            traceback.print_exc()

# ================================================================
# RESULTS
# ================================================================
print("\n" + "=" * 60)
print(" RESULTS (mean ± std)")
print("=" * 60)

print(f"\n{'Mode':<12} {'Acc':>12} {'F1':>12} {'Acc Std':>10} {'F1 Std':>10} {'Rank':>8}")
print("-" * 56)

for mode in MODES:
    if not results[mode]:
        print(f"{mode:<12} {'FAILED':>12}")
        continue

    accs = [r["accuracy"] for r in results[mode]]
    f1s = [r["f1"] for r in results[mode]]
    ranks = [r["avg_rank"] for r in results[mode]]

    print(f"{mode:<12} {np.mean(accs):>12.4f} {np.mean(f1s):>12.4f} {np.std(accs):>10.4f} {np.std(f1s):>10.4f} {np.mean(ranks):>8.1f}")

# ================================================================
# KEY QUESTION
# ================================================================
print("\n" + "=" * 60)
print(" KEY QUESTION: Does rank matter at this scale?")
print("=" * 60)

if results["r8"] and results["r32"]:
    r8_f1 = np.mean([r["f1"] for r in results["r8"]])
    r32_f1 = np.mean([r["f1"] for r in results["r32"]])
    gap = abs(r32_f1 - r8_f1)

    print(f"\n  r=8  F1: {r8_f1:.4f}")
    print(f"  r=32 F1: {r32_f1:.4f}")
    print(f"  Gap: {gap:.4f}")

    if gap > 0.02:
        print(f"\n  → YES. Rank matters ({gap:.1%} gap). The adaptive controller has a real problem to solve.")
    elif gap > 0.01:
        print(f"\n  → MAYBE. Small gap ({gap:.1%}). Marginal benefit possible.")
    else:
        print(f"\n  → NO. Rank doesn't matter at this scale either ({gap:.1%} gap).")

if results["adaptive"] and results["r16"]:
    a_f1 = np.mean([r["f1"] for r in results["adaptive"]])
    a_std = np.std([r["f1"] for r in results["adaptive"]])
    r16_f1 = np.mean([r["f1"] for r in results["r16"]])
    a_rank = np.mean([r["avg_rank"] for r in results["adaptive"]])

    print(f"\n  Adaptive F1: {a_f1:.4f} ± {a_std:.4f} (rank={a_rank:.1f})")
    print(f"  r=16    F1: {r16_f1:.4f}")
    print(f"  Delta: {a_f1 - r16_f1:+.4f}")

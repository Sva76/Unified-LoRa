"""
STABILITY TEST v2: FSM φ(t) with LoRA
======================================
Qwen2.5-3B in 4-bit + LoRA (r=16), MRPC, 3 seeds, A100

3-way comparison (all use LoRA r=16 — this is a stability test, not rank test):
  1. Baseline — fixed LR, no protection
  2. Cosine LR — cosine annealing scheduler
  3. FSM φ(t) — mode switching + adaptive LR

Previous test failed because only 4K params were trainable (no LoRA).
This version uses LoRA with 7M+ params where real instability occurs.
"""

!pip install -q transformers datasets evaluate accelerate scikit-learn bitsandbytes

import copy, torch, time, gc, math
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
MAX_RANK = 16
MIN_RANK = 4
ALPHA = 16
GRAD_CLIP = 0.5
MAX_LENGTH = 128

SEEDS = [0, 1, 2]

# ================================================================
# FSM φ(t) CONTROLLER
# ================================================================
class FSMController:
    def __init__(self, lr_base, alpha=0.1, beta=0.9, theta0=0.3, theta1=0.7):
        self.lr_base = lr_base
        self.alpha = alpha
        self.beta = beta
        self.theta0 = theta0
        self.theta1 = theta1

        self.phi = 0.0
        self.E_smooth = 0.0
        self.mode = 0
        self.step_count = 0

        self.mode_history = []
        self.phi_history = []
        self.shock_events = 0

    def update(self, loss_value):
        self.step_count += 1

        self.E_smooth = self.beta * self.E_smooth + (1 - self.beta) * loss_value
        D = self.E_smooth / (1 + self.E_smooth)

        old_phi = self.phi
        self.phi = (1 - self.alpha) * self.phi + self.alpha * D

        if self.phi - old_phi > 0.1:
            self.shock_events += 1

        if self.phi < self.theta0:
            self.mode = 0
        elif self.phi < self.theta1:
            self.mode = 1
        else:
            self.mode = 2

        if self.mode == 0:
            lr = self.lr_base
        elif self.mode == 1:
            lr = self.lr_base * 0.6
        else:
            lr = self.lr_base * 0.2

        self.mode_history.append(self.mode)
        self.phi_history.append(self.phi)

        return lr

    def get_stats(self):
        mode_counts = [0, 0, 0]
        for m in self.mode_history:
            mode_counts[m] += 1
        total = len(self.mode_history) if self.mode_history else 1
        return {
            "mode_0_pct": mode_counts[0] / total * 100,
            "mode_1_pct": mode_counts[1] / total * 100,
            "mode_2_pct": mode_counts[2] / total * 100,
            "shock_events": self.shock_events,
            "final_phi": self.phi,
            "final_mode": self.mode,
        }

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
# LoRA MODULE (fixed rank — this is a stability test)
# ================================================================
class LoRALinear(nn.Module):
    def __init__(self, base, rank=16, layer_name=""):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.rank = rank
        self.layer_name = layer_name
        self.A = nn.Parameter(torch.randn(rank, base.in_features, dtype=torch.float32) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32))

    def forward(self, x):
        base_out = self.base(x)
        A = self.A.to(device=x.device, dtype=x.dtype)
        B = self.B.to(device=x.device, dtype=x.dtype)
        lora_out = x @ A.t() @ B.t()
        lora_out = torch.clamp(lora_out, -5, 5)
        scale = ALPHA / self.rank
        return base_out + scale * lora_out

# ================================================================
# INJECT + HELPERS
# ================================================================
def inject_lora(model):
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
        setattr(parent, parts[-1], LoRALinear(original, MAX_RANK, layer_name=name))

    print(f"  Injected LoRA (r={MAX_RANK}) into {len(replace_names)} layers")
    return model

def get_lora_modules(model):
    return [m for m in model.modules() if isinstance(m, LoRALinear)]

# ================================================================
# TRAIN
# ================================================================
def train(mode="baseline", seed=0):
    set_seed(seed)
    train_loader, val_loader, metric, tokenizer = load_data()

    total_steps = len(train_loader) * EPOCHS

    # Load 4-bit model
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

    # Inject LoRA
    model = inject_lora(model)

    # Freeze base, unfreeze LoRA + classifier
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

    # LR strategy
    fsm = None
    scheduler = None

    if mode == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=total_steps)
    elif mode == "fsm":
        fsm = FSMController(lr_base=LR)

    # Tracking
    loss_trajectory = []
    lr_trajectory = []
    nan_count = 0

    t0 = time.time()

    for epoch in range(EPOCHS):
        model.train()
        for step, batch in enumerate(train_loader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            out = model(**batch)
            loss = out.loss

            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                opt.zero_grad()
                continue

            loss_val = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], GRAD_CLIP
            )

            # FSM LR control
            if mode == "fsm":
                new_lr = fsm.update(loss_val)
                for g in opt.param_groups:
                    g['lr'] = new_lr

            opt.step()
            opt.zero_grad()

            # Cosine step
            if mode == "cosine":
                scheduler.step()

            # Track
            current_lr = opt.param_groups[0]['lr']
            loss_trajectory.append(loss_val)
            lr_trajectory.append(current_lr)

            if step % 50 == 0:
                extra = ""
                if mode == "fsm":
                    extra = f" phi={fsm.phi:.3f} mode={fsm.mode}"
                print(f"    e={epoch} s={step} loss={loss_val:.4f} lr={current_lr:.6f}{extra}")

    elapsed = time.time() - t0

    # Eval
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

    # Loss stability metrics
    loss_arr = np.array(loss_trajectory) if loss_trajectory else np.array([0])
    loss_std = float(np.std(loss_arr))
    loss_max = float(np.max(loss_arr))

    # Count spikes (loss > 2x moving average)
    window = 20
    spike_count = 0
    for i in range(window, len(loss_arr)):
        ma = np.mean(loss_arr[max(0, i-window):i])
        if loss_arr[i] > 2 * ma and ma > 0.01:
            spike_count += 1

    # FSM stats
    fsm_stats = fsm.get_stats() if fsm else {}

    del model, opt
    gc.collect()
    torch.cuda.empty_cache()

    return {
        **res,
        "time": elapsed,
        "mode": mode,
        "seed": seed,
        "nan_count": nan_count,
        "loss_std": loss_std,
        "loss_max": loss_max,
        "spike_count": spike_count,
        **fsm_stats,
    }

# ================================================================
# RUN
# ================================================================
print("=" * 60)
print(f" STABILITY TEST v2: {MODEL_NAME} + LoRA")
print(f" Baseline vs Cosine vs FSM phi(t)")
print("=" * 60)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

MODES = ["baseline", "cosine", "fsm"]
results = {m: [] for m in MODES}

for seed in SEEDS:
    for mode in MODES:
        label = f"{mode}/seed={seed}"
        print(f"\n  {label}...")

        try:
            res = train(mode=mode, seed=seed)
            results[mode].append(res)
            extra = ""
            if mode == "fsm":
                extra = f" shocks={res.get('shock_events', 0)} mode2={res.get('mode_2_pct', 0):.0f}%"
            print(f"  -> acc={res['accuracy']:.4f} f1={res['f1']:.4f} spikes={res['spike_count']} nans={res['nan_count']}{extra}")
        except Exception as e:
            print(f"  -> FAILED: {e}")
            import traceback
            traceback.print_exc()

# ================================================================
# RESULTS
# ================================================================
print("\n" + "=" * 70)
print(" RESULTS (mean +/- std over 3 seeds)")
print("=" * 70)

print(f"\n{'Mode':<12} {'Acc':>10} {'F1':>10} {'F1 Std':>8} {'Loss Std':>10} {'Spikes':>8} {'NaNs':>6}")
print("-" * 66)

for mode in MODES:
    if not results[mode]:
        print(f"{mode:<12} FAILED")
        continue

    accs = [r["accuracy"] for r in results[mode]]
    f1s = [r["f1"] for r in results[mode]]
    loss_stds = [r["loss_std"] for r in results[mode]]
    spikes = [r["spike_count"] for r in results[mode]]
    nans = [r["nan_count"] for r in results[mode]]

    print(f"{mode:<12} {np.mean(accs):>10.4f} {np.mean(f1s):>10.4f} {np.std(f1s):>8.4f} {np.mean(loss_stds):>10.4f} {np.mean(spikes):>8.1f} {np.mean(nans):>6.1f}")

# FSM details
if results["fsm"]:
    print(f"\n  FSM details:")
    for r in results["fsm"]:
        print(f"    seed={r['seed']}: phi={r.get('final_phi', 0):.3f} "
              f"mode0={r.get('mode_0_pct', 0):.0f}% "
              f"mode1={r.get('mode_1_pct', 0):.0f}% "
              f"mode2={r.get('mode_2_pct', 0):.0f}% "
              f"shocks={r.get('shock_events', 0)}")

# ================================================================
# KEY QUESTIONS
# ================================================================
print("\n" + "=" * 70)
print(" KEY QUESTIONS")
print("=" * 70)

if results["baseline"] and results["fsm"]:
    b_f1 = np.mean([r["f1"] for r in results["baseline"]])
    f_f1 = np.mean([r["f1"] for r in results["fsm"]])
    b_std = np.std([r["f1"] for r in results["baseline"]])
    f_std = np.std([r["f1"] for r in results["fsm"]])
    b_spikes = np.mean([r["spike_count"] for r in results["baseline"]])
    f_spikes = np.mean([r["spike_count"] for r in results["fsm"]])
    b_loss_std = np.mean([r["loss_std"] for r in results["baseline"]])
    f_loss_std = np.mean([r["loss_std"] for r in results["fsm"]])

    print(f"\n  1. Does FSM improve performance over no protection?")
    print(f"     Baseline: {b_f1:.4f} +/- {b_std:.4f}")
    print(f"     FSM:      {f_f1:.4f} +/- {f_std:.4f}")
    print(f"     Delta:    {f_f1 - b_f1:+.4f}")

    print(f"\n  2. Does FSM reduce instability?")
    print(f"     Baseline spikes: {b_spikes:.1f}, loss_std: {b_loss_std:.4f}")
    print(f"     FSM spikes:      {f_spikes:.1f}, loss_std: {f_loss_std:.4f}")

    print(f"\n  3. Does FSM reduce result variance?")
    print(f"     Baseline F1 std: {b_std:.4f}")
    print(f"     FSM F1 std:      {f_std:.4f}")

if results["cosine"] and results["fsm"]:
    c_f1 = np.mean([r["f1"] for r in results["cosine"]])
    c_std = np.std([r["f1"] for r in results["cosine"]])
    c_spikes = np.mean([r["spike_count"] for r in results["cosine"]])

    print(f"\n  4. Does FSM beat cosine scheduler?")
    print(f"     Cosine: {c_f1:.4f} +/- {c_std:.4f} (spikes={c_spikes:.1f})")
    print(f"     FSM:    {f_f1:.4f} +/- {f_std:.4f} (spikes={f_spikes:.1f})")
    print(f"     Delta:  {f_f1 - c_f1:+.4f}")

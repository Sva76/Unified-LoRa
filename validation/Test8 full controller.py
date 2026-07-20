"""
=============================================================================
TEST DEL CONTROLLER UNIFIED-LORA COMPLETO su LLM reale (Colab GPU)
=============================================================================
DIFFERENZA CHIAVE dai test Tinker:
  - Tinker testava φ in forma "high-level" (EMA della loss), perché l'API
    non espone i gradienti per-layer.
  - QUI gira il CONTROLLER VERO del repo: OrbitalController + NestedLoRA,
    con controllo di rank E learning rate, e stati orbitali HIGH/BASE/LOW.

DISEGNO: identico ai test Tinker, così i risultati sono confrontabili.
  - Task strutturato (prompt -> target), non token casuali.
  - Finestra di SHOCK nota (target corrotti) a step 80-120.
  - Log per step di: loss, φ (phi_ema), stato orbitale, rank attivo, LR.
  - Verdetto: rapporto φ dentro/fuori shock + traiettoria di rank e stato.

CORREZIONI rispetto alla prima versione:
  1. import da orbital_controller (non controller)
  2. si usa l'optimizer restituito da setup (adapters è un dict!)
  3. set_rank(model, rank) applicato ad ogni step  <- il rank era inerte
  4. φ letto da ctrl.phi_ema (l'attributo "phi" non esiste -> era sempre 0.0)
  5. model.to(device) DOPO l'iniezione (gli adapter nascono su CPU/fp32)
  6. labels mascherate a -100 su prompt e padding
=============================================================================
"""
import subprocess, sys, os, importlib

# ── setup ────────────────────────────────────────────────────────────────
if not os.path.exists("/content/Unified-LoRa/nested_lora.py"):
    subprocess.run(["rm", "-rf", "/content/Unified-LoRa"])
    subprocess.run(["git", "clone", "-q",
                    "https://github.com/Sva76/Unified-LoRa.git",
                    "/content/Unified-LoRa"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "accelerate"], check=True)

sys.path.insert(0, "/content/Unified-LoRa")
importlib.invalidate_caches()

import torch, json, random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from orbital_controller import setup_unified_lora          # <- non "controller"
from nested_lora import set_rank, NestedLoRALinear

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"   # separa q_proj/v_proj, fp32, leggero
STEPS = 200
SHOCK_START, SHOCK_END = 80, 120
SEED = 11
BASE_LR = 3e-4
MAXLEN = 64

torch.manual_seed(SEED)
rng = np.random.default_rng(SEED)

# ── modello ──────────────────────────────────────────────────────────────
tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)

# ── controller: restituisce (adapters, ctrl, optimizer) ──────────────────
adapters, ctrl, optimizer = setup_unified_lora(
    model,
    target_modules=["q_proj", "v_proj"],
    max_rank=16,
    alpha=16.0,
    base_lr=BASE_LR,
)
model.to(DEV)                      # <- DOPO l'iniezione: adapter su GPU
n_adapters = sum(1 for m in model.modules() if isinstance(m, NestedLoRALinear))
print(f"Adapter iniettati: {n_adapters}   (se 0 -> target_modules sbagliati!)")
assert n_adapters > 0, "Nessun adapter iniettato: controlla target_modules."

# ── dati: task strutturato + shock ───────────────────────────────────────
SUBJ = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
        "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
RELS = ["is near the","is far from the","is above the","is below the","is beside the"]
CORRUPT = ["zxqw","7391","blorp","asdf","9920","qqzz","1k4m","wronk","vvxy","000z"]

r = random.Random(SEED)
subs = SUBJ[:]; r.shuffle(subs)
PAIRS = [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]

def make_batch(step):
    shocked = SHOCK_START <= step < SHOCK_END
    prompt, target = PAIRS[step % len(PAIRS)]
    if shocked:
        target = CORRUPT[rng.integers(len(CORRUPT))]
    p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
    t_ids = tok(" " + target, add_special_tokens=False)["input_ids"]
    ids = (p_ids + t_ids)[:MAXLEN]
    # labels: -100 sul prompt (e su eventuale padding) -> loss solo sul target
    labels = ([-100] * len(p_ids) + t_ids)[:MAXLEN]
    pad = MAXLEN - len(ids)
    attn = [1] * len(ids) + [0] * pad
    ids = ids + [tok.pad_token_id] * pad
    labels = labels + [-100] * pad
    return (torch.tensor([ids]).to(DEV),
            torch.tensor([attn]).to(DEV),
            torch.tensor([labels]).to(DEV),
            shocked)

# ── loop ─────────────────────────────────────────────────────────────────
log = []
model.train()
for step in range(STEPS):
    ids, attn, labels, shocked = make_batch(step)
    out = model(input_ids=ids, attention_mask=attn, labels=labels)
    loss = out.loss
    if not torch.isfinite(loss):
        print(f"step {step}: loss non finita, interrompo"); break
    loss.backward()

    rank = ctrl.step(loss.item())        # <- restituisce il rank...
    set_rank(model, rank)                # <- ...e QUI lo applichiamo davvero

    optimizer.step()
    optimizer.zero_grad()

    s = ctrl.get_summary()               # phi, state, rank, lr
    log.append({"step": step, "loss": float(loss.item()),
                "phi": s["phi"], "state": s["state"],
                "rank": s["rank"], "lr": s["lr"], "shock": int(shocked)})

    if step % 10 == 0 or SHOCK_START - 2 <= step <= SHOCK_END + 2:
        tag = "  <-- SHOCK" if shocked else ""
        print(f"step {step:3d} | loss {loss.item():7.4f} | φ {s['phi']:8.5f} "
              f"| {s['state']:<5} | rank {s['rank']:2d} | lr {s['lr']:.2e}{tag}")

# ── verdetto ─────────────────────────────────────────────────────────────
arr_phi_in  = np.array([e["phi"] for e in log if e["shock"] == 1])
arr_phi_out = np.array([e["phi"] for e in log if e["shock"] == 0])
states = [e["state"] for e in log]
ranks  = [e["rank"] for e in log]

print("\n================= VERDETTO =================")
print(f"φ medio DURANTE shock : {arr_phi_in.mean():.5f}")
print(f"φ medio FUORI shock   : {arr_phi_out.mean():.5f}")
if arr_phi_out.mean() > 0:
    print(f"rapporto dentro/fuori : {arr_phi_in.mean()/arr_phi_out.mean():.2f}x")
print(f"stati visitati        : {sorted(set(states))}")
print(f"rank min/max          : {min(ranks)} / {max(ranks)}")
print(f"transizioni di stato  : {sum(1 for i in range(1,len(states)) if states[i]!=states[i-1])}")

with open("/content/log_controller_full.json", "w") as f:
    json.dump(log, f, indent=2)
print("\nLog: /content/log_controller_full.json")

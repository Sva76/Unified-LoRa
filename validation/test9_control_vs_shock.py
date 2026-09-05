"""
=============================================================================
CONTROLLER UNIFIED-LORA — RUN DI CONTROLLO + SENSIBILITA' DELLA SOGLIA
=============================================================================
Due domande aperte dal run precedente (rapporto 2.12x, stato LOW mai visitato):

  Q1. Il 2.12x e' reale? Serve il BRACCIO DI CONTROLLO: stesso task, stesso
      seed, MA SENZA shock. Solo confrontando shock vs controllo sulla
      STESSA finestra di step il rapporto e' interpretabile (il baseline di
      phi non e' stazionario, quindi il confronto dentro/fuori nello stesso
      run e' distorto).

  Q2. LOW non scatta perche' la soglia e' adattiva (stress_thresh = mu+k*sigma
      calcolata sulla storia di phi stessa): se phi resta alto, mu sale e la
      soglia gli corre dietro. Test diretto: abbassare stress_k e vedere se
      LOW si attiva e il rapporto migliora.

DISEGNO: 4 bracci = {controllo, shock} x {stress_k=1.5 (default), 0.5 (sensibile)}
  Stesso seed e stessi dati in tutti; l'unica differenza e' la corruzione
  dei target nella finestra 80-120 e il parametro di soglia.

METRICA PRINCIPALE (appaiata):
  phi_medio(shock, step 80-120)  /  phi_medio(controllo, step 80-120)
  -> quanto lo shock alza phi RISPETTO A un training identico senza shock.
=============================================================================
"""
import subprocess, sys, os, importlib

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
from orbital_controller import setup_unified_lora
from nested_lora import set_rank, NestedLoRALinear

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS = 200
SHOCK_START, SHOCK_END = 80, 120
SEED = 11
BASE_LR = 3e-4
MAXLEN = 64

SUBJ = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
        "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
RELS = ["is near the","is far from the","is above the","is below the","is beside the"]
CORRUPT = ["zxqw","7391","blorp","asdf","9920","qqzz","1k4m","wronk","vvxy","000z"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def build_pairs(seed):
    r = random.Random(seed)
    subs = SUBJ[:]; r.shuffle(subs)
    return [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]


def run_arm(with_shock, stress_k, label):
    """Un braccio. with_shock=False -> nessuna corruzione (controllo)."""
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    pairs = build_pairs(SEED)

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    adapters, ctrl, optimizer = setup_unified_lora(
        model, target_modules=["q_proj", "v_proj"],
        max_rank=16, alpha=16.0, base_lr=BASE_LR,
        stress_k=stress_k,
    )
    model.to(DEV)
    n_ad = sum(1 for m in model.modules() if isinstance(m, NestedLoRALinear))
    assert n_ad > 0, "nessun adapter iniettato"

    log = []
    model.train()
    for step in range(STEPS):
        shocked = with_shock and (SHOCK_START <= step < SHOCK_END)
        prompt, target = pairs[step % len(pairs)]
        if shocked:
            target = CORRUPT[rng.integers(len(CORRUPT))]
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(" " + target, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[:MAXLEN]
        labels = ([-100]*len(p_ids) + t_ids)[:MAXLEN]
        pad = MAXLEN - len(ids)
        attn = [1]*len(ids) + [0]*pad
        ids = ids + [tok.pad_token_id]*pad
        labels = labels + [-100]*pad

        out = model(input_ids=torch.tensor([ids]).to(DEV),
                    attention_mask=torch.tensor([attn]).to(DEV),
                    labels=torch.tensor([labels]).to(DEV))
        loss = out.loss
        if not torch.isfinite(loss):
            print(f"  [{label}] loss non finita a step {step}"); break
        loss.backward()
        rank = ctrl.step(loss.item())
        set_rank(model, rank)
        optimizer.step(); optimizer.zero_grad()

        s = ctrl.get_summary()
        log.append({"step": step, "loss": float(loss.item()), "phi": s["phi"], "phi_raw": s["phi_raw"], "phi_ema": s["phi_ema"],
                    "state": s["state"], "rank": s["rank"], "lr": s["lr"]})

    del model; torch.cuda.empty_cache()

    win = [e["phi"] for e in log if SHOCK_START <= e["step"] < SHOCK_END]
    states = [e["state"] for e in log]
    res = {
        "label": label, "with_shock": with_shock, "stress_k": stress_k,
        "phi_window": float(np.mean(win)) if win else float("nan"),
        "phi_all": float(np.mean([e["phi"] for e in log])),
        "states": sorted(set(states)),
        "low_visits": sum(1 for i in range(1, len(states))
                          if states[i] == "LOW" and states[i-1] != "LOW"),
        "transitions": sum(1 for i in range(1, len(states)) if states[i] != states[i-1]),
        "rank_min": min(e["rank"] for e in log), "rank_max": max(e["rank"] for e in log),
        "log": log,
    }
    print(f"  [{label}] phi(finestra)={res['phi_window']:.4f} | "
          f"stati={res['states']} | ingressi LOW={res['low_visits']} | "
          f"transizioni={res['transitions']}")
    return res


print(f"4 BRACCI su {MODEL_ID} | seed {SEED} | {STEPS} step ciascuno\n")
results = {}
for k in [1.5, 0.5]:
    print(f"--- stress_k = {k} ---")
    results[("control", k)] = run_arm(False, k, f"controllo k={k}")
    results[("shock",   k)] = run_arm(True,  k, f"shock     k={k}")
    print()

print("================= VERDETTO =================")
for k in [1.5, 0.5]:
    c = results[("control", k)]["phi_window"]
    s = results[("shock", k)]["phi_window"]
    ratio = s / c if c > 0 else float("nan")
    low_c = results[("control", k)]["low_visits"]
    low_s = results[("shock", k)]["low_visits"]
    print(f"stress_k={k}:  phi controllo={c:.4f} | phi shock={s:.4f} | "
          f"RAPPORTO APPAIATO = {ratio:.2f}x   (LOW: controllo {low_c}, shock {low_s})")

print("\nLettura:")
print(" - rapporto appaiato ~1x  -> lo shock NON alza phi rispetto a un training identico")
print(" - LOW=0 anche con k=0.5  -> la soglia adattiva impedisce il rilevamento sostenuto")
print(" - LOW>0 solo con k=0.5   -> ipotesi confermata: e' un problema di calibrazione")

with open("/content/log_control_vs_shock.json", "w") as f:
    json.dump({f"{a}_k{k}": {kk: vv for kk, vv in r.items() if kk != "log"}
               for (a, k), r in results.items()}, f, indent=2)
print("\nLog: /content/log_control_vs_shock.json")

"""
=============================================================================
CONTROLLER UNIFIED-LORA — LATENZA DELLA FSM (stable_window)
=============================================================================
IPOTESI DA TESTARE (emersa dai run precedenti):
  Lo stato LOW non si attiva non perche' la soglia sia mal calibrata, ma
  perche' la FSM e' CIECA: 'stable_window' le impedisce di rivalutare per N
  step dopo una transizione. Nel run precedente la FSM e' entrata in BASE
  allo step 84 (in piena finestra di shock) e non ha potuto riguardare fino
  al 114 -- quando il picco di phi (step 87-93) era gia' passato.

  Sul banco sintetico, con stable_window ridotta, LOW si attiva. Qui lo
  verifichiamo sul modello reale.

DISEGNO: griglia stable_window x stress_k, tutti con shock a 80-120.
  stable_window in {30 (default), 10, 5}   <- variabile principale
  stress_k       in {1.5 (default), 0.5}   <- controllo secondario
  + 1 braccio di CONTROLLO (senza shock) alla configurazione piu' sensibile,
    per il rapporto appaiato.

METRICA NUOVA E CENTRALE: LATENZA DI RILEVAMENTO
  = (primo step in cui la FSM entra in LOW) - (inizio shock, step 80)
  Se LOW non scatta mai -> latenza non definita = rilevamento fallito.
  Una latenza bassa significa che il controller reagisce mentre lo stress
  e' in corso; una latenza alta (o assente) significa che arriva tardi.
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


def run_arm(with_shock, stable_window, stress_k, label):
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    r = random.Random(SEED)
    subs = SUBJ[:]; r.shuffle(subs)
    pairs = [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    adapters, ctrl, optimizer = setup_unified_lora(
        model, target_modules=["q_proj", "v_proj"],
        max_rank=16, alpha=16.0, base_lr=BASE_LR,
        stress_k=stress_k, stable_window=stable_window,
    )
    model.to(DEV)
    assert sum(1 for m in model.modules() if isinstance(m, NestedLoRALinear)) > 0

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
        log.append({"step": step, "loss": float(loss.item()), "phi": s["phi"],
                    "state": s["state"], "rank": s["rank"], "lr": s["lr"]})

    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

    states = [e["state"] for e in log]
    # primo ingresso in LOW dopo l'inizio dello shock
    first_low = None
    for e in log:
        if e["state"] == "LOW" and e["step"] >= SHOCK_START:
            first_low = e["step"]; break
    latency = (first_low - SHOCK_START) if first_low is not None else None
    win = [e["phi"] for e in log if SHOCK_START <= e["step"] < SHOCK_END]

    res = {"label": label, "with_shock": with_shock,
           "stable_window": stable_window, "stress_k": stress_k,
           "phi_window": float(np.mean(win)) if win else float("nan"),
           "states": sorted(set(states)),
           "low_steps": sum(1 for s in states if s == "LOW"),
           "first_low": first_low, "latency": latency,
           "transitions": sum(1 for i in range(1, len(states)) if states[i] != states[i-1]),
           "rank_min": min(e["rank"] for e in log),
           "rank_max": max(e["rank"] for e in log)}

    lat_txt = f"{latency} step" if latency is not None else "MAI"
    print(f"  [{label}] phi(fin)={res['phi_window']:.3f} | stati={res['states']} | "
          f"step in LOW={res['low_steps']} | latenza={lat_txt} | "
          f"transizioni={res['transitions']}")
    return res


print(f"LATENZA FSM su {MODEL_ID} | seed {SEED} | shock {SHOCK_START}-{SHOCK_END}\n")
results = []
for sw in [30, 10, 5]:
    print(f"--- stable_window = {sw} ---")
    for k in [1.5, 0.5]:
        results.append(run_arm(True, sw, k, f"shock sw={sw} k={k}"))
    print()

print("--- controllo (senza shock) alla config piu' sensibile ---")
ctrl_arm = run_arm(False, 5, 0.5, "controllo sw=5 k=0.5")
print()

print("================= VERDETTO =================")
print(f"{'config':<22}{'phi(fin)':<11}{'step LOW':<11}{'latenza':<12}{'transizioni'}")
for r in results:
    lat = f"{r['latency']}" if r['latency'] is not None else "MAI"
    print(f"sw={r['stable_window']:<3} k={r['stress_k']:<5}      "
          f"{r['phi_window']:<11.3f}{r['low_steps']:<11}{lat:<12}{r['transitions']}")

c = ctrl_arm["phi_window"]
best = min(results, key=lambda r: (r["latency"] is None, r["latency"] or 999))
print(f"\nRapporto appaiato (shock sw=5 k=0.5 / controllo): "
      f"{results[-1]['phi_window']/c:.2f}x")

any_low = [r for r in results if r["low_steps"] > 0]
print("\nLettura:")
if not any_low:
    print(" - LOW non scatta MAI, con nessuna finestra -> il problema non e' la")
    print("   latenza: il rilevamento di stress sostenuto e' strutturalmente")
    print("   impedito dalla soglia adattiva (mu insegue phi).")
else:
    print(f" - LOW si attiva in {len(any_low)}/{len(results)} configurazioni.")
    print(f" - Migliore: {best['label']} con latenza {best['latency']} step.")
    print(" - Se LOW scatta solo con finestre corte -> ipotesi LATENZA confermata:")
    print("   il controller puo' rilevare, ma l'isteresi lo rende troppo lento.")

with open("/content/log_fsm_latency.json", "w") as f:
    json.dump({"shock_arms": results, "control": ctrl_arm}, f, indent=2)
print("\nLog: /content/log_fsm_latency.json")

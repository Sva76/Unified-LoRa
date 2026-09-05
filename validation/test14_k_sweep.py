"""
=============================================================================
TEST 14 — stress_k ha un effetto? (sw fisso a 10, attuatore ACCESO)
=============================================================================
DOMANDA. Nei run di Test 10, k=1.5 e k=0.5 hanno dato risultati identici
cifra per cifra in tutte e tre le finestre. Due spiegazioni possibili:
  (a) k e' inerte -> difetto strutturale del controller;
  (b) l'isteresi (stable_window) maschera k: quando la FSM e' finalmente
      abilitata a valutare, phi e' gia' sopra tutte le soglie basse.

L'analisi offline sulle tracce pulite del seme 11 favorisce (b):
  a step 81 vengono attraversate mu+0.1s e mu+0.5s; a step 82 mu+1.5s;
  mu+6.0s (soglia 4.4-5.0) NON viene mai attraversata (phi max 4.36).

PREVISIONE PRE-REGISTRATA (dichiarata prima dell'esecuzione):
  - k=0.1 e k=1.5  -> LOW scatta, primo ingresso a step 84 (gate a sw=10)
  - k=6.0          -> LOW non scatta MAI nel braccio shock
  Se k=6.0 scatta comunque a step 84, la previsione e' falsificata e
  l'ipotesi (a) — k inerte — resta in piedi.

CRITERIO DI PUNTO OPERATIVO VALIDO (dichiarato prima dell'esecuzione):
  un k e' valido se e solo se, alla stessa configurazione:
     braccio SHOCK   -> primo LOW a step >= 80   (rilevamento)
     braccio CONTROLLO -> nessun LOW in tutto il run  (specificita')
  Test 10 non controllava la seconda condizione: il suo verdetto contava
  solo i bracci shock. Qui entrambe le condizioni sono verificate.

DISEGNO: 6 run da 200 step (3 valori di k x {shock, controllo}), seme 11,
  stable_window=10 fisso. Durata attesa su T4: ~50 minuti.

OUTPUT: oltre al riepilogo, salva le tracce per-step complete, cosi' il
  verdetto e' ricalcolabile offline senza GPU.
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
from orbital_controller import setup_unified_lora
from nested_lora import set_rank, NestedLoRALinear

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS = 200
SHOCK_START, SHOCK_END = 80, 120
SEED = 11
BASE_LR = 3e-4
MAXLEN = 64

STABLE_WINDOW = 10          # fisso: e' k che varia
K_VALUES = [0.1, 1.5, 6.0]  # 1.5 replica Test 10 -> controllo di equivalenza

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
        log.append({"step": step, "loss": float(loss.item()), "phi": s["phi"], "phi_raw": s["phi_raw"], "phi_ema": s["phi_ema"],
                    "state": s["state"], "rank": s["rank"], "lr": s["lr"]})

    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

    states = [e["state"] for e in log]

    # CORREZIONE rispetto a Test 10: si registra il primo LOW OVUNQUE, non
    # solo dopo lo step 80. Nel braccio di controllo un LOW e' un falso
    # positivo a qualunque step avvenga, e la vecchia versione lo perdeva.
    first_low_any = next((e["step"] for e in log if e["state"] == "LOW"), None)
    first_low_post = next((e["step"] for e in log
                           if e["state"] == "LOW" and e["step"] >= SHOCK_START), None)
    # la latenza ha senso solo dove uno shock esiste davvero
    latency = (first_low_post - SHOCK_START) if (with_shock and first_low_post is not None) else None

    win = [e["phi"] for e in log if SHOCK_START <= e["step"] < SHOCK_END]

    res = {"label": label, "with_shock": with_shock,
           "stable_window": stable_window, "stress_k": stress_k,
           "phi_window": float(np.mean(win)) if win else float("nan"),
           "states": sorted(set(states)),
           "low_steps": sum(1 for s in states if s == "LOW"),
           "first_low_any": first_low_any,
           "first_low_post_onset": first_low_post,
           "latency": latency,
           "transitions": sum(1 for i in range(1, len(states)) if states[i] != states[i-1]),
           "rank_min": min(e["rank"] for e in log),
           "rank_max": max(e["rank"] for e in log),
           "trace": log}

    lat_txt = f"{latency} step" if latency is not None else ("MAI" if with_shock else "-")
    fp_txt = "" if with_shock else ("  <-- FALSO POSITIVO" if first_low_any is not None else "  (nessun LOW: OK)")
    print(f"  [{label}] phi(fin)={res['phi_window']:.3f} | primo LOW={first_low_any} | "
          f"latenza={lat_txt} | transizioni={res['transitions']} | "
          f"rank {res['rank_min']}-{res['rank_max']}{fp_txt}")
    return res


# ── esecuzione ───────────────────────────────────────────────────────────
print(f"TEST 14 — effetto di stress_k | {MODEL_ID} | seme {SEED} | "
      f"stable_window={STABLE_WINDOW}\n")

arms = []
for k in K_VALUES:
    print(f"--- stress_k = {k} ---")
    arms.append(run_arm(True,  STABLE_WINDOW, k, f"shock      k={k}"))
    arms.append(run_arm(False, STABLE_WINDOW, k, f"controllo  k={k}"))
    print()

# ── verdetto ─────────────────────────────────────────────────────────────
print("================= VERDETTO =================\n")
print(f"{'k':<8}{'shock: primo LOW':<20}{'latenza':<12}"
      f"{'controllo: primo LOW':<24}{'punto valido?'}")

valid_ks = []
for k in K_VALUES:
    sh = next(a for a in arms if a["with_shock"] and a["stress_k"] == k)
    ct = next(a for a in arms if not a["with_shock"] and a["stress_k"] == k)
    rileva = sh["first_low_post_onset"] is not None
    specifico = ct["first_low_any"] is None
    valido = rileva and specifico
    if valido:
        valid_ks.append(k)
    print(f"{k:<8}{str(sh['first_low_any']):<20}"
          f"{str(sh['latency']) if sh['latency'] is not None else 'MAI':<12}"
          f"{str(ct['first_low_any']):<24}{'SI' if valido else 'no'}")

print("\n--- previsione pre-registrata ---")
sh6 = next(a for a in arms if a["with_shock"] and a["stress_k"] == 6.0)
if sh6["first_low_any"] is None:
    print("  k=6.0 non scatta -> PREVISIONE CONFERMATA: k funziona,")
    print("  ed era l'isteresi a mascherarlo nei run di Test 10.")
else:
    print(f"  k=6.0 scatta comunque (step {sh6['first_low_any']}) -> PREVISIONE")
    print("  FALSIFICATA: k non morde, e' un difetto strutturale del controller.")

print("\n--- punto operativo ---")
if valid_ks:
    print(f"  Esistono k validi (rileva + zero falsi positivi): {valid_ks}")
else:
    print("  NESSUN k soddisfa entrambe le condizioni: con l'attuatore acceso")
    print("  non esiste un punto operativo valido a questa configurazione.")

print("\n--- equivalenza con Test 10 (k=1.5, sw=10) ---")
sh15 = next(a for a in arms if a["with_shock"] and a["stress_k"] == 1.5)
print(f"  atteso da Test 10: primo LOW=84, phi(fin)=3.892, transizioni=8")
print(f"  ottenuto ora:      primo LOW={sh15['first_low_any']}, "
      f"phi(fin)={sh15['phi_window']:.3f}, transizioni={sh15['transitions']}")

# ── salvataggio: Drive se possibile, sempre in /content ──────────────────
payload = {
    "config": {"model": MODEL_ID, "seed": SEED, "steps": STEPS,
               "shock_window": [SHOCK_START, SHOCK_END],
               "stable_window": STABLE_WINDOW, "k_values": K_VALUES,
               "base_lr": BASE_LR, "max_rank": 16},
    "criterio_preregistrato": "k valido <=> shock: primo LOW >= 80 AND controllo: nessun LOW",
    "arms": arms,
}
paths = ["/content/log_k_sweep.json"]
try:
    from google.colab import drive
    drive.mount("/content/drive")
    paths.append("/content/drive/MyDrive/log_k_sweep.json")
except Exception as e:
    print(f"\n(Drive non montato: {e})")

for p in paths:
    with open(p, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"salvato: {p}")

try:
    from google.colab import files
    files.download("/content/log_k_sweep.json")
except Exception as e:
    print(f"(download automatico non riuscito: {e} — scaricalo a mano da /content)")

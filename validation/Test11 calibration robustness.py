"""
=============================================================================
CALIBRAZIONE: LA CELLA CHE DISCRIMINA E' ROBUSTA? ESISTE UNA REGIONE?
=============================================================================
CONTESTO: con seed 11, la configurazione (stable_window=10, stress_k=1.5)
e' l'unica in cui lo stato LOW si attiva SOLO sotto shock (10 step) e MAI
nel controllo (0 step). A sw=30 la FSM e' cieca, a sw=5 oscilla in entrambi.

DUE DOMANDE:
  Q1. E' robusta o e' fortuna di un seed?   -> 5 semi sulla cella (10, 1.5)
  Q2. E' un punto isolato o una REGIONE?    -> celle vicine sw in {8, 12}

CRITERIO DI DISCRIMINAZIONE PULITA (deciso PRIMA):
  un seed "discrimina" se   LOW_shock > 0  E  LOW_controllo == 0
  Riportiamo: quanti semi discriminano, la latenza media, il rapporto phi.

ATTENZIONE: qui l'onset dello shock e' FISSO (80-120). Non stiamo testando
il rilevamento alla cieca, ma la ROBUSTEZZA DELLA CALIBRAZIONE. Sono due
domande diverse e vanno riportate separatamente.

Tempo stimato: ~22 run x ~1.5 min = 30-40 minuti su T4.
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
BASE_LR = 3e-4
MAXLEN = 64
STRESS_K = 1.5                       # fissato: e' sw la variabile che conta

PRIMARY_SW = 10
PRIMARY_SEEDS = [11, 23, 37, 41, 53]
NEIGHBOUR_SW = [8, 12]
NEIGHBOUR_SEEDS = [11, 23, 37]

SUBJ = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
        "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
RELS = ["is near the","is far from the","is above the","is below the","is beside the"]
CORRUPT = ["zxqw","7391","blorp","asdf","9920","qqzz","1k4m","wronk","vvxy","000z"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def run(seed, sw, with_shock):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    subs = SUBJ[:]; r.shuffle(subs)
    pairs = [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    adapters, ctrl, optimizer = setup_unified_lora(
        model, target_modules=["q_proj", "v_proj"],
        max_rank=16, alpha=16.0, base_lr=BASE_LR,
        stress_k=STRESS_K, stable_window=sw)
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
            break
        loss.backward()
        rank = ctrl.step(loss.item())
        set_rank(model, rank)
        optimizer.step(); optimizer.zero_grad()
        s = ctrl.get_summary()
        log.append({"step": step, "phi": s["phi"], "state": s["state"]})

    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()

    states = [e["state"] for e in log]
    win = [e["phi"] for e in log if SHOCK_START <= e["step"] < SHOCK_END]
    first_low = next((e["step"] for e in log
                      if e["state"] == "LOW" and e["step"] >= SHOCK_START), None)
    return {"phi_window": float(np.mean(win)) if win else float("nan"),
            "low_steps": sum(1 for s in states if s == "LOW"),
            "latency": (first_low - SHOCK_START) if first_low is not None else None,
            "transitions": sum(1 for i in range(1, len(states)) if states[i] != states[i-1])}


def evaluate_cell(sw, seeds, title):
    print(f"\n--- {title}  (stable_window={sw}, stress_k={STRESS_K}) ---")
    print(f"{'seed':<7}{'LOW shock':<12}{'LOW ctrl':<11}{'latenza':<11}"
          f"{'phi shock':<12}{'phi ctrl':<11}{'rapp.'}")
    rows = []
    for sd in seeds:
        sh = run(sd, sw, True)
        ct = run(sd, sw, False)
        ratio = sh["phi_window"] / ct["phi_window"] if ct["phi_window"] > 0 else float("nan")
        clean = (sh["low_steps"] > 0 and ct["low_steps"] == 0)
        lat = f"{sh['latency']}" if sh["latency"] is not None else "MAI"
        print(f"{sd:<7}{sh['low_steps']:<12}{ct['low_steps']:<11}{lat:<11}"
              f"{sh['phi_window']:<12.3f}{ct['phi_window']:<11.3f}{ratio:.2f}x"
              f"{'   <- pulita' if clean else ''}")
        rows.append({"seed": sd, "sw": sw, "shock": sh, "control": ct,
                     "ratio": ratio, "clean": clean})
    n_clean = sum(1 for r in rows if r["clean"])
    lats = [r["shock"]["latency"] for r in rows if r["shock"]["latency"] is not None]
    ratios = [r["ratio"] for r in rows if np.isfinite(r["ratio"])]
    print(f"  => discriminazione pulita: {n_clean}/{len(seeds)} semi | "
          f"latenza media: {np.mean(lats):.1f} step | " if lats else
          f"  => discriminazione pulita: {n_clean}/{len(seeds)} semi | latenza: MAI | ")
    print(f"     rapporto phi medio: {np.mean(ratios):.2f}x ± {np.std(ratios):.2f}")
    return rows, n_clean


print(f"CALIBRAZIONE su {MODEL_ID} | shock {SHOCK_START}-{SHOCK_END} | k={STRESS_K}")
all_rows = {}

rows, n_primary = evaluate_cell(PRIMARY_SW, PRIMARY_SEEDS, "CELLA PRINCIPALE")
all_rows[PRIMARY_SW] = rows

neigh = {}
for sw in NEIGHBOUR_SW:
    rows_n, n_c = evaluate_cell(sw, NEIGHBOUR_SEEDS, "CELLA VICINA")
    all_rows[sw] = rows_n
    neigh[sw] = (n_c, len(NEIGHBOUR_SEEDS))

print("\n================= VERDETTO =================")
print(f"sw={PRIMARY_SW} (principale): {n_primary}/{len(PRIMARY_SEEDS)} semi discriminano puliti")
for sw in NEIGHBOUR_SW:
    n_c, tot = neigh[sw]
    print(f"sw={sw} (vicina)     : {n_c}/{tot} semi discriminano puliti")

print("\nLettura:")
if n_primary >= 4:
    print(" - La cella e' ROBUSTA: non era fortuna del seed 11.")
elif n_primary >= 2:
    print(" - Robustezza PARZIALE: funziona in alcuni semi, non in tutti.")
else:
    print(" - NON robusta: il risultato del seed 11 era un caso isolato.")
if any(neigh[sw][0] >= 2 for sw in NEIGHBOUR_SW):
    print(" - Anche le celle vicine discriminano -> esiste una REGIONE di")
    print("   calibrazione, non un punto. Vale la pena mapparla meglio.")
else:
    print(" - Le celle vicine NON discriminano -> il funzionamento e' confinato")
    print("   a un punto stretto: fragile, difficile da usare in pratica.")

with open("/content/log_calibration.json", "w") as f:
    json.dump({str(sw): [{k: v for k, v in r.items()} for r in rr]
               for sw, rr in all_rows.items()}, f, indent=2, default=str)
print("\nLog: /content/log_calibration.json")

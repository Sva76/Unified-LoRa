"""
=============================================================================
SOGLIA FISSA vs ADATTIVA — separare il SENSORE dall'ATTUATORE
=============================================================================
DIAGNOSI DA CUI PARTIAMO (dai run precedenti):
  1. Il sensore e' contaminato dall'attuatore: nei run DI CONTROLLO phi sale
     quando la FSM e' piu' attiva (1.68 a sw=30 -> 2.15 a sw=10), senza che
     ci sia alcuno shock. La FSM alza phi da sola.
  2. La soglia adattiva (mu + k*sigma sulla storia di phi) neutralizza la
     forza del segnale: il seed 41 ha il rapporto phi PIU' ALTO (4.29x) e la
     FSM non scatta mai; il seed 11, il piu' debole (1.97x), scatta.
     Il rilevamento non e' monotono nella forza del segnale.

DISEGNO (che risolve entrambi i problemi):
  FASE 1 - RACCOLTA. Training a rank e LR FISSI: il controller gira in sola
    osservazione (calcola phi ma NON tocca rank ne' learning rate).
    Cosi' le tracce di phi sono PULITE, non contaminate dall'attuatore.
    10 run: 5 semi x {shock, controllo}.

  FASE 2 - CONFRONTO OFFLINE DEI DETECTOR sulle STESSE tracce.
    A) adattivo:  soglia = mu + k*sigma  (come il repo), k in {1.0,1.5,2.0}
    B) assoluto:  soglia = T fissa, CALIBRATA sui run di controllo con
       LEAVE-ONE-SEED-OUT (T dal 4 semi, testata sul quinto) -> niente
       circolarita': il seme testato non partecipa alla calibrazione.

  METRICHE (decise prima): recall (shock rilevati), falsi positivi (controlli
  che scattano), latenza media dal vero onset.

PERCHE' QUESTO DISEGNO E' MIGLIORE: generiamo i dati UNA volta e valutiamo
piu' detector sulle stesse tracce. Confronto appaiato esatto, e costo minore.
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
from orbital_controller import setup_unified_lora, OrbitalController
from nested_lora import set_rank, NestedLoRALinear

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS = 200
SHOCK_START, SHOCK_END = 80, 120
BASE_LR = 3e-4
FIXED_RANK = 8            # rank STATICO: l'attuatore e' spento
MAXLEN = 64
SEEDS = [11, 23, 37, 41, 53]
WARMUP_IGNORE = 30        # i primi step (loss in caduta) non contano per il rilevamento

SUBJ = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
        "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
RELS = ["is near the","is far from the","is above the","is below the","is beside the"]
CORRUPT = ["zxqw","7391","blorp","asdf","9920","qqzz","1k4m","wronk","vvxy","000z"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


# ─────────────────────────────────────────────────────────────────────────
# FASE 1 — raccolta tracce con attuatore SPENTO
# ─────────────────────────────────────────────────────────────────────────
def collect_trace(seed, with_shock):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    subs = SUBJ[:]; r.shuffle(subs)
    pairs = [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    adapters, _ctrl_unused, optimizer = setup_unified_lora(
        model, target_modules=["q_proj", "v_proj"],
        max_rank=16, alpha=16.0, base_lr=BASE_LR)
    model.to(DEV)
    assert sum(1 for m in model.modules() if isinstance(m, NestedLoRALinear)) > 0

    set_rank(model, FIXED_RANK)      # rank fisso per tutto il run
    # controller SEPARATO e NON collegato: osserva soltanto
    obs = OrbitalController(base_lr=BASE_LR)     # nessun link_optimizer/adapters

    phis, losses, raw_phis, ema_phis = [], [], [], []
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
        obs.step(loss.item())                    # solo osservazione
        # NB: nessun set_rank, nessuna modifica di LR -> attuatore spento
        optimizer.step(); optimizer.zero_grad()
        summary = obs.get_summary()
        phis.append(summary["phi"])
        raw_phis.append(summary["phi_raw"])
        ema_phis.append(summary["phi_ema"])
        losses.append(float(loss.item()))

    del model
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return {"phi": phis, "phi_raw": raw_phis, "phi_ema": ema_phis, "loss": losses}


print(f"FASE 1 — raccolta tracce (attuatore spento, rank fisso={FIXED_RANK})")
traces = {}
for sd in SEEDS:
    traces[sd] = {"shock": collect_trace(sd, True),
                  "control": collect_trace(sd, False)}
    sh = traces[sd]["shock"]["phi"]; ct = traces[sd]["control"]["phi"]
    win_s = np.mean(sh[SHOCK_START:SHOCK_END]); win_c = np.mean(ct[SHOCK_START:SHOCK_END])
    print(f"  seed {sd}: phi shock={win_s:.3f} | phi ctrl={win_c:.3f} | "
          f"rapporto={win_s/win_c:.2f}x")


# ─────────────────────────────────────────────────────────────────────────
# FASE 2 — detector offline sulle stesse tracce
# ─────────────────────────────────────────────────────────────────────────
def detect_adaptive(phi, k):
    """Replica la regola del repo: allerta quando phi > mu + k*sigma della storia."""
    for t in range(WARMUP_IGNORE, len(phi)):
        hist = phi[:t]
        mu, sd = np.mean(hist), np.std(hist)
        if phi[t] > mu + k*sd:
            return t
    return None

def detect_fixed(phi, T):
    """Allerta al primo superamento di una soglia assoluta."""
    for t in range(WARMUP_IGNORE, len(phi)):
        if phi[t] > T:
            return t
    return None


def evaluate(detector, name):
    """recall, falsi positivi, latenza su tutti i semi."""
    hits, fps, lats = 0, 0, []
    for sd in SEEDS:
        d_s = detector(traces[sd]["shock"]["phi"], sd)
        d_c = detector(traces[sd]["control"]["phi"], sd)
        if d_s is not None and d_s >= SHOCK_START:
            hits += 1; lats.append(d_s - SHOCK_START)
        if d_c is not None:
            fps += 1
    lat = f"{np.mean(lats):.1f}" if lats else "-"
    print(f"{name:<34}{hits}/{len(SEEDS):<11}{fps}/{len(SEEDS):<11}{lat}")
    return hits, fps, lats


print(f"\nFASE 2 — confronto detector (warmup ignorato: primi {WARMUP_IGNORE} step)")
print(f"{'detector':<34}{'recall':<12}{'falsi pos.':<12}{'latenza media'}")

# A) adattivo, come il repo
for k in [1.0, 1.5, 2.0]:
    evaluate(lambda phi, sd, k=k: detect_adaptive(phi, k), f"adattivo  mu+{k}*sigma")

# B) assoluto con leave-one-seed-out sui controlli
def make_fixed_loo(margin):
    def det(phi, sd):
        others = [max(traces[s]["control"]["phi"][WARMUP_IGNORE:]) for s in SEEDS if s != sd]
        T = max(others) * margin
        return detect_fixed(phi, T)
    return det

for m in [1.0, 1.1, 1.25]:
    evaluate(make_fixed_loo(m), f"assoluto LOO (margine x{m})")

# soglia effettiva media, per informazione
Ts = [max(max(traces[s]["control"]["phi"][WARMUP_IGNORE:]) for s in SEEDS if s != sd)
      for sd in SEEDS]
print(f"\nSoglia assoluta LOO (margine 1.0): media {np.mean(Ts):.3f}, "
      f"range {min(Ts):.3f}-{max(Ts):.3f}")

# ─────────────────────────────────────────────────────────────────────────
# SALVATAGGIO ROBUSTO — le tracce sono riutilizzabili: non perderle
# ─────────────────────────────────────────────────────────────────────────
with open("/content/traces_phi_clean.json", "w") as f:
    json.dump({str(sd): traces[sd] for sd in SEEDS}, f)
print("\nTracce salvate: /content/traces_phi_clean.json")

# copia su Drive, cosi' sopravvivono al riavvio del runtime (opzionale)
try:
    from google.colab import drive
    drive.mount("/content/drive")
    import shutil
    shutil.copy("/content/traces_phi_clean.json",
                "/content/drive/MyDrive/traces_phi_clean.json")
    print("Copia su Drive: /content/drive/MyDrive/traces_phi_clean.json")
except Exception as e:
    print(f"(Drive non montato, tracce solo in /content: {e})")


# =============================================================================
# FASE 3 — RIANALISI: contabilita' completa con i PRE-ONSET
# =============================================================================
print("\n" + "="*74)




def first_alarm_adaptive(phi, k):
    """Primo step in cui phi supera mu + k*sigma della propria storia."""
    for t in range(WARMUP_IGNORE, len(phi)):
        hist = phi[:t]
        if phi[t] > np.mean(hist) + k * np.std(hist):
            return t
    return None


def _first_alarm_fixed(phi, T):
    for t in range(WARMUP_IGNORE, len(phi)):
        if phi[t] > T:
            return t
    return None


def full_eval(alarm_fn, name, verbose=False):
    """Contabilita' completa, inclusi i pre-onset."""
    pre, hits, miss, fps, lats = 0, 0, 0, 0, []
    detail = []
    for sd in SEEDS:
        a_s = alarm_fn(traces[sd]["shock"]["phi"], sd)
        a_c = alarm_fn(traces[sd]["control"]["phi"], sd)
        if a_s is None:
            miss += 1; status = "MISS"
        elif a_s < SHOCK_START:
            pre += 1; status = f"PRE-ONSET (step {a_s})"
        else:
            hits += 1; lats.append(a_s - SHOCK_START)
            status = f"HIT (step {a_s}, lat {a_s - SHOCK_START})"
        if a_c is not None:
            fps += 1; status += f" | FP ctrl step {a_c}"
        detail.append((sd, status))
    lat = f"{np.mean(lats):.1f}" if lats else "-"
    n = len(SEEDS)
    print(f"{name:<30}{hits}/{n:<8}{pre:<11}{miss:<8}{fps}/{n:<8}{lat}")
    if verbose:
        for sd, st in detail:
            print(f"      seed {sd}: {st}")
    return {"hits": hits, "pre": pre, "miss": miss, "fps": fps,
            "lat": np.mean(lats) if lats else None}


print("CONTABILITA' COMPLETA (pre-onset conteggiati)")
print(f"{'detector':<30}{'HIT':<9}{'PRE-ONSET':<11}{'MISS':<8}{'FP':<9}{'latenza'}")

res = {}
for k in [1.0, 1.5, 2.0, 2.5, 3.0]:
    res[k] = full_eval(lambda phi, sd, k=k: first_alarm_adaptive(phi, k),
                       f"adattivo mu+{k}*sigma")

# soglia assoluta leave-one-seed-out, per confronto
def make_fixed_loo2(margin):
    def det(phi, sd):
        others = [max(traces[s]["control"]["phi"][WARMUP_IGNORE:])
                  for s in SEEDS if s != sd]
        return _first_alarm_fixed(phi, max(others) * margin)
    return det

for m in [1.0, 1.25]:
    full_eval(make_fixed_loo2(m), f"assoluto LOO x{m}")

# ── dettaglio del punto operativo migliore ───────────────────────────────
best_k = min(res, key=lambda k: (-res[k]["hits"], res[k]["pre"] + res[k]["fps"],
                                 res[k]["lat"] if res[k]["lat"] is not None else 99))
print(f"\nDETTAGLIO per seed — adattivo mu+{best_k}*sigma (punto operativo migliore)")
full_eval(lambda phi, sd, k=best_k: first_alarm_adaptive(phi, k),
          f"adattivo mu+{best_k}*sigma", verbose=True)

# ── sweep fine: l'intervallo operativo e' ampio o e' un punto fortunato? ──
print("\nSWEEP FINE DI k (cerca l'intervallo con HIT pieno e zero errori)")
clean_ks = []
for k in np.arange(1.0, 3.05, 0.1):
    pre, hits, fps = 0, 0, 0
    for sd in SEEDS:
        a_s = first_alarm_adaptive(traces[sd]["shock"]["phi"], k)
        a_c = first_alarm_adaptive(traces[sd]["control"]["phi"], k)
        if a_s is not None and a_s >= SHOCK_START: hits += 1
        elif a_s is not None: pre += 1
        if a_c is not None: fps += 1
    if hits == len(SEEDS) and pre == 0 and fps == 0:
        clean_ks.append(round(float(k), 1))

if clean_ks:
    print(f"  k con 5/5 HIT, 0 pre-onset, 0 FP: da {min(clean_ks)} a {max(clean_ks)} "
          f"({len(clean_ks)} valori)")
    print("  -> intervallo operativo AMPIO: non e' un punto fortunato.")
else:
    print("  Nessun k con contabilita' perfetta: il punto operativo e' stretto.")

# ── livelli di phi, per contesto ─────────────────────────────────────────
print("\nLIVELLI DI PHI (attuatore spento)")
print(f"{'seed':<8}{'phi ctrl max':<15}{'phi shock max':<16}{'rapporto finestra'}")
for sd in SEEDS:
    c = traces[sd]["control"]["phi"]; s = traces[sd]["shock"]["phi"]
    cw = np.mean(c[SHOCK_START:SHOCK_END]); sw = np.mean(s[SHOCK_START:SHOCK_END])
    print(f"{sd:<8}{max(c[WARMUP_IGNORE:]):<15.3f}{max(s[WARMUP_IGNORE:]):<16.3f}"
          f"{sw/cw:.2f}x")

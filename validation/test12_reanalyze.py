"""
=============================================================================
RIANALISI OFFLINE — nessun run, solo le tracce gia' salvate
=============================================================================
Corregge un difetto della valutazione precedente: un allarme che scatta PRIMA
dell'onset dello shock (step 80) nel braccio shockato non era contato ne' come
successo ne' come falso positivo. Spariva. Qui viene contato esplicitamente.

Contabilita' completa per ogni detector:
  - PRE-ONSET : allarme nel braccio shock prima dello step 80  (falso allarme)
  - HIT       : primo allarme nel braccio shock a partire dallo step 80
  - MISS      : nessun allarme nel braccio shock
  - FP        : allarme nel braccio di controllo (nessuno shock esiste)
  - LATENZA   : (step del primo allarme valido) - 80

Piu' uno sweep fine di k, per vedere se esiste un intervallo operativo ampio
o se 1.5 e' un punto fortunato.

USO: mettere questo file in Colab dopo aver girato la raccolta tracce.
     Legge /content/traces_phi_clean.json
=============================================================================
"""
import argparse
import json
from pathlib import Path
import numpy as np

parser = argparse.ArgumentParser(description="Historical EMA detector accounting, not FSM replay")
parser.add_argument("traces", nargs="?", type=Path,
                    default=Path(__file__).with_name("traces_phi_clean.json"))
TRACES_PATH = parser.parse_args().traces
SHOCK_START, SHOCK_END = 80, 120
WARMUP_IGNORE = 30

with open(TRACES_PATH) as f:
    raw = json.load(f)
traces = {int(k): v for k, v in raw.items()}
SEEDS = sorted(traces.keys())
print(f"Tracce caricate: {len(SEEDS)} semi {SEEDS}\n")


def first_alarm_adaptive(phi, k):
    """Primo step in cui phi supera mu + k*sigma della propria storia."""
    for t in range(WARMUP_IGNORE, len(phi)):
        hist = phi[:t]
        if phi[t] > np.mean(hist) + k * np.std(hist):
            return t
    return None


def first_alarm_fixed(phi, T):
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


print("CONTABILITA' COMPLETA: detector offline su phi EMA; NON replay del controller")
print(f"{'detector':<30}{'HIT':<9}{'PRE-ONSET':<11}{'MISS':<8}{'FP':<9}{'latenza'}")

res = {}
for k in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
    res[k] = full_eval(lambda phi, sd, k=k: first_alarm_adaptive(phi, k),
                       f"adattivo mu+{k}*sigma")

# soglia assoluta leave-one-seed-out, per confronto
def make_fixed_loo(margin):
    def det(phi, sd):
        others = [max(traces[s]["control"]["phi"][WARMUP_IGNORE:])
                  for s in SEEDS if s != sd]
        return first_alarm_fixed(phi, max(others) * margin)
    return det

for m in [1.0, 1.25]:
    full_eval(make_fixed_loo(m), f"assoluto LOO x{m}")

# ── dettaglio del punto operativo migliore ───────────────────────────────
best_k = min(res, key=lambda k: (-res[k]["hits"], res[k]["pre"] + res[k]["fps"],
                                 res[k]["lat"] if res[k]["lat"] is not None else 99))
print(f"\nDETTAGLIO per seed — adattivo mu+{best_k}*sigma (punto operativo migliore)")
full_eval(lambda phi, sd, k=best_k: first_alarm_adaptive(phi, k),
          f"adattivo mu+{best_k}*sigma", verbose=True)

# ── sweep fine: l'intervallo operativo e' ampio o e' un punto fortunato? ──
print("\nSWEEP FINE DI k (cerca l'intervallo con HIT pieno e zero errori)")
clean_ks = []
for k in np.arange(1.0, 6.05, 0.1):
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

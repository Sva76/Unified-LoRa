"""
=============================================================================
TEST 7 — RIANALISI del log esistente (phi_lead_time_log.json)
=============================================================================
Riapplica i rilevatori CORRETTI (fase I / fase II) al log già prodotto
dal run v1. Nessun nuovo training, zero crediti Tinker.

Uso:  python3 test7_reanalyze.py [percorso/phi_lead_time_log.json]
=============================================================================
"""
import json, sys
import numpy as np

PATH = sys.argv[1] if len(sys.argv) > 1 else "phi_lead_time_log.json"

# --- stesse soglie pre-registrate del Test 7 ---
BASELINE_WIN     = (20, 60)
DETECT_FROM      = BASELINE_WIN[1]   # fase II: monitoraggio da step 60
COLLAPSE_MULT    = 3.0
COLLAPSE_SUSTAIN = 5
PHI_THRESH       = 0.10
PHI_SUSTAIN      = 3
NAIVE_MULT       = 1.5
NAIVE_SUSTAIN    = 3
LEAD_USEFUL      = 10

def _first_sustained(flags, sustain):
    count = 0
    for i, f in enumerate(flags):
        count = count + 1 if f else 0
        if count >= sustain:
            return i - sustain + 1
    return None

def detect_collapse(log):
    base = [l for (s, l, _, _) in log
            if BASELINE_WIN[0] <= s < BASELINE_WIN[1] and l is not None and np.isfinite(l)]
    if not base:
        return None, None
    med = float(np.median(base))
    thresh = COLLAPSE_MULT * med
    flags = [s >= DETECT_FROM and (l is None or not np.isfinite(l) or l >= thresh)
             for (s, l, _, _) in log]
    first_nan = next((s for (s, l, _, _) in log
                      if s >= DETECT_FROM and (l is None or not np.isfinite(l))), None)
    t_sust = _first_sustained(flags, COLLAPSE_SUSTAIN)
    candidates = [t for t in (t_sust, first_nan) if t is not None]
    return (min(candidates) if candidates else None), med

def detect_phi_alarm(log):
    flags = [s >= DETECT_FROM and p >= PHI_THRESH for (s, _, p, _) in log]
    return _first_sustained(flags, PHI_SUSTAIN)

def detect_naive_alarm(log, med):
    thresh = NAIVE_MULT * med
    flags = [s >= DETECT_FROM and l is not None and np.isfinite(l) and l > thresh
             for (s, l, _, _) in log]
    return _first_sustained(flags, NAIVE_SUSTAIN)

def norm(log):
    """json ha liste, non tuple; e i NaN possono essere null."""
    out = []
    for row in log:
        s, l, p, lr = row
        l = float("nan") if l is None else float(l)
        out.append((int(s), l, float(p), float(lr)))
    return out

data = json.load(open(PATH))
logs_c = {k: norm(v) for k, v in data["logs"]["collapse"].items()}
logs_h = {k: norm(v) for k, v in data["logs"]["healthy"].items()}
t_dbl = {str(r["seed"]): r["t_double"] for r in data["collapse"]}

print("RIANALISI TEST 7 — rilevatori fase I/II (monitoraggio da step "
      f"{DETECT_FROM})\n")
print("=== BRACCIO COLLASSO ===")
print(f"{'seed':<6}{'T_dbl':<7}{'t_c':<7}{'t_a(φ)':<8}{'t_a(naive)':<11}{'L_phi':<7}{'L_naive'}")
collapse_rows = []
for seed, log in sorted(logs_c.items(), key=lambda kv: int(kv[0])):
    t_c, med = detect_collapse(log)
    t_a = detect_phi_alarm(log)
    t_n = detect_naive_alarm(log, med) if med is not None else None
    L_phi   = (t_c - t_a) if (t_c is not None and t_a is not None) else None
    L_naive = (t_c - t_n) if (t_c is not None and t_n is not None) else None
    collapse_rows.append({"seed": int(seed), "t_c": t_c, "t_a_phi": t_a,
                          "t_a_naive": t_n, "lead_phi": L_phi,
                          "lead_naive": L_naive, "baseline_median": med})
    f = lambda v: "-" if v is None else str(v)
    print(f"{seed:<6}{t_dbl.get(seed,'?'):<7}{f(t_c):<7}{f(t_a):<8}"
          f"{f(t_n):<11}{f(L_phi):<7}{f(L_naive)}")

print("\n=== BRACCIO SANO (falsi allarmi) ===")
print(f"{'seed':<6}{'falso allarme φ':<18}{'step'}")
healthy_rows = []
for seed, log in sorted(logs_h.items(), key=lambda kv: int(kv[0])):
    t_a = detect_phi_alarm(log)
    healthy_rows.append({"seed": int(seed), "false_alarm_step": t_a})
    print(f"{seed:<6}{'SÌ' if t_a is not None else 'no':<18}"
          f"{t_a if t_a is not None else '-'}")

# --- verdetto pre-registrato ---
leads_phi   = [r["lead_phi"]   for r in collapse_rows if r["lead_phi"]   is not None]
leads_naive = [r["lead_naive"] for r in collapse_rows if r["lead_naive"] is not None]
n_useful    = sum(1 for L in leads_phi if L >= LEAD_USEFUL)
n_collapsed = sum(1 for r in collapse_rows if r["t_c"] is not None)
false_alarms = sum(1 for r in healthy_rows if r["false_alarm_step"] is not None)
med_phi   = float(np.median(leads_phi))   if leads_phi   else float("nan")
med_naive = float(np.median(leads_naive)) if leads_naive else float("nan")

print("\n================= VERDETTO =================")
print(f"run collassati            : {n_collapsed}/{len(collapse_rows)}")
print(f"lead time φ (mediana)     : {med_phi}")
print(f"lead time naive (mediana) : {med_naive}")
print(f"run con lead >= {LEAD_USEFUL}        : {n_useful}/{len(collapse_rows)}")
print(f"falsi allarmi (sani)      : {false_alarms}/{len(healthy_rows)}")

if n_collapsed < 6:
    print("=> ATTENZIONE: meno di 6 collassi rilevati. Rampa troppo lenta o cap"
          " troppo basso: nessun verdetto, servono run con rampa più aggressiva.")
elif (np.isfinite(med_phi) and med_phi <= 0) or false_alarms >= 3:
    print("=> FALSIFICATO: φ non anticipa il collasso (o grida al lupo).")
elif n_useful >= 6 and med_phi > 0 and false_alarms <= 1 and med_phi > med_naive:
    print("=> SUPPORTATO: lead time azionabile, pochi falsi allarmi,"
          " batte il monitor ingenuo.")
else:
    print("=> AMBIGUO: risultato parziale, da riportare cosí com'è.")

with open("phi_lead_time_reanalysis.json", "w") as f:
    json.dump({"detect_from": DETECT_FROM, "collapse": collapse_rows,
               "healthy": healthy_rows,
               "median_lead_phi": med_phi, "median_lead_naive": med_naive,
               "n_useful": n_useful, "false_alarms": false_alarms}, f, indent=2)
print("\nSalvato in phi_lead_time_reanalysis.json")

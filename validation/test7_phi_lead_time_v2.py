"""
=============================================================================
TEST 7 — LEAD TIME PREDITTIVO di φ su Tinker
=============================================================================
DOMANDA: φ supera la soglia di allarme un numero UTILE di step PRIMA del
collasso della loss, con pochi falsi allarmi, e PRIMA di un monitor ingenuo
sulla loss?

DESIGN PRE-REGISTRATO (tutte le definizioni fissate QUI, prima dei dati):

 BRACCIO COLLASSO (8 run):
   LR a rampa esponenziale: LR(t) = LR0 * 2^(t / T_double)
   T_double estratto per seed in [35, 55] -> il MOMENTO del collasso varia
   per seed e non è noto in anticipo. Cap a 400 step.

 BRACCIO SANO (6 run):
   LR costante 1e-4, 300 step. Misura i FALSI ALLARMI.

 COMPARATORE INGENUO (stesso log del braccio collasso, zero run extra):
   allarme se loss(t) > 1.5 * mediana(loss step 20-60) per 3 step consecutivi.

 FASE I / FASE II (stile carte di controllo):
   step 0-59  = fase I, stima della baseline: NESSUN rilevamento attivo
                (la discesa iniziale della loss non è un collasso).
   step >= 60 = fase II, monitoraggio: collasso e allarmi rilevati da qui.

 DEFINIZIONI (pre-fissate):
   COLLASSO t_c  : primo step con loss >= 3 * mediana(loss step 20-60)
                   sostenuto per 5 step consecutivi, oppure primo NaN.
   ALLARME φ t_a : primo step con φ >= 0.10 sostenuto per 3 step consecutivi.
                   (0.10 = la soglia "healthy/altro" GIÀ fissata nel Test 6.)
   LEAD TIME L   : t_c - t_a.

 VERDETTO (pre-fissato):
   SUPPORTATO  : L >= 10 in almeno 6/8 run collasso, mediana L > 0,
                 <= 1 falso allarme su 6 run sani,
                 e mediana(L_phi) > mediana(L_naive).
   FALSIFICATO : mediana L <= 0, oppure >= 3 falsi allarmi sui run sani.
   AMBIGUO     : tutto il resto (utile ma non superiore: si riporta cosí).

NOTA ONESTA: φ è derivato dalla loss e il collasso è definito sulla loss.
Il claim NON è "informazione indipendente", ma "allarme online con soglia
pre-fissata che dà lead time azionabile meglio di un monitor ingenuo".
È il criterio con cui si valutano gli early-warning system (carte EWMA).
=============================================================================
"""
import os, json, subprocess, sys, random

try:
    import tinker  # noqa
except ModuleNotFoundError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tinker"], check=True)
    import importlib; importlib.invalidate_caches()
    import tinker  # noqa
import numpy as np

# ---------------------------------------------------------------------------
# 0. Setup e parametri PRE-REGISTRATI
# ---------------------------------------------------------------------------
os.environ["TINKER_API_KEY"] = "MY KEY"  # <-- inserisci

BASE_MODEL   = "Qwen/Qwen3-8B"
LR0          = 1e-4          # LR iniziale (identico al regime sano)
LR_HEALTHY   = 1e-4
STEPS_COLLAPSE_CAP = 400     # cap di sicurezza sul braccio collasso
STEPS_HEALTHY      = 300

SEEDS_COLLAPSE = [3, 7, 11, 17, 23, 31, 41, 53]   # 8 run
SEEDS_HEALTHY  = [5, 13, 19, 29, 37, 43]          # 6 run

T_DOUBLE_MIN, T_DOUBLE_MAX = 35, 55  # raddoppio LR ogni T step (per seed)

# --- soglie PRE-FISSATE ---
PHI_THRESH      = 0.10   # dal Test 6 (classe "healthy" sotto 0.10)
PHI_SUSTAIN     = 3      # step consecutivi sopra soglia per l'allarme φ
COLLAPSE_MULT   = 3.0    # loss >= 3x mediana baseline
COLLAPSE_SUSTAIN = 5     # step consecutivi per dichiarare il collasso
NAIVE_MULT      = 1.5    # comparatore ingenuo: loss > 1.5x mediana baseline
NAIVE_SUSTAIN   = 3
BASELINE_WIN    = (20, 60)  # finestra per la mediana di baseline
DETECT_FROM     = BASELINE_WIN[1]  # fase I: baseline (0-60); fase II: monitoraggio (>=60)
LEAD_USEFUL     = 10     # lead time minimo "azionabile"

service = tinker.ServiceClient()

_SUBJECTS = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
             "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
_RELS = ["is near the","is far from the","is above the","is below the","is beside the"]

def build_dataset(seed):
    r = random.Random(seed)
    subs = _SUBJECTS[:]; r.shuffle(subs)
    return [(f"Complete: The {a} {r.choice(_RELS)}", r.choice(_SUBJECTS)) for a in subs]

class PhiMonitor:
    """φ = EMA dei salti di loss verso l'alto. Identico ai Test 3-6."""
    def __init__(self, beta=0.8):
        self.beta = beta; self.ema_jump = 0.0; self.prev_loss = None
    def update(self, loss, grad_norm=None):
        if not np.isfinite(loss):
            return self.ema_jump
        jump = 0.0 if self.prev_loss is None else max(0.0, loss - self.prev_loss)
        self.prev_loss = loss
        self.ema_jump = self.beta * self.ema_jump + (1 - self.beta) * jump
        g = 0.0 if grad_norm is None else grad_norm
        return self.ema_jump + 0.01 * g

def dig(obj, *names):
    for n in names:
        if isinstance(obj, dict) and n in obj: return obj[n]
        if hasattr(obj, n): return getattr(obj, n)
    return None

def extract_loss(fb_res):
    loss = dig(fb_res, "loss")
    m = dig(fb_res, "metrics") or {}
    if loss is None:
        # controllo esplicito su None (fix del Test 5: 0.0 è un valore valido)
        if "loss:sum" in m and m["loss:sum"] is not None:
            loss = m["loss:sum"]
        elif "loss" in m and m["loss"] is not None:
            loss = m["loss"]
        else:
            loss = float("nan")
    gn = m.get("grad_norm") if isinstance(m, dict) else None
    return float(loss), (float(gn) if gn is not None else None)

# ---------------------------------------------------------------------------
# 1. Un run: registra (step, loss, phi, lr). Nessuna decisione online.
#    Tutte le rilevazioni (t_c, t_a) sono calcolate DOPO, dal log, con le
#    definizioni pre-registrate — cosí il codice di allarme non può
#    influenzare il training.
# ---------------------------------------------------------------------------
def run_arm(seed, lr_fn, max_steps, tc_client, tok):
    pairs = build_dataset(seed)
    phi_mon = PhiMonitor()
    log = []  # (step, loss, phi, lr)
    for step in range(max_steps):
        lr = lr_fn(step)
        prompt, target = pairs[step % len(pairs)]
        pi = tok.encode(prompt); ti = tok.encode(" " + target)
        toks = pi + ti
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=toks),
            loss_fn_inputs=dict(weights=[0.0]*len(pi) + [1.0]*len(ti),
                                target_tokens=toks[1:] + [toks[-1]]),
        )
        fb = tc_client.forward_backward([datum], "cross_entropy")
        op = tc_client.optim_step(tinker.AdamParams(learning_rate=lr))
        fb_res = fb.result(); op.result()
        loss, gn = extract_loss(fb_res)
        phi = phi_mon.update(loss, gn)
        log.append((step, loss, phi, lr))
        # stop anticipato solo su NaN conclamato e persistente (risparmio crediti):
        # 10 NaN consecutivi = run morto, non c'è più nulla da osservare.
        if len(log) >= 10 and all(not np.isfinite(l) for (_, l, _, _) in log[-10:]):
            break
    return log

# ---------------------------------------------------------------------------
# 2. Rilevatori POST-HOC con definizioni pre-registrate
# ---------------------------------------------------------------------------
def _first_sustained(flags, sustain):
    """Primo indice in cui flags è True per `sustain` step consecutivi."""
    count = 0
    for i, f in enumerate(flags):
        count = count + 1 if f else 0
        if count >= sustain:
            return i - sustain + 1
    return None

def detect_collapse(log):
    base = [l for (s, l, _, _) in log
            if BASELINE_WIN[0] <= s < BASELINE_WIN[1] and np.isfinite(l)]
    if not base:
        return None, None
    med = float(np.median(base))
    thresh = COLLAPSE_MULT * med
    # Fase II: il monitoraggio parte DOPO la finestra di baseline (step >= DETECT_FROM).
    # La discesa iniziale della loss (step 0-20) non è un collasso.
    flags = [s >= DETECT_FROM and ((not np.isfinite(l)) or (l >= thresh))
             for (s, l, _, _) in log]
    # un NaN in fase di monitoraggio è già collasso da solo
    first_nan = next((s for (s, l, _, _) in log
                      if s >= DETECT_FROM and not np.isfinite(l)), None)
    t_sust = _first_sustained(flags, COLLAPSE_SUSTAIN)
    candidates = [t for t in (t_sust, first_nan) if t is not None]
    return (min(candidates) if candidates else None), med

def detect_phi_alarm(log):
    flags = [s >= DETECT_FROM and p >= PHI_THRESH for (s, _, p, _) in log]
    return _first_sustained(flags, PHI_SUSTAIN)

def detect_naive_alarm(log, med):
    thresh = NAIVE_MULT * med
    flags = [s >= DETECT_FROM and np.isfinite(l) and l > thresh
             for (s, l, _, _) in log]
    return _first_sustained(flags, NAIVE_SUSTAIN)

# ---------------------------------------------------------------------------
# 3. Esecuzione — braccio collasso
# ---------------------------------------------------------------------------
print(f"TEST 7 — LEAD TIME | {BASE_MODEL}")
print(f"soglia φ = {PHI_THRESH} (pre-fissata, dal Test 6) | lead utile >= {LEAD_USEFUL} step\n")

collapse_results = []
print("=== BRACCIO COLLASSO (LR a rampa) ===")
print(f"{'seed':<6}{'T_dbl':<7}{'t_c':<7}{'t_a(φ)':<8}{'t_a(naive)':<11}{'L_phi':<7}{'L_naive'}")
for seed in SEEDS_COLLAPSE:
    rng = random.Random(seed * 991)
    t_double = rng.randint(T_DOUBLE_MIN, T_DOUBLE_MAX)
    lr_fn = lambda step, td=t_double: LR0 * (2.0 ** (step / td))
    tc = service.create_lora_training_client(base_model=BASE_MODEL)
    tok = tc.get_tokenizer()
    log = run_arm(seed, lr_fn, STEPS_COLLAPSE_CAP, tc, tok)
    t_c, med = detect_collapse(log)
    t_a = detect_phi_alarm(log)
    t_n = detect_naive_alarm(log, med) if med is not None else None
    L_phi   = (t_c - t_a) if (t_c is not None and t_a is not None) else None
    L_naive = (t_c - t_n) if (t_c is not None and t_n is not None) else None
    collapse_results.append({
        "seed": seed, "t_double": t_double, "t_c": t_c, "t_a_phi": t_a,
        "t_a_naive": t_n, "lead_phi": L_phi, "lead_naive": L_naive,
        "baseline_median": med,
        "log": log,
    })
    fmt = lambda v: "-" if v is None else str(v)
    print(f"{seed:<6}{t_double:<7}{fmt(t_c):<7}{fmt(t_a):<8}{fmt(t_n):<11}"
          f"{fmt(L_phi):<7}{fmt(L_naive)}")

# ---------------------------------------------------------------------------
# 4. Esecuzione — braccio sano (falsi allarmi)
# ---------------------------------------------------------------------------
healthy_results = []
print("\n=== BRACCIO SANO (LR costante) ===")
print(f"{'seed':<6}{'falso allarme φ':<18}{'step'}")
for seed in SEEDS_HEALTHY:
    tc = service.create_lora_training_client(base_model=BASE_MODEL)
    tok = tc.get_tokenizer()
    log = run_arm(seed, lambda s: LR_HEALTHY, STEPS_HEALTHY, tc, tok)
    t_a = detect_phi_alarm(log)
    healthy_results.append({"seed": seed, "false_alarm_step": t_a, "log": log})
    print(f"{seed:<6}{'SÌ' if t_a is not None else 'no':<18}{t_a if t_a is not None else '-'}")

# ---------------------------------------------------------------------------
# 5. VERDETTO (criteri pre-registrati)
# ---------------------------------------------------------------------------
leads_phi   = [r["lead_phi"]   for r in collapse_results if r["lead_phi"]   is not None]
leads_naive = [r["lead_naive"] for r in collapse_results if r["lead_naive"] is not None]
n_useful    = sum(1 for L in leads_phi if L >= LEAD_USEFUL)
n_collapsed = sum(1 for r in collapse_results if r["t_c"] is not None)
false_alarms = sum(1 for r in healthy_results if r["false_alarm_step"] is not None)

med_phi   = float(np.median(leads_phi))   if leads_phi   else float("nan")
med_naive = float(np.median(leads_naive)) if leads_naive else float("nan")

print("\n================= VERDETTO =================")
print(f"run collassati            : {n_collapsed}/{len(SEEDS_COLLAPSE)}")
print(f"lead time φ (mediana)     : {med_phi}")
print(f"lead time naive (mediana) : {med_naive}")
print(f"run con lead >= {LEAD_USEFUL}        : {n_useful}/{len(SEEDS_COLLAPSE)}")
print(f"falsi allarmi (sani)      : {false_alarms}/{len(SEEDS_HEALTHY)}")

if n_collapsed < 6:
    print("=> ATTENZIONE: meno di 6 collassi. La rampa era troppo lenta o il cap"
          " troppo basso: aggiusta T_DOUBLE/CAP e ripeti. Nessun verdetto.")
elif (np.isfinite(med_phi) and med_phi <= 0) or false_alarms >= 3:
    print("=> FALSIFICATO: φ non anticipa il collasso (o grida al lupo)."
          " Il claim predittivo NON è supportato — va detto.")
elif n_useful >= 6 and med_phi > 0 and false_alarms <= 1 and med_phi > med_naive:
    print("=> SUPPORTATO: φ dà lead time azionabile, con pochi falsi allarmi,"
          " e batte il monitor ingenuo. Claim predittivo SUPPORTATO.")
else:
    print("=> AMBIGUO: lead time positivo ma non chiaramente utile/superiore"
          " al monitor ingenuo. Riportare il risultato parziale cosí com'è.")

# ---------------------------------------------------------------------------
# 6. Salvataggio log completi (per grafici e per Montebello)
# ---------------------------------------------------------------------------
out = {
    "preregistered": {
        "phi_thresh": PHI_THRESH, "phi_sustain": PHI_SUSTAIN,
        "collapse_mult": COLLAPSE_MULT, "collapse_sustain": COLLAPSE_SUSTAIN,
        "naive_mult": NAIVE_MULT, "naive_sustain": NAIVE_SUSTAIN,
        "baseline_window": BASELINE_WIN, "detect_from": DETECT_FROM, "lead_useful": LEAD_USEFUL,
        "lr0": LR0, "t_double_range": [T_DOUBLE_MIN, T_DOUBLE_MAX],
    },
    "collapse": [{k: v for k, v in r.items() if k != "log"} for r in collapse_results],
    "healthy":  [{k: v for k, v in r.items() if k != "log"} for r in healthy_results],
    "logs": {
        "collapse": {str(r["seed"]): r["log"] for r in collapse_results},
        "healthy":  {str(r["seed"]): r["log"] for r in healthy_results},
    },
}
with open("phi_lead_time_log.json", "w") as f:
    json.dump(out, f)
print("\nLog completo salvato in phi_lead_time_log.json")

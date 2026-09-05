# September 2026 correction: see validation/corrections_2026_09.md.
# Pre-correction scripts and logs are preserved at commit 72b4d08b7fbdcbb6d395db1460a4afd8d0d90884.
"""
=============================================================================
TEST 7 — LEAD TIME PREDITTIVO di φ su Tinker
=============================================================================
DOMANDA: φ supera la soglia di allarme un numero UTILE di step PRIMA del
collasso della loss, con pochi falsi allarmi, e PRIMA di un monitor ingenuo
sulla loss?

DISEGNO STORICO (parametri conservati come riferimento; questa e una correzione):

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
   ALLARME φ t_a : conferma al TERZO step consecutivo con φ >= 0.10.
                   (0.10 = la soglia "healthy/altro" GIÀ fissata nel Test 6.)
   LEAD TIME L   : onset retrospettivo del collasso meno conferma causale dell'allarme.

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
try:
    from .phi_utils import (PhiJumpMonitor as PhiMonitor, completion_example,
                            new_run_output, write_run_json)
except ImportError:
    from phi_utils import (PhiJumpMonitor as PhiMonitor, completion_example,
                           new_run_output, write_run_json)

# Correction of the historical task, NOT an execution of the proposed v3 stream.
PROTOCOL_VERSION = "corrected-alignment-v1"
print("Diagnostic rerun with corrected token alignment; legacy thresholds are unvalidated.")

# ---------------------------------------------------------------------------
# 0. Setup e parametri PRE-REGISTRATI
# ---------------------------------------------------------------------------
if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError("Set TINKER_API_KEY in the environment before running this experiment")

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

OUTPUT = new_run_output("test7_lead_time.json")
service = tinker.ServiceClient()

_SUBJECTS = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
             "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
_RELS = ["is near the","is far from the","is above the","is below the","is beside the"]

def build_dataset(seed):
    r = random.Random(seed)
    subs = _SUBJECTS[:]; r.shuffle(subs)
    return [(f"Complete: The {a} {r.choice(_RELS)}", r.choice(_SUBJECTS)) for a in subs]

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
    return float(loss)

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
        inputs, targets, weights = completion_example(pi, ti)
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=inputs),
            loss_fn_inputs=dict(weights=weights, target_tokens=targets),
        )
        fb = tc_client.forward_backward([datum], "cross_entropy")
        op = tc_client.optim_step(tinker.AdamParams(learning_rate=lr))
        fb_res = fb.result(); op.result()
        loss = extract_loss(fb_res)
        phi = phi_mon.update(loss)
        log.append((step, loss, phi, lr))
        # stop anticipato solo su NaN conclamato e persistente (risparmio crediti):
        # 10 NaN consecutivi = run morto, non c'è più nulla da osservare.
        if len(log) >= 10 and all(not np.isfinite(l) for (_, l, _, _) in log[-10:]):
            break
    return log

# ---------------------------------------------------------------------------
# Corrected diagnostic collection and causal timing accounting.
# The old thresholds remain visible for comparison, not as new calibration.
# ---------------------------------------------------------------------------
try:
    from .test7_metrics import summarize
except ImportError:
    from test7_metrics import summarize

logs = {"collapse": {}, "healthy": {}}
ramp_periods = {}
for seed in SEEDS_COLLAPSE:
    rng = random.Random(seed * 991)
    t_double = rng.randint(T_DOUBLE_MIN, T_DOUBLE_MAX)
    ramp_periods[str(seed)] = t_double
    tc = service.create_lora_training_client(base_model=BASE_MODEL)
    tok = tc.get_tokenizer()
    logs["collapse"][str(seed)] = run_arm(
        seed, lambda step, td=t_double: LR0 * 2.0 ** (step / td),
        STEPS_COLLAPSE_CAP, tc, tok)
    print(f"Completed ramp seed {seed}")

for seed in SEEDS_HEALTHY:
    tc = service.create_lora_training_client(base_model=BASE_MODEL)
    tok = tc.get_tokenizer()
    logs["healthy"][str(seed)] = run_arm(seed, lambda s: LR_HEALTHY,
                                           STEPS_HEALTHY, tc, tok)
    print(f"Completed healthy seed {seed}")

out = {
    "protocol_version": PROTOCOL_VERSION,
    "model": BASE_MODEL,
    "confirmatory": False,
    "legacy_reference_settings": {
        "phi_thresh": PHI_THRESH, "phi_sustain": PHI_SUSTAIN,
        "collapse_mult": COLLAPSE_MULT, "collapse_sustain": COLLAPSE_SUSTAIN,
        "naive_mult": NAIVE_MULT, "naive_sustain": NAIVE_SUSTAIN,
        "baseline_window": BASELINE_WIN, "detect_from": DETECT_FROM,
        "lr0": LR0, "t_double_by_seed": ramp_periods,
    },
    "logs": logs,
}
write_run_json(OUTPUT, out)
report = summarize(out)
write_run_json(OUTPUT.with_name("test7_timing_report.json"), report)
print(json.dumps(report, indent=2))
print(f"Diagnostic results saved to {OUTPUT}")

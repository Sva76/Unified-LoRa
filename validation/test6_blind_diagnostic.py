# September 2026 correction: see validation/corrections_2026_09.md.
# Pre-correction scripts and logs are preserved at commit 72b4d08b7fbdcbb6d395db1460a4afd8d0d90884.
"""
=============================================================================
TEST: φ è un DIAGNOSTICO OPERATIVO? (classificazione in cieco)
=============================================================================
DOMANDA: guardando SOLO φ (senza conoscere il learning rate usato), si può
classificare correttamente il regime di stress di un run?

DISEGNO a prova di revisore:
  1. 12 run, ognuno con LR scelto a caso tra:
        sano       = 1e-4
        intermedio = 5e-4  e  1e-3
        stressato  = 3e-3
     L'etichetta vera viene NASCOSTA durante la classificazione.
  2. Le SOGLIE di decisione sono FISSATE PRIMA, dai dati che già abbiamo
     (run precedenti: φ_sano ~0.011, φ_stressato ~4):
        φ_mean < 0.10           -> "healthy"
        0.10 <= φ_mean < 1.0    -> "intermediate"
        φ_mean >= 1.0           -> "stressed"
     (la mappa LR->classe e le soglie sono decise QUI, non dopo aver visto i risultati)
  3. Si classifica ogni run con queste soglie, POI si rivela l'etichetta vera.
  4. Si misura l'accuratezza. Il test passa se l'accuratezza è alta
     (es. >= 10/12) -> φ è un diagnostico operativo, non solo descrittivo.

NOTA ONESTA: "intermedio" è la classe più difficile; è lì che ci aspettiamo
gli errori. Riportiamo la confusion matrix completa, non solo l'accuratezza.
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

if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError("Set TINKER_API_KEY in the environment before running this experiment")
BASE_MODEL = "Qwen/Qwen3-8B"
STEPS = 200
N_RUNS = 12
MASTER_SEED = 2026

# --- mappa LR -> classe vera (DECISA PRIMA) ---
LR_TO_CLASS = {1e-4: "healthy", 5e-4: "intermediate", 1e-3: "intermediate", 3e-3: "stressed"}
LR_CHOICES = list(LR_TO_CLASS.keys())

# --- soglie di decisione su φ_mean (FISSATE PRIMA, dai run precedenti) ---
def classify_phi(phi_mean):
    if phi_mean < 0.10:
        return "healthy"
    elif phi_mean < 1.0:
        return "intermediate"
    else:
        return "stressed"

OUTPUT = new_run_output("test6_blind_diagnostic.json")
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


def run_once(seed, lr, tok, tc):
    pairs = build_dataset(seed)
    phi_mon = PhiMonitor(); phis = []; log = []
    for step in range(STEPS):
        prompt, target = pairs[step % len(pairs)]
        pi = tok.encode(prompt); ti = tok.encode(" " + target)
        inputs, targets, weights = completion_example(pi, ti)
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=inputs),
            loss_fn_inputs=dict(weights=weights, target_tokens=targets),
        )
        fb = tc.forward_backward([datum], "cross_entropy")
        opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
        fb_res = fb.result(); opt.result()
        loss = dig(fb_res, "loss"); m = dig(fb_res, "metrics") or {}
        if loss is None:
            loss = m["loss:sum"] if ("loss:sum" in m and m["loss:sum"] is not None) else float("nan")
        phi = float(phi_mon.update(float(loss)))
        phis.append(phi)
        log.append([step, float(loss), phi, lr])
    return float(np.mean(phis)), log


# ---------------------------------------------------------------------------
# 1. Genera i run con LR casuale (etichetta nascosta)
# ---------------------------------------------------------------------------
rng = random.Random(MASTER_SEED)
plan = []  # (run_id, seed, lr_VERO)
for i in range(N_RUNS):
    lr = rng.choice(LR_CHOICES)
    seed = rng.randint(1, 9999)
    plan.append((i, seed, lr))

# Eseguo e salvo SOLO φ + run_id (le etichette restano "in busta chiusa")
phi_only = {}   # run_id -> phi_mean
logs = {}
for run_id, seed, lr in plan:
    tc = service.create_lora_training_client(base_model=BASE_MODEL)
    tok = tc.get_tokenizer()
    phi_mean, logs[run_id] = run_once(seed, lr, tok, tc)
    phi_only[run_id] = phi_mean
    print(f"run {run_id:2d} | φ_mean = {phi_mean:.4f}   (etichetta nascosta)")

# ---------------------------------------------------------------------------
# 2. CLASSIFICAZIONE IN CIECO (solo φ, soglie pre-fissate)
# ---------------------------------------------------------------------------
invalid_counts = {rid: sum(not np.isfinite(row[1]) for row in log)
                  for rid, log in logs.items()}
predictions = {rid: "invalid" if invalid_counts[rid] else classify_phi(phi_only[rid])
               for rid in phi_only}

# ---------------------------------------------------------------------------
# 3. Rivelo le etichette vere e confronto
# ---------------------------------------------------------------------------
print("\n========== RIVELAZIONE ETICHETTE ==========")
print(f"{'run':<5}{'φ_mean':<10}{'predetto':<14}{'vero':<14}{'ok'}")
correct = 0
classes = ["healthy", "intermediate", "stressed"]
prediction_classes = classes + ["invalid"]
confusion = {t: {p: 0 for p in prediction_classes} for t in classes}
for run_id, seed, lr in plan:
    true_cls = LR_TO_CLASS[lr]
    pred = predictions[run_id]
    ok = (pred == true_cls)
    correct += ok
    confusion[true_cls][pred] += 1
    print(f"{run_id:<5}{phi_only[run_id]:<10.4f}{pred:<14}{true_cls:<14}{'✓' if ok else '✗'}")

acc = correct / N_RUNS
print(f"\nAccuratezza: {correct}/{N_RUNS} = {acc*100:.0f}%")
print("\nConfusion matrix (righe=vero, colonne=predetto):")
print(f"{'':<14}" + "".join(f"{c:<14}" for c in prediction_classes))
for t in classes:
    print(f"{t:<14}" + "".join(f"{confusion[t][p]:<14}" for p in prediction_classes))

print("\n================= VERDETTO =================")
if any(invalid_counts.values()):
    print("=> Telemetria non finita presente: run invalidi conteggiati come insuccessi; nessun verdetto diagnostico.")
elif acc >= 10/12:
    print("=> φ classifica il regime di stress in cieco: accordo descrittivo con classi LR; non validazione operativa.")
elif acc >= 0.6:
    print("=> φ classifica parzialmente: utile ma con confusione (vedi matrice).")
else:
    print("=> φ NON classifica affidabilmente: claim diagnostico NON supportato.")

write_run_json(OUTPUT, {"protocol_version": PROTOCOL_VERSION,
                        "confirmatory": False, "plan": plan, "logs": logs,
                        "phi": phi_only, "predictions": predictions,
                        "accuracy": acc, "confusion": confusion,
                        "nonfinite_loss_count": invalid_counts})
print(f"Diagnostic results saved to {OUTPUT}")

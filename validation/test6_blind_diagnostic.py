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

os.environ["TINKER_API_KEY"] = "MY KEY"   # <-- inserisci
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

service = tinker.ServiceClient()

_SUBJECTS = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
             "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
_RELS = ["is near the","is far from the","is above the","is below the","is beside the"]


def build_dataset(seed):
    r = random.Random(seed)
    subs = _SUBJECTS[:]; r.shuffle(subs)
    return [(f"Complete: The {a} {r.choice(_RELS)}", r.choice(_SUBJECTS)) for a in subs]


class PhiMonitor:
    def __init__(self, beta=0.8):
        self.beta = beta; self.ema_jump = 0.0; self.prev_loss = None
    def update(self, loss, grad_norm=None):
        if not np.isfinite(loss): return self.ema_jump
        jump = 0.0 if self.prev_loss is None else max(0.0, loss - self.prev_loss)
        self.prev_loss = loss
        self.ema_jump = self.beta*self.ema_jump + (1-self.beta)*jump
        g = 0.0 if grad_norm is None else grad_norm
        return self.ema_jump + 0.01*g


def dig(obj, *names):
    for n in names:
        if isinstance(obj, dict) and n in obj: return obj[n]
        if hasattr(obj, n): return getattr(obj, n)
    return None


def run_once(seed, lr, tok, tc):
    pairs = build_dataset(seed)
    phi_mon = PhiMonitor(); phis = []
    for step in range(STEPS):
        prompt, target = pairs[step % len(pairs)]
        pi = tok.encode(prompt); ti = tok.encode(" " + target)
        toks = pi + ti
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=toks),
            loss_fn_inputs=dict(weights=[0.0]*len(pi)+[1.0]*len(ti),
                                target_tokens=toks[1:]+[toks[-1]]),
        )
        fb = tc.forward_backward([datum], "cross_entropy")
        opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
        fb_res = fb.result(); opt.result()
        loss = dig(fb_res, "loss"); m = dig(fb_res, "metrics") or {}
        if loss is None:
            loss = m["loss:sum"] if ("loss:sum" in m and m["loss:sum"] is not None) else float("nan")
        phis.append(phi_mon.update(float(loss),
                    float(m["grad_norm"]) if isinstance(m, dict) and m.get("grad_norm") is not None else None))
    return float(np.mean(phis))


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
for run_id, seed, lr in plan:
    tc = service.create_lora_training_client(base_model=BASE_MODEL)
    tok = tc.get_tokenizer()
    phi_mean = run_once(seed, lr, tok, tc)
    phi_only[run_id] = phi_mean
    print(f"run {run_id:2d} | φ_mean = {phi_mean:.4f}   (etichetta nascosta)")

# ---------------------------------------------------------------------------
# 2. CLASSIFICAZIONE IN CIECO (solo φ, soglie pre-fissate)
# ---------------------------------------------------------------------------
predictions = {rid: classify_phi(phi_only[rid]) for rid in phi_only}

# ---------------------------------------------------------------------------
# 3. Rivelo le etichette vere e confronto
# ---------------------------------------------------------------------------
print("\n========== RIVELAZIONE ETICHETTE ==========")
print(f"{'run':<5}{'φ_mean':<10}{'predetto':<14}{'vero':<14}{'ok'}")
correct = 0
classes = ["healthy", "intermediate", "stressed"]
confusion = {t: {p: 0 for p in classes} for t in classes}
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
print(f"{'':<14}" + "".join(f"{c:<14}" for c in classes))
for t in classes:
    print(f"{t:<14}" + "".join(f"{confusion[t][p]:<14}" for p in classes))

print("\n================= VERDETTO =================")
if acc >= 10/12:
    print("=> φ classifica il regime di stress in cieco: DIAGNOSTICO OPERATIVO supportato.")
elif acc >= 0.6:
    print("=> φ classifica parzialmente: utile ma con confusione (vedi matrice).")
else:
    print("=> φ NON classifica affidabilmente: claim diagnostico NON supportato.")

with open("phi_blind_diagnostic.json", "w") as f:
    json.dump({"plan": [(r, s, l) for r, s, l in plan],
               "phi": phi_only, "predictions": predictions,
               "accuracy": acc, "confusion": confusion}, f, indent=2)
print("\nLog salvato in phi_blind_diagnostic.json")

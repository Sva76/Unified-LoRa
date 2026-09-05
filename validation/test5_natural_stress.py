# September 2026 correction: see validation/corrections_2026_09.md.
# Pre-correction scripts and logs are preserved at commit 72b4d08b7fbdcbb6d395db1460a4afd8d0d90884.
"""
=============================================================================
TEST STRESS NATURALE su Tinker — φ rileva instabilità NON indotta?
=============================================================================
DIFFERENZA CHIAVE rispetto ai test precedenti:
  - PRIMA: shock artificiale (target corrotti) -> salto netto di loss da cui
    φ "bara" facilmente. Modalità facile.
  - ORA: nessuna corruzione. Lo stress nasce dalla DINAMICA del training:
    un learning rate troppo alto rende il training instabile in modo
    GRADUALE, senza un salto improvviso e annunciato.

DISEGNO (confronto appaiato, stesso task, stessi dati):
  - Braccio SANO:     LR moderato  -> training stabile, φ dovrebbe restare basso
  - Braccio STRESSATO: LR aggressivo -> training instabile, φ dovrebbe salire
  e lo stress NON è confinato a una finestra: è presente per tutto il run.

IL TEST PASSA se:
  φ_medio(stressato) >> φ_medio(sano)   su più semi.
SE φ non distingue i due regimi -> il claim "sentinella di stress naturale"
NON è supportato, e va detto.

Multi-seed (3 semi) per non dipendere dalla fortuna.
=============================================================================
"""
import os, json, subprocess, sys

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
# Setup
# ---------------------------------------------------------------------------
if not os.environ.get("TINKER_API_KEY"):
    raise RuntimeError("Set TINKER_API_KEY in the environment before running this experiment")
BASE_MODEL = "Qwen/Qwen3-8B"
STEPS = 200
SEEDS = [3, 7, 11]

LR_HEALTHY = 1e-4      # regime sano
LR_STRESS  = 3e-3      # regime aggressivo (~30x): instabilità graduale attesa
# NB: se a 3e-3 il training diverge a NaN subito, alza un po' LR_HEALTHY o
#     abbassa LR_STRESS a 1e-3; vogliamo instabilità, non morte istantanea.

OUTPUT = new_run_output("test5_natural_stress.json")
service = tinker.ServiceClient()

_SUBJECTS = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
             "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
_RELS = ["is near the","is far from the","is above the","is below the","is beside the"]


def build_dataset(seed):
    import random
    r = random.Random(seed)
    subs = _SUBJECTS[:]; r.shuffle(subs)
    return [(f"Complete: The {a} {r.choice(_RELS)}", r.choice(_SUBJECTS)) for a in subs]


def dig(obj, *names):
    for n in names:
        if isinstance(obj, dict) and n in obj: return obj[n]
        if hasattr(obj, n): return getattr(obj, n)
    return None


def run_arm(seed, lr, tok, tc):
    """Un braccio: training a un dato LR, nessuno shock indotto."""
    pairs = build_dataset(seed)
    phi_mon = PhiMonitor()
    phis, losses, nan_count = [], [], 0
    log = []
    for step in range(STEPS):
        prompt, target = pairs[step % len(pairs)]
        prompt_ids = tok.encode(prompt)
        target_ids = tok.encode(" " + target)
        input_tokens, target_tokens, weights = completion_example(prompt_ids, target_ids)
        datum = tinker.types.Datum(
            model_input=tinker.types.ModelInput.from_ints(tokens=input_tokens),
            loss_fn_inputs=dict(weights=weights, target_tokens=target_tokens),
        )
        fb = tc.forward_backward([datum], "cross_entropy")
        opt = tc.optim_step(tinker.AdamParams(learning_rate=lr))
        fb_res = fb.result(); opt.result()
        loss = dig(fb_res, "loss")
        m = dig(fb_res, "metrics") or {}
        if loss is None:
            # FIX: controllo esplicito su None. Prima un "or" trattava
            # loss:sum == 0.0 come falso -> falsi NaN nel braccio sano.
            if "loss:sum" in m and m["loss:sum"] is not None:
                loss = m["loss:sum"]
            elif "loss" in m and m["loss"] is not None:
                loss = m["loss"]
            else:
                loss = float("nan")
        loss = float(loss)
        if not np.isfinite(loss): nan_count += 1
        phi = phi_mon.update(loss)
        phis.append(phi); losses.append(loss)
        log.append([step, loss, float(phi), lr])
    phis = np.array(phis)
    finite_losses = np.array([l for l in losses if np.isfinite(l)])
    return {
        "phi_mean": float(phis.mean()),
        "phi_max": float(phis.max()),
        "loss_std": float(finite_losses.std()) if len(finite_losses) else float("nan"),
        "nan_count": nan_count,
        "log": log,
    }


# ---------------------------------------------------------------------------
# Esecuzione: per ogni seed, due bracci (sano vs stressato)
# ---------------------------------------------------------------------------
print(f"STRESS NATURALE | {BASE_MODEL} | {len(SEEDS)} semi | {STEPS} step")
print(f"LR sano = {LR_HEALTHY}  |  LR stress = {LR_STRESS}\n")
print(f"{'seed':<6}{'arm':<10}{'φ mean':<10}{'φ max':<10}{'loss std':<10}{'nan'}")

rows = []
for seed in SEEDS:
    for arm, lr in [("healthy", LR_HEALTHY), ("stressed", LR_STRESS)]:
        tc = service.create_lora_training_client(base_model=BASE_MODEL)
        tok = tc.get_tokenizer()
        r = run_arm(seed, lr, tok, tc)
        r.update(seed=seed, arm=arm)
        rows.append(r)
        print(f"{seed:<6}{arm:<10}{r['phi_mean']:<10.4f}{r['phi_max']:<10.4f}"
              f"{r['loss_std']:<10.4f}{r['nan_count']}")

# ---------------------------------------------------------------------------
# Verdetto: φ del braccio stressato è sistematicamente più alto del sano?
# ---------------------------------------------------------------------------
healthy = np.array([r["phi_mean"] for r in rows if r["arm"] == "healthy"])
stressed = np.array([r["phi_mean"] for r in rows if r["arm"] == "stressed"])
ratios = stressed / (healthy + 1e-9)

print("\n================= VERDETTO STRESS NATURALE =================")
print(f"φ medio SANO      : {healthy.mean():.4f}")
print(f"φ medio STRESSATO : {stressed.mean():.4f}")
print(f"rapporto stress/sano per seed : " + ", ".join(f"{x:.2f}x" for x in ratios))
print(f"rapporto medio    : {ratios.mean():.2f}x  ± {ratios.std():.2f}")
if any(r["nan_count"] for r in rows):
    print("=> Telemetria non finita presente: confronto incompleto, nessun verdetto di separazione.")
elif ratios.min() > 1.5:
    print("=> φ distingue lo stress NATURALE in tutti i semi: separazione descrittiva; conferma scientifica non valutata.")
elif ratios.mean() > 1.5:
    print("=> φ distingue in media ma con varianza: risultato parziale, da dire.")
else:
    print("=> φ NON distingue lo stress naturale: claim 'sentinella' NON supportato.")

write_run_json(OUTPUT, {"protocol_version": PROTOCOL_VERSION,
                        "confirmatory": False, "rows": rows})
print(f"Diagnostic results saved to {OUTPUT}")

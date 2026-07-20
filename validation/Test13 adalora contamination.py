"""
=============================================================================
IL PRINCIPIO VALE ANCHE PER ADALORA?
=============================================================================
SCOPERTA DA GENERALIZZARE:
  In Unified-LoRA il controller contamina il proprio sensore: le sue azioni
  (cambio di rank e LR) perturbano la loss, e phi -- che legge la loss --
  registra quel rumore come se fosse stress. Separando i due, lo stesso
  segnale passa da 2/5 a 5/5 rilevamenti con latenza 0-2 step.

IPOTESI: e' un difetto di CATEGORIA dei metodi adattivi, non un bug del tuo.
  AdaLoRA alloca il rank in base a punteggi di importanza calcolati sui
  gradienti -- e le sue riallocazioni perturbano proprio quei gradienti.
  Stessa struttura d'anello.

DISEGNO (identico in spirito al test su phi):
  Braccio A - ATTUATORE ACCESO : AdaLoRA normale
      model.base_model.update_and_allocate(step)   -> punteggi E riallocazione
  Braccio B - ATTUATORE SPENTO : rank congelato
      rankallocator.update_ipt(...)                -> SOLO punteggi

  In entrambi si registra il segnale di importanza. Se l'ipotesi e' vera,
  in A il segnale sara' piu' rumoroso e discriminera' PEGGIO lo shock.

SEGNALE MISURATO (per step):
  S_ema = media di exp_avg_ipt su tutti i parametri LoRA tracciati
  S_raw = media di ipt (sensibilita' grezza, non mediata)

METRICHE:
  1. VOLATILITA' nei run SENZA shock: media |S(t)-S(t-1)|.
     Se A >> B, l'attuatore inietta rumore nel proprio sensore. <- contaminazione
  2. DISCRIMINAZIONE: S(finestra shock) / S(stessa finestra, controllo).
     Se B > A, la contaminazione degrada la capacita' di rilevare.

3 semi x 2 bracci x {shock, controllo} = 12 run, ~20-25 min su T4.
=============================================================================
"""
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "peft", "accelerate"], check=True)

import torch, json, random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AdaLoraConfig, get_peft_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS = 200
SHOCK_START, SHOCK_END = 80, 120
SEEDS = [11, 23, 37]
LR = 3e-4
MAXLEN = 64
INIT_R, TARGET_R = 12, 8

SUBJ = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
        "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
RELS = ["is near the","is far from the","is above the","is below the","is beside the"]
CORRUPT = ["zxqw","7391","blorp","asdf","9920","qqzz","1k4m","wronk","vvxy","000z"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def importance_signal(rankallocator):
    """Scalare riassuntivo dei punteggi di importanza di AdaLoRA."""
    ema = [v.mean().item() for v in rankallocator.exp_avg_ipt.values()] \
        if rankallocator.exp_avg_ipt else [0.0]
    raw = [v.mean().item() for v in rankallocator.ipt.values()] \
        if rankallocator.ipt else [0.0]
    return float(np.mean(ema)), float(np.mean(raw))


def run(seed, with_shock, actuator_on):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    r = random.Random(seed)
    subs = SUBJ[:]; r.shuffle(subs)
    pairs = [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    cfg = AdaLoraConfig(
        task_type="CAUSAL_LM", init_r=INIT_R, target_r=TARGET_R, lora_alpha=16,
        tinit=int(STEPS*0.1), tfinal=int(STEPS*0.2), deltaT=10,
        total_step=STEPS, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, cfg).to(DEV)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR)
    ra = model.base_model.rankallocator

    sig_ema, sig_raw, ranks = [], [], []
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
            print(f"    loss non finita a step {step}"); break
        loss.backward()

        # ORDINE CORRETTO (come da documentazione PEFT):
        #   backward -> optimizer.step() -> update_and_allocate -> zero_grad
        # Invertirlo fa si' che AdamW riporti a valori non nulli gli elementi
        # appena mascherati: l'attuatore risulta di fatto spento.
        opt.step()
        if actuator_on:
            # punteggi + RIALLOCAZIONE del rank (AdaLoRA normale)
            model.base_model.update_and_allocate(step)
        else:
            # SOLO punteggi: nessuna riallocazione -> attuatore spento
            ra.update_ipt(model.base_model.model)
        opt.zero_grad()

        e, w = importance_signal(ra)
        sig_ema.append(e); sig_raw.append(w)
        nz = sum(int((p.detach().abs() > 0).sum().item())
                 for n, p in model.named_parameters() if "lora_E" in n)
        ranks.append(nz)

    del model, base
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return {"ema": sig_ema, "raw": sig_raw, "rank": ranks}


def volatility(x):
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan")
    d = np.abs(np.diff(x[SHOCK_START:SHOCK_END]))
    return float(np.mean(d))


print(f"CONTAMINAZIONE IN ADALORA | {MODEL_ID} | {len(SEEDS)} semi\n")
data = {}
for actuator_on in [True, False]:
    tag = "A (attuatore ACCESO)" if actuator_on else "B (attuatore SPENTO)"
    print(f"--- braccio {tag} ---")
    for sd in SEEDS:
        sh = run(sd, True,  actuator_on)
        ct = run(sd, False, actuator_on)
        data[(actuator_on, sd)] = {"shock": sh, "control": ct}
        vol_c = volatility(ct["ema"])
        s_win = np.mean(sh["ema"][SHOCK_START:SHOCK_END])
        c_win = np.mean(ct["ema"][SHOCK_START:SHOCK_END])
        ratio = s_win / c_win if c_win > 0 else float("nan")
        rk = ct["rank"]
        drop = (rk[0] - min(rk)) / rk[0] * 100 if rk and rk[0] else 0.0
        print(f"  seed {sd}: volatilita'(controllo)={vol_c:.3e} | "
              f"discriminazione={ratio:.2f}x | rank {rk[0]}->{min(rk)} ({drop:.0f}%)")
    print()

# ── verifica preliminare: l'attuatore ha agito? ──────────────────────────
rk_on = [data[(True, sd)]["control"]["rank"] for sd in SEEDS]
drops = [(r[0] - min(r)) / r[0] * 100 for r in rk_on if r and r[0]]
mean_drop = float(np.mean(drops)) if drops else 0.0
print("========== VERIFICA ATTUATORE ==========")
print(f"Riduzione media del rank nel braccio A: {mean_drop:.1f}%")
if mean_drop < 3:
    print("ATTENZIONE: l'attuatore non ha agito -> il confronto NON e' valido.")
    print("Aumentare l'aggressivita' del budget (init_r=24, target_r=4).")
else:
    print("Attuatore attivo: il confronto e' valido.\n")

print("================= VERDETTO =================")
print(f"{'braccio':<26}{'volatilita ctrl':<20}{'discriminazione'}")
summary = {}
for actuator_on in [True, False]:
    vols, ratios = [], []
    for sd in SEEDS:
        d = data[(actuator_on, sd)]
        vols.append(volatility(d["control"]["ema"]))
        s_win = np.mean(d["shock"]["ema"][SHOCK_START:SHOCK_END])
        c_win = np.mean(d["control"]["ema"][SHOCK_START:SHOCK_END])
        if c_win > 0:
            ratios.append(s_win / c_win)
    summary[actuator_on] = (np.mean(vols), np.mean(ratios) if ratios else float("nan"))
    tag = "A attuatore ACCESO" if actuator_on else "B attuatore SPENTO"
    print(f"{tag:<26}{summary[actuator_on][0]:<20.3e}{summary[actuator_on][1]:.2f}x")

vol_on, disc_on = summary[True]
vol_off, disc_off = summary[False]
print("\nLettura:")
if vol_off > 0 and vol_on / vol_off > 1.3:
    print(f" - Volatilita' {vol_on/vol_off:.2f}x MAGGIORE con attuatore acceso,")
    print("   in run SENZA shock -> CONTAMINAZIONE CONFERMATA anche in AdaLoRA.")
elif vol_off > 0 and vol_on / vol_off < 0.8:
    print(" - Volatilita' MINORE con attuatore acceso: risultato inatteso,")
    print("   la riallocazione sembra stabilizzare il segnale. Da investigare.")
else:
    print(" - Volatilita' comparabile: nessuna contaminazione evidente in AdaLoRA.")
    print("   Il problema sarebbe allora specifico dell'anello di Unified-LoRA.")

if np.isfinite(disc_on) and np.isfinite(disc_off):
    if disc_off > disc_on * 1.2:
        print(f" - Discriminazione migliore ad attuatore spento "
              f"({disc_off:.2f}x vs {disc_on:.2f}x): la riallocazione degrada")
        print("   la capacita' del segnale di distinguere lo stress.")
    else:
        print(f" - Discriminazione simile ({disc_off:.2f}x vs {disc_on:.2f}x).")

with open("/content/adalora_contamination.json", "w") as f:
    json.dump({f"{'on' if a else 'off'}_{sd}": v for (a, sd), v in data.items()}, f)
print("\nLog: /content/adalora_contamination.json")

"""
=============================================================================
VERIFICA: l'attuatore di AdaLoRA ha DAVVERO agito nel braccio A?
=============================================================================
Il test precedente non ha trovato contaminazione in AdaLoRA. Prima di
concludere, va escluso il sospetto opposto: che la riallocazione non sia
mai avvenuta (o sia stata trascurabile), rendendo il confronto A vs B vuoto.

COSA MISURA:
  AdaLoRA maschera i valori singolari azzerando elementi di lora_E.
  Il "rank effettivo" = numero di elementi NON nulli in lora_E, sommato su
  tutti i moduli adattati. Se l'attuatore agisce, questo numero DEVE calare
  nel tempo (da init_r*n_moduli verso target_r*n_moduli).

  Braccio A (update_and_allocate) -> il rank effettivo deve scendere
  Braccio B (solo update_ipt)     -> il rank effettivo deve restare costante

Se in A il rank scende in modo sostanziale, l'attuatore ha agito e il
risultato "nessuna contaminazione" e' valido.
Se in A il rank NON scende, il confronto precedente e' nullo e va rifatto.

2 run, ~4 minuti.
=============================================================================
"""
import subprocess, sys

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "transformers", "peft", "accelerate"], check=True)

import torch, random
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import AdaLoraConfig, get_peft_model

DEV = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
STEPS = 200
SEED = 11
LR = 3e-4
MAXLEN = 64
INIT_R, TARGET_R = 12, 8

SUBJ = ["cat","dog","car","tree","river","star","book","clock","stone","bird",
        "house","road","cloud","field","lamp","ship","bridge","forest","valley","coast"]
RELS = ["is near the","is far from the","is above the","is below the","is beside the"]

tok = AutoTokenizer.from_pretrained(MODEL_ID)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token


def effective_rank(model):
    """Elementi non nulli in tutti i lora_E = rank effettivo totale."""
    tot, nz = 0, 0
    for n, p in model.named_parameters():
        if "lora_E" in n:
            tot += p.numel()
            nz += int((p.detach().abs() > 0).sum().item())
    return nz, tot


def run(actuator_on, label):
    torch.manual_seed(SEED)
    r = random.Random(SEED)
    subs = SUBJ[:]; r.shuffle(subs)
    pairs = [(f"Complete: The {a} {r.choice(RELS)}", r.choice(SUBJ)) for a in subs]

    base = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32)
    cfg = AdaLoraConfig(
        task_type="CAUSAL_LM", init_r=INIT_R, target_r=TARGET_R, lora_alpha=16,
        tinit=int(STEPS*0.1), tfinal=int(STEPS*0.2), deltaT=10,
        total_step=STEPS, target_modules=["q_proj", "v_proj"])
    model = get_peft_model(base, cfg).to(DEV)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    ra = model.base_model.rankallocator

    traj = []
    model.train()
    for step in range(STEPS):
        prompt, target = pairs[step % len(pairs)]
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
        loss.backward()
        if actuator_on:
            model.base_model.update_and_allocate(step)
        else:
            ra.update_ipt(model.base_model.model)
        opt.step(); opt.zero_grad()
        nz, tot = effective_rank(model)
        traj.append(nz)
        if step % 20 == 0 or step == STEPS - 1:
            print(f"    step {step:3d} | rank effettivo {nz}/{tot}")

    del model, base
    if DEV == "cuda":
        torch.cuda.empty_cache()
    return traj, tot


print(f"VERIFICA ATTUATORE ADALORA | {MODEL_ID} | seed {SEED}\n")

print("--- Braccio A: update_and_allocate (attuatore ACCESO) ---")
traj_on, tot = run(True, "A")
print("\n--- Braccio B: solo update_ipt (attuatore SPENTO) ---")
traj_off, _ = run(False, "B")

print("\n================= VERDETTO =================")
print(f"capacita' totale (elementi lora_E): {tot}")
print(f"A acceso : inizio {traj_on[0]} -> minimo {min(traj_on)} -> fine {traj_on[-1]}"
      f"   (variazioni: {sum(1 for i in range(1,len(traj_on)) if traj_on[i]!=traj_on[i-1])} step)")
print(f"B spento : inizio {traj_off[0]} -> minimo {min(traj_off)} -> fine {traj_off[-1]}"
      f"   (variazioni: {sum(1 for i in range(1,len(traj_off)) if traj_off[i]!=traj_off[i-1])} step)")

drop = (traj_on[0] - min(traj_on)) / traj_on[0] * 100 if traj_on[0] else 0
print(f"\nRiduzione massima del rank nel braccio A: {drop:.1f}%")
if drop > 15:
    print("=> L'ATTUATORE HA AGITO in modo sostanziale.")
    print("   Il confronto precedente e' valido: in AdaLoRA la riallocazione")
    print("   avviene eppure NON contamina il segnale di importanza.")
elif drop > 3:
    print("=> Attuatore attivo ma con effetto modesto: il confronto regge,")
    print("   ma la conclusione va riportata con questa cautela.")
else:
    print("=> ATTUATORE PRATICAMENTE INATTIVO: il confronto precedente NON e'")
    print("   informativo. Va rifatto con un budget piu' aggressivo")
    print("   (es. init_r=24, target_r=4, deltaT piu' basso).")

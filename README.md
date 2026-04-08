{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Unified LoRA - MRPC Benchmark Example\n",
    "\n",
    "This notebook demonstrates Unified LoRA on the GLUE MRPC task.\n",
    "\n",
    "**Expected results:**\n",
    "- Baseline LoRA: F1 ~0.78-0.79\n",
    "- Unified LoRA: F1 ~0.78-0.79\n"
   ]
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "!pip install -q transformers datasets peft evaluate scikit-learn accelerate"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "import os\n",
    "os.environ['WANDB_DISABLED'] = 'true'\n",
    "\n",
    "import torch\n",
    "from datasets import load_dataset\n",
    "from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments\n",
    "from peft import LoraConfig, get_peft_model\n",
    "from torch.utils.data import DataLoader\n",
    "import evaluate\n",
    "\n",
    "from controller import UnifiedController\n",
    "\n",
    "device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')\n",
    "print(device)"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "dataset = load_dataset('glue','mrpc')['train'].train_test_split(test_size=0.2, seed=42)\n",
    "\n",
    "model_name = 'distilbert-base-uncased'\n",
    "tokenizer = AutoTokenizer.from_pretrained(model_name)\n",
    "\n",
    "def tokenize(ex):\n",
    "    return tokenizer(ex['sentence1'], ex['sentence2'], truncation=True, padding=True)\n",
    "\n",
    "train = dataset['train'].map(tokenize, batched=True).rename_column('label','labels')\n",
    "test  = dataset['test'].map(tokenize, batched=True).rename_column('label','labels')\n",
    "\n",
    "metric = evaluate.combine(['accuracy','f1'])\n",
    "\n",
    "def compute_metrics(p):\n",
    "    logits, labels = p\n",
    "    preds = torch.argmax(torch.tensor(logits), axis=-1)\n",
    "    return metric.compute(predictions=preds, references=labels)"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# BASELINE\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)\n",
    "model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=['q_lin','v_lin']))\n",
    "\n",
    "trainer = Trainer(\n",
    " model=model,\n",
    " train_dataset=train,\n",
    " eval_dataset=test,\n",
    " args=TrainingArguments(output_dir='./b', num_train_epochs=3, per_device_train_batch_size=16, fp16=True, report_to=None),\n",
    " compute_metrics=compute_metrics\n",
    ")\n",
    "\n",
    "trainer.train()\n",
    "base = trainer.evaluate()"
   ],
   "outputs": [],
   "execution_count": null
  },
  {
   "cell_type": "code",
   "metadata": {},
   "source": [
    "# UNIFIED\n",
    "ctrl = UnifiedController()\n",
    "\n",
    "model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)\n",
    "model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules=['q_lin','v_lin']))\n",
    "model.to(device)\n",
    "\n",
    "loader = DataLoader(train.remove_columns(['sentence1','sentence2','idx']), batch_size=16, shuffle=True)\n",
    "opt = torch.optim.AdamW(model.parameters(), lr=3e-5)\n",
    "\n",
    "model.train()\n",
    "\n",
    "for _ in range(3):\n",
    " for batch in loader:\n",
    "  batch = {k:v.to(device) for k,v in batch.items() if k in ['input_ids','attention_mask','labels']}\n",
    "  out = model(**batch)\n",
    "  lr = ctrl.update(out.loss.item())\n",
    "  for g in opt.param_groups: g['lr'] = lr\n",
    "  out.loss.backward()\n",
    "  opt.step(); opt.zero_grad()\n",
    "\n",
    "model.eval()\n",
    "trainer = Trainer(model=model, eval_dataset=test, args=TrainingArguments(output_dir='./u', per_device_eval_batch_size=16, fp16=True, report_to=None), compute_metrics=compute_metrics)\n",
    "uni = trainer.evaluate()"
   ],
   "outputs": [],
   "execution_count": null
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}

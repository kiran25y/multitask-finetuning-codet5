# encoder–decoder fine-tuning, tokenization here

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, os, sys, inspect, numpy as np, torch
import evaluate

from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Trainer, TrainingArguments
)
from torch.utils.data import Dataset
from datasets import Dataset as HFDataset  # HF library

# PEFT (LoRA / QLoRA)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
try:
    import bitsandbytes as bnb
    from transformers import BitsAndBytesConfig
    BNB_OK = True
except Exception:
    BNB_OK = False
    BitsAndBytesConfig = None

import my_data as myds  # your loader (renamed to avoid shadowing)

# ---------------- Dataset wrapper ----------------
class Seq2SeqJsonDataset(Dataset):
    def __init__(self, items):
        self.items = items
    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        it = self.items[idx]
        return {"task": it.task, "input_text": it.input_text, "target_text": it.target_text}

def build_hf_dataset(tokenizer, items, max_input_len=768, max_target_len=256):
    ds = Seq2SeqJsonDataset(items)
    hf = HFDataset.from_generator(lambda: (ds[i] for i in range(len(ds))))
    def _tok(batch):
        model_inputs = tokenizer(
            batch["input_text"],
            max_length=max_input_len,
            truncation=True,
            padding=False
        )
        # use text_target (no deprecation)
        labels = tokenizer(
            text_target=batch["target_text"],
            max_length=max_target_len,
            truncation=True,
            padding=False
        )
        model_inputs["labels"] = labels["input_ids"]
        model_inputs["task"] = batch["task"]
        return model_inputs
    return hf.map(_tok, batched=True, remove_columns=["input_text", "target_text"])

# ---------------- Metrics ----------------
bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

def postprocess_text(preds, labels):
    preds = [p.strip() for p in preds]
    labels = [l.strip() for l in labels]
    return preds, [[l] for l in labels]

def make_compute_metrics(tokenizer):
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        labels = np.where(labels != -100, labels, 0)
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        dp, dl = postprocess_text(decoded_preds, decoded_labels)
        b = bleu.compute(predictions=dp, references=dl)["score"]
        r = rouge.compute(predictions=dp, references=[x[0] for x in dl])["rougeL"]
        def looks_like_sig(s): return "(" in s and s.endswith(")")
        sig_pairs = [(p, l[0]) for p, l in zip(dp, dl) if looks_like_sig(l[0])]
        em = float(np.mean([p.strip() == l.strip() for p, l in sig_pairs])) if sig_pairs else 0.0
        return {"bleu": round(b,3), "rougeL": round(r,3), "sig_exact_match": round(em,4),
                "eval_count": len(dp)}
    return compute_metrics

# ---------------- PEFT helpers ----------------
def apply_lora(model, r=16, alpha=32, dropout=0.05, target_modules=None):
    if target_modules is None:
        target_modules = ["q", "k", "v", "o"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=target_modules, bias="none", task_type="SEQ_2_SEQ_LM"
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    return model

def load_model_with_mode(model_name, peft_mode: str, gradient_checkpointing: bool):
    """
    peft_mode: 'none' | 'lora' | 'qlora'
    """
    if peft_mode == "qlora":
        assert BNB_OK and BitsAndBytesConfig is not None, \
            "QLoRA needs bitsandbytes and a recent transformers. pip install -U bitsandbytes transformers"
        quant_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=gradient_checkpointing)
        model = apply_lora(model, r=16, alpha=32, dropout=0.05)
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        if peft_mode == "lora":
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
            model = apply_lora(model, r=16, alpha=32, dropout=0.05)
        else:
            if gradient_checkpointing:
                model.gradient_checkpointing_enable()
    return model

def build_training_args(
    save_dir, lr, epochs, bs, grad_accum, seed, bf16_ok, max_eval_bs
):
    """
    Creates TrainingArguments compatible with both older and newer transformers versions.
    Falls back if 'evaluation_strategy' is not supported.
    """
    sig = inspect.signature(TrainingArguments.__init__)
    kwargs = dict(
        output_dir=save_dir,
        overwrite_output_dir=True,
        learning_rate=lr,
        num_train_epochs=epochs,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=max_eval_bs,
        gradient_accumulation_steps=grad_accum,
        weight_decay=0.01,
        save_total_limit=2,
        logging_steps=50,
        report_to="none",
        seed=seed,
        bf16=bf16_ok,
        fp16=(not bf16_ok) and torch.cuda.is_available(),
    )
    if "evaluation_strategy" in sig.parameters:
        kwargs.update(dict(
            evaluation_strategy="epoch",
            save_strategy="epoch",
        ))
    else:
        # Older transformers: no evaluation scheduling keywords.
        # We’ll still call trainer.evaluate() after train.
        pass
    return TrainingArguments(**kwargs)

# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", default="Salesforce/codet5p-220m",
                    help="Encoder–decoder (CodeT5+). Try codet5p-770m if VRAM allows.")
    ap.add_argument("--peft", choices=["none","lora","qlora"], default="lora",
                    help="Fine-tuning mode.")
    ap.add_argument("--train_ratio", type=float, default=0.9)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max_input_len", type=int, default=768)
    ap.add_argument("--max_target_len", type=int, default=256)
    ap.add_argument("--save_dir", default="outputs/codet5p_mt_pandas")
    ap.add_argument("--limit_per_task", type=int, default=0, help="Balance tasks; 0=use all")
    ap.add_argument("--gradient_checkpointing", action="store_true")
    args = ap.parse_args()

    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Data
    limit = args.limit_per_task if args.limit_per_task > 0 else None
    all_items = myds.load_all_examples(limit_per_task=limit)
    train_items, val_items = myds.split_examples(all_items, train_ratio=args.train_ratio, seed=args.seed)
    print(f"[data] train={len(train_items)} val={len(val_items)} total={len(all_items)}")

    train_ds = build_hf_dataset(tokenizer, train_items, args.max_input_len, args.max_target_len)
    val_ds   = build_hf_dataset(tokenizer, val_items,   args.max_input_len, args.max_target_len)

    # Model (full / LoRA / QLoRA)
    model = load_model_with_mode(args.model_name, args.peft, args.gradient_checkpointing)

    # Collator & precision
    collator = DataCollatorForSeq2Seq(tokenizer, model=model)
    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = build_training_args(
        save_dir=args.save_dir,
        lr=args.lr,
        epochs=args.epochs,
        bs=args.batch_size,
        grad_accum=args.grad_accum,
        seed=args.seed,
        bf16_ok=bf16_ok,
        max_eval_bs=max(1, args.batch_size // 2)
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=make_compute_metrics(tokenizer),
    )

    trainer.train()
    # Even on very old transformers without scheduling, we still evaluate here:
    print("[eval]", trainer.evaluate())

    # Qualitative demo for slides
    demo_inputs = [
        "infer signature from body:\nreturn a + b",
        "summarize code:\ndef foo(x):\n    '''adds 1'''\n    return x+1",
        "retrieve code for query:\nresample a time series to business day frequency",
        "fix a bug:\n- if value is None:\n+ if value is not None:\n  do_something(value)"
    ]
    device = model.device
    toks = tokenizer(demo_inputs, return_tensors="pt", padding=True, truncation=True,
                     max_length=args.max_input_len).to(device)
    out_ids = model.generate(**toks, max_length=args.max_target_len)
    outs = tokenizer.batch_decode(out_ids, skip_special_tokens=True)
    print("\n[demo generations]")
    for q, a in zip(demo_inputs, outs):
        print(">>", q.split("\n", 1)[0]); print(a[:400], "\n")

if __name__ == "__main__":
    main()

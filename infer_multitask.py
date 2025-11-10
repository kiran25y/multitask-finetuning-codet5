# pretty outputs for your report

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse, json, torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def run_batch(model_name, inputs, max_input_len=768, max_target_len=256):
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    mdl = AutoModelForSeq2SeqLM.from_pretrained(model_name).eval()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(dev)
    enc = tok(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_input_len).to(dev)
    out = mdl.generate(**enc, max_length=max_target_len)
    return tok.batch_decode(out, skip_special_tokens=True)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True, help="checkpoint dir or HF id")
    ap.add_argument("--out_json", default="inference_report.json")
    args = ap.parse_args()

    prompts = {
        "signature": "infer signature from body:\nreturn a + b",
        "summarization": "summarize code:\n\ndef foo(x):\n    '''adds 1'''\n    return x+1",
        "search": "retrieve code for query:\nread CSV then set index to first column",
        "repair": "fix a bug:\n- if value is None:\n+ if value is not None:\n  do(value)"
    }
    order = list(prompts.keys())
    outputs = run_batch(args.model_path, [prompts[k] for k in order])
    report = {k: v for k, v in zip(order, outputs)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import random
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW  # <-- from torch, not transformers
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    get_linear_schedule_with_warmup,  # <-- add this line
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------
#                     DATASET WRAPPER (ALL 4 TASKS)
# ---------------------------------------------------------------------
class MultiTaskDataset(Dataset):
    def __init__(self, datasets: Dict[str, List[Dict]], tokenizer, max_len=512):
        self.items = []
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Flatten task->rows into a single list of (task, row)
        for task_name, rows in datasets.items():
            for r in rows:
                self.items.append((task_name, r))

        random.shuffle(self.items)

    def __len__(self):
        return len(self.items)

    def encode(self, text):
        return self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
        )

    def __getitem__(self, idx):
        task, row = self.items[idx]

        if task == "code_summarization":
            inp = "summarize: " + row["input"]
            tgt = row["target"]

        elif task == "signature_generation":
            inp = "signature: " + row["input"]
            tgt = row["target"]

        elif task == "code_repair":
            inp = "repair: " + row["input"]
            tgt = row["target"]

        elif task == "code_search":
            # "find code: <query>" -> "<positive snippet>"
            inp = "search: " + row["query"]
            tgt = row["positive"]

        else:
            raise ValueError(f"Unknown task: {task}")

        source = self.encode(inp)
        target = self.encode(tgt)

        return {
            "input_ids": source["input_ids"].squeeze(0),
            "attention_mask": source["attention_mask"].squeeze(0),
            "labels": target["input_ids"].squeeze(0),
        }


# ---------------------------------------------------------------------
#                     LOAD JSONL HELPER
# ---------------------------------------------------------------------
def load_jsonl(path: Path):
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # skip bad lines
                continue
    return out


# ---------------------------------------------------------------------
#                     TRAINING LOOP
# ---------------------------------------------------------------------
def train(model, tokenizer, dataloader, epochs=3, lr=3e-5, warmup=0.1, save_dir="checkpoints"):
    model.train()
    model.to(DEVICE)

    optim = AdamW(model.parameters(), lr=lr)
    total_steps = len(dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optim,
        num_warmup_steps=int(total_steps * warmup),
        num_training_steps=total_steps,
    )

    save_dir_path = Path(save_dir)
    save_dir_path.mkdir(exist_ok=True, parents=True)

    for epoch in range(1, epochs + 1):
        print(f"\n===== Epoch {epoch}/{epochs} =====")
        total_loss = 0.0

        for step, batch in enumerate(dataloader):
            optim.zero_grad()

            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            labels = batch["labels"].to(DEVICE)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optim.step()
            scheduler.step()

            if step % 100 == 0:
                print(f"Step {step}/{len(dataloader)} - Loss: {loss.item():.4f}")

        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch} Average Loss = {avg:.4f}")

        ckpt_path = save_dir_path / f"epoch_{epoch}"
        ckpt_path.mkdir(exist_ok=True, parents=True)
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    print("\nTraining complete!")


# ---------------------------------------------------------------------
#                             MAIN
# ---------------------------------------------------------------------
if __name__ == "__main__":
    data_root = Path("data/clean_data")

    print("Loading datasets...")
    datasets = {
        "code_summarization": load_jsonl(data_root / "code_summarization.jsonl"),
        "signature_generation": load_jsonl(data_root / "signature_generation.jsonl"),
        "code_search": load_jsonl(data_root / "code_search.jsonl"),
        "code_repair": load_jsonl(data_root / "code_repair.jsonl"),
    }

    print("Loading tokenizer & model...")
    model_name = "Salesforce/codet5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    dataset = MultiTaskDataset(datasets, tokenizer, max_len=512)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    print(f"Dataset size: {len(dataset)} items")

    train(
        model=model,
        tokenizer=tokenizer,
        dataloader=dataloader,
        epochs=5,
        lr=3e-5,
        warmup=0.1,
        save_dir="mtl_codet5_checkpoints",
    )

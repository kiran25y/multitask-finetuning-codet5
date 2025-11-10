# oader + on-the-fly split + task balancing

# my_data.py
from __future__ import annotations
import json, random
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass

@dataclass
class Example:
    task: str
    input_text: str
    target_text: str

CLEAN_DIR = Path("data/clean_data")
FILES = {
    "code_summarization": CLEAN_DIR / "code_summarization.jsonl",
    "signature_generation": CLEAN_DIR / "signature_generation.jsonl",
    "code_search": CLEAN_DIR / "code_search.jsonl",
    "code_repair": CLEAN_DIR / "code_repair.jsonl",
}

def _read_jsonl(p: Path):
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)

# ---- Short, consistent prompts per task ----
def fmt_summarization(d: Dict):   return f"summarize code:\n{d['input']}", d["target"]
def fmt_signature(d: Dict):       return f"infer signature from body:\n{d['input']}", d["target"]
def fmt_search(d: Dict):
    q = d["query"].strip().replace("\n", " ")
    return f"retrieve code for query:\n{q}", d["positive"]
def fmt_repair(d: Dict):          return "fix a bug:\n" + d["input"], d["target"]

FORMATTERS = {
    "code_summarization": fmt_summarization,
    "signature_generation": fmt_signature,
    "code_search": fmt_search,
    "code_repair": fmt_repair,
}

def load_all_examples(limit_per_task: int | None = None) -> List[Example]:
    """Read cleaned JSONLs and return a unified list of Examples (no tokenization)."""
    out: List[Example] = []
    for task, path in FILES.items():
        if not path.exists():
            print(f"[WARN] missing {path}")
            continue
        fmt = FORMATTERS[task]
        cnt = 0
        for d in _read_jsonl(path):
            inp, tgt = fmt(d)
            if len(inp) < 20 or len(tgt) < 3:
                continue
            out.append(Example(task=task, input_text=inp, target_text=tgt))
            cnt += 1
            if limit_per_task and cnt >= limit_per_task:
                break
        print(f"[load] {task}: {cnt} examples")
    return out

def split_examples(examples: List[Example], train_ratio=0.9, seed=42) -> Tuple[List[Example], List[Example]]:
    random.seed(seed)
    xs = examples[:]
    random.shuffle(xs)
    n_train = int(len(xs) * train_ratio)
    return xs[:n_train], xs[n_train:]


# import json, sys
# from pathlib import Path

# root = Path("data/clean_data")
# ok = True
# for fn in ["code_search.jsonl","code_repair.jsonl","code_summarization.jsonl","signature_generation.jsonl"]:
#     p = root / fn
#     if not p.exists():
#         print(f"[MISS] {p}"); ok=False; continue
#     n=0
#     with p.open() as f:
#         for i,l in enumerate(f,1):
#             try:
#                 obj=json.loads(l)
#                 assert obj.get("task"), "missing task"
#             except Exception as e:
#                 print(f"[BAD] {fn}:{i}: {e}"); ok=False; break
#             n+=1
#     print(f"[OK] {fn} lines={n}")
# sys.exit(0 if ok else 1)



#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
from pathlib import Path

CLEAN = Path("data/clean_data")
FILES = [
    "code_search.jsonl",
    "code_repair.jsonl",
    "code_summarization.jsonl",
    "signature_generation.jsonl",
]

def peek(path, n=2):
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= n: break
            print(" ", line[:160].rstrip())

def check_schema(name, row):
    if name == "code_search.jsonl":
        return all(k in row for k in ("task","language","query","positive"))
    if name == "code_repair.jsonl":
        return all(k in row for k in ("task","language","input","target"))
    if name == "code_summarization.jsonl":
        return all(k in row for k in ("task","language","input","target"))
    if name == "signature_generation.jsonl":
        return all(k in row for k in ("task","language","input","target"))
    return True

def main():
    assert CLEAN.exists(), "data/clean_data/ not found"
    for fname in FILES:
        p = CLEAN / fname
        assert p.exists(), f"{p} is missing"
        n = sum(1 for _ in p.open(encoding="utf-8"))
        assert n > 0, f"{p} is empty"
        with p.open(encoding="utf-8") as f:
            first = json.loads(next(f))
        assert check_schema(fname, first), f"{fname} schema looks wrong"
        print(f"[OK] {fname} lines={n}")
        peek(p)
    print("[ALL GOOD] cleaned datasets are ready.")
if __name__ == "__main__":
    main()

import json, sys
from pathlib import Path

root = Path("data/clean_data")
ok = True
for fn in ["code_search.jsonl","code_repair.jsonl","code_summarization.jsonl","signature_generation.jsonl"]:
    p = root / fn
    if not p.exists():
        print(f"[MISS] {p}"); ok=False; continue
    n=0
    with p.open() as f:
        for i,l in enumerate(f,1):
            try:
                obj=json.loads(l)
                assert obj.get("task"), "missing task"
            except Exception as e:
                print(f"[BAD] {fn}:{i}: {e}"); ok=False; break
            n+=1
    print(f"[OK] {fn} lines={n}")
sys.exit(0 if ok else 1)

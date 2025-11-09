#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations
import argparse, ast, json, re, sys, textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Dict, Optional

# Optional deps
try:
    from git import Repo
except Exception:
    Repo = None
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x
try:
    from rapidfuzz import process, fuzz
except Exception:
    process = None; fuzz = None

# ----------------------- Paths / Config -----------------------
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
REPO_DIR = RAW_DIR / "repo"
MINED_DIR = RAW_DIR / "mined"
CLEAN_DIR = DATA_DIR / "clean_data"

PY_EXTS = {".py"}
SKIP_SUBSTRINGS = [
    ".venv", "site-packages", "build/", "dist/", "docs/", "asv_bench/", "benchmarks/",
    "__pycache__", "typings/", "doc/", "doc/source", "scripts/bench"
]
# We’ll allow mining code from package sources but skip tests by default elsewhere.
REPAIR_TESTS_SKIP = re.compile(r"(?:^|/)tests?/", re.I)

BUGGY_HINTS = re.compile(r"\b(fix|bug|issue|regress|broken|error|typo|fail|incorrect)\b", re.I)

# ----------------------- Helpers -----------------------
def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")

def write_jsonl(path: Path, rows: Iterable[Dict]):
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def dedent_strip(s: str) -> str:
    return textwrap.dedent(s).strip()

def first_sentence(s: str) -> str:
    # very light heuristic: take first non-empty line, cut at ~sentence end
    s = s.strip()
    line = next((ln for ln in s.splitlines() if ln.strip()), "")
    # cut to 180 chars to avoid overly-long “queries/targets”
    cut = re.split(r"(?<=[.!?])\s", line, maxsplit=1)[0]
    return (cut[:180]).strip()

def iter_python_files(root: Path):
    for p in root.rglob("*.py"):
        sp = str(p)
        if any(s in sp for s in SKIP_SUBSTRINGS):
            continue
        yield p

# ----------------------- AST extraction -----------------------
@dataclass
class FuncInfo:
    file: str
    lineno: int
    end_lineno: int
    name: str
    signature: str
    docstring: Optional[str]
    body_src: str
    full_src: str

def _get_source_segment(text: str, node: ast.AST) -> str:
    lines = text.splitlines()
    start = getattr(node, "lineno", 1) - 1
    end = getattr(node, "end_lineno", start + 1)
    return "\n".join(lines[start:end])

def extract_functions(py_path: Path) -> List[FuncInfo]:
    txt = read_text(py_path)
    try:
        tree = ast.parse(txt)
    except Exception:
        return []
    out: List[FuncInfo] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            ds = ast.get_docstring(node)
            header_src = _get_source_segment(txt, node)
            # signature text
            pos = [a.arg for a in node.args.args]
            vararg = node.args.vararg.arg if node.args.vararg else None
            kwarg = node.args.kwarg.arg if node.args.kwarg else None
            parts = pos[:]
            if vararg: parts.append("*" + vararg)
            if node.args.kwonlyargs: parts += [ka.arg + "=…" for ka in node.args.kwownlyargs] if hasattr(node.args, "kwownlyargs") else [ka.arg + "=…" for ka in node.args.kwonlyargs]
            if kwarg: parts.append("**" + kwarg)
            sig = f"{node.name}({', '.join(parts)})"
            # body without leading docstring
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(getattr(node.body[0], "value", None), ast.Constant) and isinstance(node.body[0].value.value, str):
                body_nodes = node.body[1:]
            else:
                body_nodes = node.body
            if body_nodes:
                start = body_nodes[0].lineno
                end = getattr(body_nodes[-1], "end_lineno", start)
                body_src = "\n".join(txt.splitlines()[start-1:end])
            else:
                body_src = ""
            out.append(FuncInfo(
                file=str(py_path),
                lineno=node.lineno,
                end_lineno=getattr(node, "end_lineno", node.lineno),
                name=node.name,
                signature=sig,
                docstring=ds,
                body_src=dedent_strip(body_src),
                full_src=dedent_strip(header_src),
            ))
    return out

# ----------------------- Builders -----------------------
def build_code_summarization(repo_root: Path, min_chars=20) -> List[Dict]:
    rows = []
    for f in tqdm(list(iter_python_files(repo_root)), desc="Summarization"):
        for fx in extract_functions(f):
            if not fx.docstring:
                continue
            tgt = first_sentence(fx.docstring)
            if len(tgt) < min_chars:
                continue
            inp = f"# file: {fx.file}\n# signature: {fx.signature}\n\n{fx.full_src}"
            rows.append({
                "id": f"{fx.file}:{fx.lineno}",
                "task": "code_summarization",
                "language": "python",
                "input": inp,
                "target": tgt
            })
    return rows

def build_signature_generation(repo_root: Path, min_body=40) -> List[Dict]:
    rows = []
    for f in tqdm(list(iter_python_files(repo_root)), desc="SignatureGen"):
        for fx in extract_functions(f):
            body = fx.body_src
            if len(body) < min_body:
                continue
            rows.append({
                "id": f"{fx.file}:{fx.lineno}",
                "task": "signature_generation",
                "language": "python",
                "input": f"# Infer the function signature from the body.\n{body}",
                "target": fx.signature
            })
    return rows

def build_code_search(repo_root: Path, k_neg=3) -> List[Dict]:
    if process is None:
        print("[WARN] rapidfuzz not installed -> code_search will have empty negatives.", file=sys.stderr)
    funcs = []
    for f in tqdm(list(iter_python_files(repo_root)), desc="Search scan"):
        funcs.extend(extract_functions(f))
    names = [f"{fx.name} in {Path(fx.file).name}" for fx in funcs]
    rows = []
    for fx in tqdm(funcs, desc="Search pairs"):
        if not fx.docstring:
            continue
        query = first_sentence(fx.docstring)
        if not query:
            continue
        code = fx.full_src
        negs = []
        if process is not None:
            candidates = process.extract(fx.name, names, scorer=fuzz.token_set_ratio, limit=20)
            for _, _, idx in candidates:
                other = funcs[idx]
                if other.file != fx.file and other.name != fx.name:
                    negs.append(other.full_src)
                if len(negs) >= k_neg:
                    break
        rows.append({
            "id": f"{fx.file}:{fx.lineno}",
            "task": "code_search",
            "language": "python",
            "query": query,
            "positive": code,
            "negatives": [dedent_strip(n) for n in negs]
        })
    return rows

def build_code_repair(repo_root: Path, max_commits=3000, context=3, branch="main",
                      include_tests=False) -> List[Dict]:
    if Repo is None:
        print("[ERROR] GitPython not installed: pip install GitPython", file=sys.stderr)
        return []
    repo = Repo(repo_root)
    rows = []
    commits = list(repo.iter_commits(branch, max_count=max_commits))
    for c in tqdm(commits, desc="Repair mining"):
        msg = (c.message or "").lower()
        if not BUGGY_HINTS.search(msg):
            continue
        parent = c.parents[0] if c.parents else None
        if not parent:
            continue
        diffs = c.diff(parent, create_patch=True)
        for d in diffs:
            if not d.b_path or not d.b_path.endswith(".py"):
                continue
            if (not include_tests) and REPAIR_TESTS_SKIP.search(d.b_path or ""):
                continue
            patch = d.diff.decode("utf-8", errors="ignore")
            lines = patch.splitlines()
            buggy_chunk, fixed_chunk = [], []
            ctx_before: List[str] = []

            def flush_pair():
                nonlocal buggy_chunk, fixed_chunk
                bi = dedent_strip("\n".join(buggy_chunk))
                fi = dedent_strip("\n".join(fixed_chunk))
                # skip empty & whitespace-only changes
                if not bi or not fi:
                    buggy_chunk.clear(); fixed_chunk.clear(); return
                if bi.replace(" ", "") == fi.replace(" ", ""):
                    buggy_chunk.clear(); fixed_chunk.clear(); return
                rows.append({
                    "id": f"{c.hexsha}:{d.b_path}:{len(rows)}",
                    "task": "code_repair",
                    "language": "python",
                    "meta": {"commit": c.hexsha, "file": d.b_path, "message": (c.message or "").strip()},
                    "input": bi,
                    "target": fi
                })
                buggy_chunk.clear(); fixed_chunk.clear()

            for ln in lines:
                if ln.startswith("@@"):
                    flush_pair()
                    ctx_before = []
                    continue
                if ln.startswith("+") and not ln.startswith("+++"):
                    fixed_chunk.append(ln[1:])
                    if ctx_before:
                        fixed_chunk = ctx_before[-context:] + fixed_chunk
                elif ln.startswith("-") and not ln.startswith("---"):
                    buggy_chunk.append(ln[1:])
                    if ctx_before:
                        buggy_chunk = ctx_before[-context:] + buggy_chunk
                else:
                    ctx_line = ln[1:] if ln.startswith(" ") else ln
                    ctx_before.append(ctx_line)
                    ctx_before = ctx_before[-(context + 1):]
            flush_pair()
    return rows

# ----------------------- Cleaning -----------------------
def clean_generic(rows: List[Dict], min_input_chars=40, min_target_chars=8, max_len_bytes=20000) -> List[Dict]:
    seen = set(); cleaned = []
    for r in rows:
        # choose input candidate field
        candidate = r.get("input") or r.get("positive") or r.get("query") or ""
        if len(candidate) < min_input_chars:
            continue
        if r["task"] in ("code_summarization", "signature_generation", "code_repair"):
            if len(r.get("target", "")) < min_target_chars:
                continue
        if len(json.dumps(r)) > max_len_bytes:
            continue
        key = (r["task"], r.get("id"))
        if key in seen: 
            continue
        seen.add(key); cleaned.append(r)
    return cleaned

def clean_code_search(rows: List[Dict]) -> List[Dict]:
    out = []
    for r in rows:
        if not r.get("query") or not r.get("positive"):
            continue
        pos = dedent_strip(r["positive"])
        negs = [n for n in (r.get("negatives") or []) if n.strip() and dedent_strip(n) != pos]
        r["positive"] = pos
        r["negatives"] = negs
        out.append(r)
    return clean_generic(out, min_input_chars=20)

# ----------------------- Orchestration -----------------------
def clone_repo(repo_url: str, out: Path, depth: int = 0):
    if Repo is None:
        print("[ERROR] GitPython not installed: pip install GitPython", file=sys.stderr)
        sys.exit(1)
    if out.exists() and (out / ".git").exists():
        print(f"[clone] repo already exists at {out}")
        return
    ensure_dir(out.parent)
    print(f"[clone] cloning {repo_url} -> {out} (depth={depth})")
    Repo.clone_from(repo_url, out, depth=depth if depth and depth > 0 else None)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo-url", default="https://github.com/pandas-dev/pandas")
    ap.add_argument("--no-clone", action="store_true")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--depth", type=int, default=0)
    ap.add_argument("--max-commits", type=int, default=3000)
    ap.add_argument("--steps", nargs="+", default=["summarization","signature","search","repair"],
        choices=["summarization","signature","search","repair"])
    ap.add_argument("--skip-clean", action="store_true")
    ap.add_argument("--include-tests-for-repair", action="store_true")
    args = ap.parse_args()

    ensure_dir(DATA_DIR); ensure_dir(RAW_DIR); ensure_dir(MINED_DIR); ensure_dir(CLEAN_DIR)

    if not args.no_clone:
        clone_repo(args.repo_url, REPO_DIR, depth=args.depth)
    elif not REPO_DIR.exists():
        print("[ERROR] --no-clone but data/raw/repo not found", file=sys.stderr); sys.exit(1)

    # mine raw
    raw_sum = raw_sig = raw_search = raw_rep = []
    if "summarization" in args.steps:
        raw_sum = build_code_summarization(REPO_DIR)
        write_jsonl(MINED_DIR / "code_summarization_all_raw.jsonl", raw_sum)
    if "signature" in args.steps:
        raw_sig = build_signature_generation(REPO_DIR)
        write_jsonl(MINED_DIR / "signature_generation_all_raw.jsonl", raw_sig)
    if "search" in args.steps:
        raw_search = build_code_search(REPO_DIR)
        write_jsonl(MINED_DIR / "code_search_all_raw.jsonl", raw_search)
    if "repair" in args.steps:
        raw_rep = build_code_repair(REPO_DIR, max_commits=args.max_commits, branch=args.branch,
                                    include_tests=args.include_tests_for_repair)
        write_jsonl(MINED_DIR / "code_repair_all_raw.jsonl", raw_rep)

    if args.skip_clean:
        print("[done] Raw JSONLs under data/raw/mined/. Skipped cleaning.")
        return

    # clean -> single JSONL per task (no splits)
    if raw_sum:
        clean_sum = clean_generic(raw_sum, min_input_chars=20, min_target_chars=12)
        write_jsonl(CLEAN_DIR / "code_summarization.jsonl", clean_sum)
        print(f"[clean] summarization: {len(clean_sum)} items")
    if raw_sig:
        clean_sig = clean_generic(raw_sig, min_input_chars=40, min_target_chars=6)
        write_jsonl(CLEAN_DIR / "signature_generation.jsonl", clean_sig)
        print(f"[clean] signature_generation: {len(clean_sig)} items")
    if raw_search:
        clean_s = clean_code_search(raw_search)
        write_jsonl(CLEAN_DIR / "code_search.jsonl", clean_s)
        print(f"[clean] code_search: {len(clean_s)} items")
    if raw_rep:
        rr = [r for r in raw_rep if r.get("input","").strip() and r.get("target","").strip()]
        clean_r = clean_generic(rr, min_input_chars=10, min_target_chars=5, max_len_bytes=40000)
        write_jsonl(CLEAN_DIR / "code_repair.jsonl", clean_r)
        print(f"[clean] code_repair: {len(clean_r)} items")

    print("[done] Clean JSONLs saved under data/clean_data/.")
if __name__ == "__main__":
    main()

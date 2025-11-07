import os
import ast
import json
import subprocess
from git import Repo
from pydriller import Repository

# ======================================
# Configuration
# ======================================
REPO_URL = "https://github.com/pandas-dev/pandas.git"
REPO_PATH = "D:/Github/pandas"  # local clone path
OUTPUT_DIR = "D:/Github/multitask-finetuning-codet5/dataset"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ======================================
# 0. Clone repository if not exists
# ======================================
if not os.path.exists(REPO_PATH):
    print(f"Cloning {REPO_URL} to {REPO_PATH} ...")
    Repo.clone_from(REPO_URL, REPO_PATH)
else:
    print(f"Repository already exists at {REPO_PATH}")

# ======================================
# 1. Extract (code, docstring) pairs — for code summarization
# ======================================
def extract_code_docstrings(repo_path):
    results = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            doc = ast.get_docstring(node)
                            code = ast.get_source_segment(source, node)
                            if doc and code:
                                results.append({
                                    "docstring": doc.strip(),
                                    "code": code.strip(),
                                    "file": filepath.replace(repo_path, "")
                                })
                except Exception as e:
                    print(f"[Docstring] Skipped {file}: {e}")
    return results

# ======================================
# 2. Extract buggy/fixed code pairs — for code repair
# ======================================
def extract_bug_fix_pairs(repo_path):
    pairs = []
    print("Traversing commit history for bug-fix pairs...")
    for commit in Repository(repo_path, only_modifications_with_file_types=[".py"]).traverse_commits():
        msg = commit.msg.lower()
        if "fix" in msg or "bug" in msg or "error" in msg or "issue" in msg:
            for mod in commit.modified_files:
                if mod.change_type.name == "MODIFY" and mod.source_code_before and mod.source_code:
                   
                    pairs.append({
                        "buggy_code": mod.source_code_before,
                        "fixed_code": mod.source_code,
                        "commit_msg": commit.msg.strip(),
                        "commit_hash": commit.hash,
                        "file": mod.new_path or mod.old_path
                    })
    return pairs

# ======================================
# 3. Extract (signature, body) pairs — for method generation
# ======================================
def extract_signature_body(repo_path):
    pairs = []
    for root, _, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                filepath = os.path.join(root, file)
                try:
                    with open(filepath, "r", encoding="utf-8") as f:
                        src = f.read()
                    tree = ast.parse(src)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            signature = f"def {node.name}({', '.join([arg.arg for arg in node.args.args])}):"
                            body_lines = []
                            for b in node.body:
                                if hasattr(b, "lineno") and hasattr(b, "end_lineno"):
                                    body_lines.extend(src.splitlines()[b.lineno - 1:b.end_lineno])
                            body = "\n".join(body_lines).strip()
                            if signature and body:
                                pairs.append({
                                    "signature": signature,
                                    "body": body,
                                    "file": filepath.replace(repo_path, "")
                                })
                except Exception as e:
                    print(f"[Signature] Skipped {file}: {e}")
    return pairs

# ======================================
# 4. Extract commit messages & code diffs — for code search
# ======================================
def extract_commit_msg_and_diff(repo_path, max_commits=2000):
    repo = Repo(repo_path)
    data = []
    commits = list(repo.iter_commits("main", max_count=max_commits))
    for i, commit in enumerate(commits):
        for parent in commit.parents:
            diff = parent.diff(commit, create_patch=True)
            for patch in diff:
                if patch.change_type == "M" and patch.b_path and patch.b_path.endswith(".py"):
                    try:
                        diff_text = patch.diff
                        if isinstance(diff_text, bytes):
                            diff_text = diff_text.decode("utf-8", errors="ignore")

                        data.append({
                            "query": commit.message.strip(),
                            "code_diff": diff_text or "",
                            "file": patch.b_path,
                            "commit": commit.hexsha
                        })
                    except Exception as e:
                        print(f"[Diff] Skipped commit {commit.hexsha}: {e}")
        if (i + 1) % 100 == 0:
            print(f"Processed {i+1}/{max_commits} commits...")
    return data

# ======================================
# Main pipeline
# ======================================
print("\nExtracting code-docstring pairs...")
code_doc_data = extract_code_docstrings(REPO_PATH)
with open(os.path.join(OUTPUT_DIR, "code_docstring.json"), "w", encoding="utf-8") as f:
    json.dump(code_doc_data, f, indent=2)

print("\nExtracting buggy-fixed pairs...")
bug_fix_data = extract_bug_fix_pairs(REPO_PATH)
with open(os.path.join(OUTPUT_DIR, "buggy_fixed.json"), "w", encoding="utf-8") as f:
    json.dump(bug_fix_data, f, indent=2)

print("\nExtracting signature-body pairs...")
sig_body_data = extract_signature_body(REPO_PATH)
with open(os.path.join(OUTPUT_DIR, "signature_body.json"), "w", encoding="utf-8") as f:
    json.dump(sig_body_data, f, indent=2)

print("\nExtracting commit messages and code diffs...")
commit_diff_data = extract_commit_msg_and_diff(REPO_PATH)
with open(os.path.join(OUTPUT_DIR, "commitmsg_code.json"), "w", encoding="utf-8") as f:
    json.dump(commit_diff_data, f, indent=2)

print("\n✅ Data extraction complete!")
print(f"Files saved to: {OUTPUT_DIR}")
print({
    "code_docstring": len(code_doc_data),
    "buggy_fixed": len(bug_fix_data),
    "signature_body": len(sig_body_data),
    "commitmsg_code": len(commit_diff_data)
})

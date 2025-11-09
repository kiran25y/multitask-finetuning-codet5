# import os
# import json
# import re

# # ======================================
# # CONFIGURATION
# # ======================================
# DATA_DIR = "D:/Github/multitask-finetuning-codet5/dataset"
# OUTPUT_DIR = os.path.join(DATA_DIR, "cleaned")
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# MAX_CODE_LENGTH = 8000  # truncate very long code to avoid training issues
# MAX_TEXT_LENGTH = 2000  # for docstrings, commit messages, etc.

# # ======================================
# # UTILITY FUNCTIONS
# # ======================================

# def clean_text(text):
#     """Remove excessive whitespace, newlines, and strange chars."""
#     text = re.sub(r"[^\x00-\x7F]+", " ", text)  # remove non-ASCII
#     text = re.sub(r"\s+", " ", text).strip()
#     return text

# def clean_code(code):
#     """Remove trailing spaces, normalize indentation, and strip."""
#     code = re.sub(r"\t", "    ", code)
#     code = re.sub(r" +\n", "\n", code)
#     code = code.strip()
#     return code

# def truncate(text, max_len):
#     """Limit very long samples."""
#     return text[:max_len]

# # ======================================
# # CLEANERS FOR EACH TASK
# # ======================================

# def clean_code_docstring(data):
#     cleaned = []
#     seen = set()
#     for d in data:
#         doc = clean_text(d.get("docstring", ""))
#         code = clean_code(d.get("code", ""))
#         if len(doc) < 5 or len(code) < 10:
#             continue
#         key = (doc, code)
#         if key in seen:
#             continue
#         seen.add(key)
#         cleaned.append({
#             "docstring": truncate(doc, MAX_TEXT_LENGTH),
#             "code": truncate(code, MAX_CODE_LENGTH),
#             "file": d.get("file", "")
#         })
#     return cleaned

# def clean_buggy_fixed(data):
#     cleaned = []
#     seen = set()
#     for d in data:
#         buggy = clean_code(d.get("buggy_code", ""))
#         fixed = clean_code(d.get("fixed_code", ""))
#         msg = clean_text(d.get("commit_msg", ""))
#         if len(buggy) < 10 or len(fixed) < 10:
#             continue
#         key = (buggy, fixed)
#         if key in seen:
#             continue
#         seen.add(key)
#         cleaned.append({
#             "buggy_code": truncate(buggy, MAX_CODE_LENGTH),
#             "fixed_code": truncate(fixed, MAX_CODE_LENGTH),
#             "commit_msg": truncate(msg, MAX_TEXT_LENGTH),
#             "commit_hash": d.get("commit_hash", ""),
#             "file": d.get("file", "")
#         })
#     return cleaned

# def clean_signature_body(data):
#     cleaned = []
#     seen = set()
#     for d in data:
#         sig = clean_text(d.get("signature", ""))
#         body = clean_code(d.get("body", ""))
#         if len(sig) < 5 or len(body) < 10:
#             continue
#         key = (sig, body)
#         if key in seen:
#             continue
#         seen.add(key)
#         cleaned.append({
#             "signature": truncate(sig, MAX_TEXT_LENGTH),
#             "body": truncate(body, MAX_CODE_LENGTH),
#             "file": d.get("file", "")
#         })
#     return cleaned

# def clean_commitmsg_code(data):
#     cleaned = []
#     seen = set()
#     for d in data:
#         query = clean_text(d.get("query", ""))
#         diff = d.get("code_diff", "")
#         # remove non-diff lines
#         diff = "\n".join([line for line in diff.splitlines() if line.startswith(("+", "-", "@@"))])
#         diff = clean_code(diff)
#         if len(query) < 5 or len(diff) < 10:
#             continue
#         key = (query, diff)
#         if key in seen:
#             continue
#         seen.add(key)
#         cleaned.append({
#             "query": truncate(query, MAX_TEXT_LENGTH),
#             "code_diff": truncate(diff, MAX_CODE_LENGTH),
#             "file": d.get("file", ""),
#             "commit": d.get("commit", "")
#         })
#     return cleaned

# # ======================================
# # MAIN PIPELINE
# # ======================================

# def clean_and_save(filename, cleaner):
#     src = os.path.join(DATA_DIR, filename)
#     if not os.path.exists(src):
#         print(f"⚠️ Missing {filename}, skipping.")
#         return
#     with open(src, "r", encoding="utf-8") as f:
#         data = json.load(f)
#     print(f"Cleaning {filename} ({len(data)} samples)...")
#     cleaned = cleaner(data)
#     outpath = os.path.join(OUTPUT_DIR, filename.replace(".json", "_clean.json"))
#     with open(outpath, "w", encoding="utf-8") as f:
#         json.dump(cleaned, f, indent=2, ensure_ascii=False)
#     print(f"✅ Saved cleaned data to {outpath} ({len(cleaned)} valid samples)\n")

# if __name__ == "__main__":
#     clean_and_save("code_docstring.json", clean_code_docstring)
#     clean_and_save("buggy_fixed.json", clean_buggy_fixed)
#     clean_and_save("signature_body.json", clean_signature_body)
#     clean_and_save("commitmsg_code.json", clean_commitmsg_code)
#     print("🎯 All datasets cleaned successfully!")



import os
import re
import json
import ijson

# ======================================
# Configuration
# ======================================
DATA_DIR = "D:/Github/multitask-finetuning-codet5/dataset"
OUTPUT_DIR = os.path.join(DATA_DIR, "cleaned")
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_CODE_LENGTH = 8000
MAX_TEXT_LENGTH = 2000

# ======================================
# Cleaning Utilities
# ======================================
def clean_text(text):
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def clean_code(code):
    code = code.replace("\t", "    ")
    code = re.sub(r" +\n", "\n", code)
    code = re.sub(r"[^\x00-\x7F]+", " ", code)
    return code.strip()

def truncate(text, max_len):
    return text[:max_len]

# ======================================
# Cleaning Functions for Each Dataset
# ======================================
def clean_code_docstring_item(d):
    try:
        code = clean_code(d.get("code", ""))
        doc = clean_text(d.get("docstring", ""))
        if len(code) < 10 or len(doc) < 5:
            return None
        return {
            "code": truncate(code, MAX_CODE_LENGTH),
            "docstring": truncate(doc, MAX_TEXT_LENGTH),
            "file": d.get("file", "")
        }
    except Exception:
        return None

def clean_buggy_fixed_item(d):
    try:
        buggy = clean_code(d.get("buggy_code", ""))
        fixed = clean_code(d.get("fixed_code", ""))
        msg = clean_text(d.get("commit_msg", ""))
        if len(buggy) < 10 or len(fixed) < 10:
            return None
        return {
            "buggy_code": truncate(buggy, MAX_CODE_LENGTH),
            "fixed_code": truncate(fixed, MAX_CODE_LENGTH),
            "commit_msg": truncate(msg, MAX_TEXT_LENGTH),
            "commit_hash": d.get("commit_hash", ""),
            "file": d.get("file", "")
        }
    except Exception:
        return None

def clean_signature_body_item(d):
    try:
        sig = clean_text(d.get("signature", ""))
        body = clean_code(d.get("body", ""))
        if len(sig) < 5 or len(body) < 10:
            return None
        return {
            "signature": truncate(sig, 400),
            "body": truncate(body, MAX_CODE_LENGTH),
            "file": d.get("file", "")
        }
    except Exception:
        return None

def clean_commitmsg_code_item(d):
    try:
        msg = clean_text(d.get("query", ""))
        diff = clean_code(d.get("code_diff", ""))
        if len(msg) < 5 or len(diff) < 10:
            return None
        return {
            "query": truncate(msg, MAX_TEXT_LENGTH),
            "code_diff": truncate(diff, MAX_CODE_LENGTH),
            "file": d.get("file", ""),
            "commit": d.get("commit", "")
        }
    except Exception:
        return None

# ======================================
# Safe JSON Load (for small/medium files)
# ======================================
def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# ======================================
# Stream Cleaning (for large JSON files)
# ======================================
def stream_clean(input_file, output_file, clean_func):
    count, valid = 0, 0
    with open(input_file, "r", encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        out.write("[\n")
        parser = ijson.items(f, "item")
        first = True
        for item in parser:
            count += 1
            cleaned = clean_func(item)
            if cleaned:
                if not first:
                    out.write(",\n")
                json.dump(cleaned, out, ensure_ascii=False)
                first = False
                valid += 1
            if count % 1000 == 0:
                print(f"Processed {count} items... kept {valid}")
        out.write("\n]")
    print(f"✅ Cleaned {valid}/{count} valid samples -> {output_file}")

# ======================================
# Smart Cleaner (auto-choose method)
# ======================================
def clean_and_save(filename, clean_func):
    src = os.path.join(DATA_DIR, filename)
    out = os.path.join(OUTPUT_DIR, filename.replace(".json", "_clean.json"))

    # Skip if file missing
    if not os.path.exists(src):
        print(f"⚠️ Skipping {filename} (file not found)")
        return

    file_size = os.path.getsize(src) / (1024 * 1024)  # in MB
    print(f"\nCleaning {filename} ({file_size:.2f} MB)...")

    try:
        # Stream for large files > 200MB
        if file_size > 200:
            stream_clean(src, out, clean_func)
        else:
            data = safe_load_json(src)
            cleaned = [clean_func(d) for d in data if clean_func(d)]
            with open(out, "w", encoding="utf-8") as f:
                json.dump(cleaned, f, indent=2, ensure_ascii=False)
            print(f"✅ Saved cleaned data to {out} ({len(cleaned)} valid samples)")
    except MemoryError:
        print("⚠️ MemoryError! Retrying with streaming mode...")
        stream_clean(src, out, clean_func)
    except Exception as e:
        print(f"❌ Error cleaning {filename}: {e}")

# ======================================
# Run All Cleaning Tasks
# ======================================
if __name__ == "__main__":
    clean_and_save("code_docstring.json", clean_code_docstring_item)
    clean_and_save("buggy_fixed.json", clean_buggy_fixed_item)
    clean_and_save("signature_body.json", clean_signature_body_item)
    clean_and_save("commitmsg_code.json", clean_commitmsg_code_item)

    print("\n🎯 All datasets cleaned and saved to:", OUTPUT_DIR)

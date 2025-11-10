import os
import re
import json
import nltk
import numpy as np
from tqdm import tqdm
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

# Ensure necessary resources
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# === CONFIGURATION ===
BASE_DIR = "D:/Github/multitask-finetuning-codet5/dataset/cleaned"
OUTPUT_DIR = os.path.join(BASE_DIR, "preprocessed")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === UTILITIES ===
def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def clean_text(text):
    """Clean code or text by removing excessive whitespace, comments, etc."""
    text = re.sub(r"#.*", "", text)  # remove comments
    text = re.sub(r'""".*?"""', '', text, flags=re.DOTALL)
    text = re.sub(r"'''(.*?)'''", '', text, flags=re.DOTALL)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def tokenize_dataset(data, keys):
    """Tokenize selected text fields and combine into a list of token lists."""
    tokenized = []
    for item in tqdm(data, desc="Tokenizing"):
        tokens = []
        for k in keys:
            if k in item and item[k]:
                cleaned = clean_text(item[k])
                tokens.extend(word_tokenize(cleaned))
        if tokens:
            tokenized.append(tokens)
    return tokenized

# === MAIN ===
def main():
    datasets = {}

    paths = {
        "code_docstring": os.path.join(BASE_DIR, "code_docstring_clean.json"),
        "commitmsg_code": os.path.join(BASE_DIR, "commitmsg_code_clean.json"),
        "buggy_fixed": os.path.join(BASE_DIR, "buggy_fixed_clean.json"),
        "signature_body": os.path.join(BASE_DIR, "signature_body_clean.json"),
    }

    # Load all datasets
    for name, path in paths.items():
        if os.path.exists(path):
            datasets[name] = load_json(path)
            print(f"Loaded {len(datasets[name])} samples from {os.path.basename(path)}")
        else:
            print(f"Missing file: {path}")

    # === TOKENIZATION ===
    tokenized_codedoc = tokenize_dataset(datasets["code_docstring"], ["code", "docstring"])
    tokenized_commitmsg = tokenize_dataset(datasets["commitmsg_code"], ["query", "code"])
    tokenized_buggyfix = tokenize_dataset(datasets["buggy_fixed"], ["buggy_code", "fixed_code"])
    tokenized_signature = tokenize_dataset(datasets["signature_body"], ["signature", "body"])

    # Combine all tokenized datasets for shared Word2Vec
    all_tokens = tokenized_codedoc + tokenized_commitmsg + tokenized_buggyfix + tokenized_signature

    print(f"Training Word2Vec on {len(all_tokens)} samples...")
    w2v_model = Word2Vec(
        sentences=all_tokens,
        vector_size=300,
        window=5,
        min_count=2,
        workers=4,
        sg=1  # skip-gram model
    )

    w2v_model.save(os.path.join(OUTPUT_DIR, "word2vec.model"))
    print("Word2Vec model saved.")

    # === FEATURE EXTRACTION ===
    def vectorize(tokens, model):
        """Convert list of tokens into a mean embedding vector."""
        vectors = [model.wv[w] for w in tokens if w in model.wv]
        return np.mean(vectors, axis=0).tolist() if vectors else np.zeros(model.vector_size).tolist()

    def save_vectors(name, tokenized_data):
        vectors = [vectorize(tokens, w2v_model) for tokens in tqdm(tokenized_data, desc=f"Vectorizing {name}")]
        np.save(os.path.join(OUTPUT_DIR, f"{name}_vectors.npy"), np.array(vectors))
        print(f"Saved {name} vectors to {OUTPUT_DIR}/{name}_vectors.npy")

    save_vectors("code_docstring", tokenized_codedoc)
    save_vectors("commitmsg_code", tokenized_commitmsg)
    save_vectors("buggy_fixed", tokenized_buggyfix)
    save_vectors("signature_body", tokenized_signature)

    print("\nData preprocessing and feature extraction complete!")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
scripts/build_embeddings.py

Compute and cache OpenAI embeddings for the labeled dataset.

Usage:
  python3 scripts/build_embeddings.py \
    --labels data/labels/human_labeled_clean.csv \
    --out_dir data/embeddings \
    --model text-embedding-3-small \
    --batch 50

Outputs:
 - data/embeddings/embeddings.npy      (N x D float32 array)
 - data/embeddings/meta.csv           (original rows with index -> mapping)
 - data/embeddings/clean_texts.txt    (one-per-line)
"""
import os, argparse, json, time
from pathlib import Path
import numpy as np
import pandas as pd
from openai import OpenAI

def embed_batch(client, texts, model):
    # client.embeddings.create supports list inputs; returns data list
    resp = client.embeddings.create(model=model, input=texts)
    return [r.embedding for r in resp.data]

def main(args):
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    model = args.model
    batch = int(args.batch)

    out_dir.mkdir(parents=True, exist_ok=True)
    print("[+] Loading labels:", labels_path)
    df = pd.read_csv(labels_path, low_memory=False)
    if "clean_text" not in df.columns:
        raise SystemExit("label CSV must contain 'clean_text' column")

    texts = df["clean_text"].fillna("").astype(str).tolist()
    n = len(texts)
    print(f"[+] Will embed {n} examples with model {model}")

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in this terminal")
    client = OpenAI(api_key=api_key)

    vectors = []
    for i in range(0, n, batch):
        batch_texts = texts[i:i+batch]
        print(f"[+] Embedding batch {i}..{i+len(batch_texts)-1}")
        try:
            embs = embed_batch(client, batch_texts, model)
        except Exception as e:
            print("Embedding call failed:", e)
            raise
        vectors.extend(embs)
        time.sleep(0.2)

    arr = np.array(vectors, dtype=np.float32)
    np.save(out_dir / "embeddings.npy", arr)
    df_meta = df.reset_index().rename(columns={"index":"row_index"})
    df_meta.to_csv(out_dir / "meta.csv", index=False)
    (out_dir / "clean_texts.txt").write_text("\n".join(texts), encoding="utf-8")
    print("[+] Saved embeddings:", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", default="data/embeddings")
    p.add_argument("--model", default="text-embedding-3-small")
    p.add_argument("--batch", type=int, default=50)
    args = p.parse_args()
    main(args)

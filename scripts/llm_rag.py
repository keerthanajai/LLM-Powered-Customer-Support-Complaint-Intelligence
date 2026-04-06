#!/usr/bin/env python3
"""
scripts/llm_rag.py

Retrieval-Augmented Generation (RAG) classification pipeline.

Features:
- Loads precomputed embeddings for your human-labeled examples.
- For each ticket (or labeled subset), retrieves top-k nearest labeled examples.
- Builds a dynamic few-shot prompt using nearest examples.
- Calls OpenAI via the new client API with function-calling to force structured JSON.
- Robustly extracts function_call.arguments (works with object or dict resp shapes).
- Canonicalizes model outputs to your fixed label set.
- Saves per-sample outputs and an evaluation summary (if labels provided).

Usage (example):
python3 scripts/llm_rag.py \
  --tickets data/clean/tickets.parquet \
  --labels data/labels/human_labeled_clean.csv \
  --emb_dir data/embeddings \
  --out_dir results/llm_rag \
  --k 3 \
  --model gpt-4o-mini \
  --embedding_model text-embedding-3-small \
  --batch 20

Notes:
- Requires data/embeddings/embeddings.npy, meta.csv, clean_texts.txt produced by scripts/build_embeddings.py
- Make sure OPENAI_API_KEY is exported in your terminal.
"""
import os
import json
import time
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import classification_report, f1_score, accuracy_score

from openai import OpenAI

# -------------------------
# Prompt + function schema
# -------------------------
EXAMPLE_SNIPPET = """Ticket:
```{text}```
Output:
{json}
"""

PROMPT_HEADER = """You are a strict ticket classification assistant. Output ONLY a single JSON object (no extra text).
Use keys (case-sensitive): {"complaint_type","severity","sentiment","root_cause","action_required"}.
Allowed complaint_type: ["Billing","Login","Delivery","Product Defect","Performance","Customer Service","Other"].
Allowed severity: ["Low","Medium","High"].
Allowed sentiment: ["Positive","Neutral","Negative"].
Allowed action_required: ["Yes","No"].

Normalization rules:
- Map text containing 'bill' or 'payment' -> Billing
- Map text containing 'login' or 'auth' -> Login
- Map text containing 'deliver' -> Delivery
- Map text containing 'product' or 'defect' -> Product Defect
- Map text containing 'perf' or 'latency' -> Performance
- Map text containing 'customer' or 'support' -> Customer Service
- root_cause: short phrase (1-6 words); if unknown -> "Unknown"

Return only JSON for the ticket you are given below.
"""

FUNCTION_SCHEMA = {
  "name": "classify_ticket",
  "description": "Structured classification of a user support ticket",
  "parameters": {
    "type": "object",
    "properties": {
      "complaint_type": {"type":"string", "enum":["Billing","Login","Delivery","Product Defect","Performance","Customer Service","Other"]},
      "severity": {"type":"string", "enum":["Low","Medium","High"]},
      "sentiment": {"type":"string", "enum":["Positive","Neutral","Negative"]},
      "root_cause": {"type":"string"},
      "action_required": {"type":"string", "enum":["Yes","No"]}
    },
    "required": ["complaint_type","severity","sentiment","root_cause","action_required"]
  }
}

# -------------------------
# Utilities
# -------------------------
def load_embeddings(emb_dir: Path):
    """Load saved embeddings and metadata produced by build_embeddings.py"""
    emb_dir = Path(emb_dir)
    vec_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "meta.csv"
    texts_path = emb_dir / "clean_texts.txt"
    if not vec_path.exists() or not meta_path.exists() or not texts_path.exists():
        raise FileNotFoundError(f"Missing embeddings assets in {emb_dir}. Run scripts/build_embeddings.py first.")
    vectors = np.load(vec_path)
    meta = pd.read_csv(meta_path)
    texts = texts_path.read_text(encoding="utf-8").splitlines()
    return vectors, meta, texts

def build_index(vectors):
    """Build sklearn NearestNeighbors index (cosine)."""
    nn = NearestNeighbors(n_neighbors=min(10, len(vectors)), metric="cosine")
    nn.fit(vectors)
    return nn

def canonicalize_complaint(label):
    """Map model output variants to canonical complaint types."""
    import re
    if not label or not isinstance(label, str):
        return "Other"
    s = label.strip().lower()
    # heuristics
    if re.search(r"bill|payment|refund|invoice|charge", s):
        return "Billing"
    if re.search(r"login|auth|password|signin|sign-in|authentication", s):
        return "Login"
    if re.search(r"deliver|shipment|tracking|order not|not arrived", s):
        return "Delivery"
    if re.search(r"product|defect|bug|crash|issue|error|ui|ux", s):
        return "Product Defect"
    if re.search(r"perf|latency|slow|lag|timeout|responsive", s):
        return "Performance"
    if re.search(r"customer|support|service|agent|rep", s):
        return "Customer Service"
    # split on separators and try parts
    if "/" in label or "|" in label or ";" in label:
        parts = [p.strip() for p in re.split(r"[\/\|\;]", label)]
        for p in parts:
            mapped = canonicalize_complaint(p)
            if mapped != "Other":
                return mapped
    return "Other"

def canonicalize_simple(val, allowed):
    """Validate allowed categorical outputs, else return 'Unknown'."""
    if not isinstance(val, str):
        return "Unknown"
    v = val.strip()
    return v if v in allowed else "Unknown"

def robust_extract_function_arguments(resp):
    """
    Extract the function_call.arguments JSON string robustly from resp.
    Works for both object-like response and dict-like resp.to_dict().
    Returns parsed dict or None.
    """
    func_args = None
    parsed = None
    try:
        # try attribute path common in new client
        msg = resp.choices[0].message
        # attribute style
        if hasattr(msg, "function_call") and getattr(msg, "function_call") is not None:
            fc = getattr(msg, "function_call")
            if hasattr(fc, "arguments"):
                func_args = getattr(fc, "arguments")
            elif isinstance(fc, dict) and "arguments" in fc:
                func_args = fc["arguments"]
        # fallback to dict
        if func_args is None:
            try:
                d = resp.to_dict() if hasattr(resp, "to_dict") else dict(resp)
                func_args = d["choices"][0]["message"].get("function_call", {}).get("arguments")
            except Exception:
                func_args = None
        if func_args:
            # sometimes already a dict; sometimes a JSON string
            if isinstance(func_args, str):
                parsed = json.loads(func_args)
            elif isinstance(func_args, dict):
                parsed = func_args
            else:
                # last resort: try to json.loads the stringified version
                parsed = json.loads(str(func_args))
    except Exception:
        parsed = None
    return parsed

def embed_texts(client, texts, model):
    """Call OpenAI embeddings.create in a single batch and return list of vectors."""
    resp = client.embeddings.create(model=model, input=texts)
    return [r.embedding for r in resp.data]

# -------------------------
# Main pipeline
# -------------------------
def main(args):
    tickets_path = Path(args.tickets)
    labels_path = Path(args.labels)
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    k = int(args.k)
    batch = int(args.batch)
    model = args.model
    emb_model = args.embedding_model

    out_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Loading tickets:", tickets_path)
    df_t = pd.read_parquet(tickets_path)
    print("[+] Loading labels:", labels_path)
    df_lab = pd.read_csv(labels_path, low_memory=False)

    print("[+] Loading embeddings from", emb_dir)
    vectors, meta, lab_texts = load_embeddings(emb_dir)
    nn = build_index(vectors)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in env (export OPENAI_API_KEY=...)")
    client = OpenAI(api_key=api_key)

    # We'll run over the labeled set for evaluation. Change to df_t for full dataset if desired.
    df = df_lab.copy().reset_index(drop=True)
    n = df.shape[0]
    print(f"[+] Running RAG classify for {n} rows; top_k={k}")

    results = []

    for i in range(0, n, batch):
        batch_df = df.iloc[i:i+batch]
        q_texts = batch_df["clean_text"].fillna("").astype(str).tolist()
        print(f"[+] Embedding + processing batch {i}..{i+len(q_texts)-1}")

        # compute embeddings for queries
        try:
            q_embs = embed_texts(client, q_texts, emb_model)
        except Exception as e:
            print("[!] Embedding call failed for queries:", e)
            # fall back: skip this batch
            for j in range(len(q_texts)):
                results.append({
                    "index": i+j,
                    "ticket_id": batch_df.iloc[j].get("ticket_id") if "ticket_id" in batch_df.columns else None,
                    "input_text": q_texts[j],
                    "complaint_pred_raw": None,
                    "complaint_pred": "Other",
                    "severity_pred": "Unknown",
                    "sentiment_pred": "Unknown",
                    "action_required_pred": "Unknown",
                    "root_cause_pred": "Unknown",
                    "raw_response": None,
                    "error": str(e)
                })
            continue

        for j, qv in enumerate(q_embs):
            global_idx = i + j
            # find nearest labeled examples
            try:
                dists, ids = nn.kneighbors([qv], n_neighbors=min(k, len(vectors)))
                ids = ids[0].tolist()
            except Exception as e:
                print("[!] NN lookup failed:", e)
                ids = []

            # build few-shot examples using meta -> map to original label rows
            few_shot = ""
            for ex_id in ids:
                # meta row_index should point to original df_lab row
                try:
                    meta_row = meta.iloc[ex_id]
                    row_index = int(meta_row["row_index"])
                    lab_row = df_lab.iloc[row_index]
                    ex_text = lab_texts[ex_id]
                    ex_json = {
                        "complaint_type": str(lab_row.get("label_complaint_type", "Other")),
                        "severity": str(lab_row.get("label_severity", "Unknown")),
                        "sentiment": str(lab_row.get("label_sentiment", "Unknown")),
                        "root_cause": str(lab_row.get("label_root_cause", "Unknown")) if "label_root_cause" in lab_row else "Unknown",
                        "action_required": str(lab_row.get("label_action_required", "Unknown")) if "label_action_required" in lab_row else "Unknown"
                    }
                    few_shot += EXAMPLE_SNIPPET.format(text=ex_text.replace("```",""), json=json.dumps(ex_json))
                except Exception:
                    # if anything goes wrong, skip that example
                    continue

            prompt = PROMPT_HEADER + "\n" + few_shot + "\nNow classify:\n```" + q_texts[j].replace("```","") + "```\n"

            # call LLM with function-calling
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role":"user","content":prompt}],
                    functions=[FUNCTION_SCHEMA],
                    function_call={"name":"classify_ticket"},
                    temperature=0,
                    max_tokens=300,
                )
            except Exception as e:
                # log failure and continue
                results.append({
                    "index": global_idx,
                    "ticket_id": int(batch_df.iloc[j].get("ticket_id")) if "ticket_id" in batch_df.columns else None,
                    "input_text": q_texts[j],
                    "complaint_pred_raw": None,
                    "complaint_pred": "Other",
                    "severity_pred": "Unknown",
                    "sentiment_pred": "Unknown",
                    "action_required_pred": "Unknown",
                    "root_cause_pred": "Unknown",
                    "raw_response": None,
                    "error": str(e)
                })
                continue

            # robust extraction of function_call.arguments
            parsed = robust_extract_function_arguments(resp)

            # canonicalize parsed output
            complaint_raw = parsed.get("complaint_type") if parsed and parsed.get("complaint_type") else None
            complaint_pred = canonicalize_complaint(complaint_raw)
            severity_pred = canonicalize_simple(parsed.get("severity") if parsed and parsed.get("severity") else None, ["Low","Medium","High"])
            sentiment_pred = canonicalize_simple(parsed.get("sentiment") if parsed and parsed.get("sentiment") else None, ["Positive","Neutral","Negative"])
            action_pred = canonicalize_simple(parsed.get("action_required") if parsed and parsed.get("action_required") else None, ["Yes","No"])
            root_pred = parsed.get("root_cause") if parsed and parsed.get("root_cause") else "Unknown"

            # prepare serializable raw_response
            try:
                raw_response = resp.to_dict() if hasattr(resp, "to_dict") else str(resp)
            except Exception:
                raw_response = str(resp)

            out = {
                "index": global_idx,
                "ticket_id": int(batch_df.iloc[j].get("ticket_id")) if "ticket_id" in batch_df.columns else None,
                "input_text": q_texts[j],
                "complaint_pred_raw": complaint_raw,
                "complaint_pred": complaint_pred,
                "severity_pred": severity_pred,
                "sentiment_pred": sentiment_pred,
                "action_required_pred": action_pred,
                "root_cause_pred": root_pred,
                "raw_response": raw_response
            }
            results.append(out)

            # tiny throttle
            time.sleep(0.25)

    # Save outputs
    df_res = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    df_res.to_parquet(out_dir / "llm_rag_responses.parquet", index=False)
    df_res.to_csv(out_dir / "llm_rag_responses.csv", index=False)

    # Evaluation (if labels present)
    eval_df = df.copy().reset_index(drop=True).merge(df_res, left_index=True, right_on="index", how="left")
    metrics = {}
    if "label_complaint_type" in eval_df.columns:
        y_true = eval_df["label_complaint_type"].fillna("Unknown")
        y_pred = eval_df["complaint_pred"].fillna("Other")
        metrics["complaint_report"] = classification_report(y_true, y_pred, output_dict=True)
        metrics["complaint_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["complaint_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    if "label_severity" in eval_df.columns:
        y_true = eval_df["label_severity"].fillna("Unknown")
        y_pred = eval_df["severity_pred"].fillna("Unknown")
        metrics["severity_report"] = classification_report(y_true, y_pred, output_dict=True)
        metrics["severity_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["severity_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    if "label_sentiment" in eval_df.columns:
        y_true = eval_df["label_sentiment"].fillna("Unknown")
        y_pred = eval_df["sentiment_pred"].fillna("Unknown")
        metrics["sentiment_report"] = classification_report(y_true, y_pred, output_dict=True)
        metrics["sentiment_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["sentiment_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))

    with open(out_dir / "llm_rag_summary.json", "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("[+] Done. Artifacts in", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickets", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--emb_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--k", default=3, type=int)
    p.add_argument("--batch", default=20, type=int)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--embedding_model", default="text-embedding-3-small")
    args = p.parse_args()
    main(args)

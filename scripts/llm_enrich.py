#!/usr/bin/env python3
"""
scripts/llm_enrich.py

Usage:
  python scripts/llm_enrich.py \
    --tickets data/clean/tickets.parquet \
    --labels data/labels/human_labeled_clean.csv \
    --prompt prompts/prompt_v1.txt \
    --out_dir results/llm \
    --model gpt-4o-mini \
    --batch 20

Requires OPENAI_API_KEY in env.
"""

import os, time, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np
from openai import OpenAI
from sklearn.metrics import classification_report, accuracy_score, f1_score
import joblib

# helper: safe JSON parse
def safe_parse_json(s):
    try:
        return json.loads(s)
    except Exception:
        # try to extract JSON substring
        import re
        m = re.search(r"(\{.*\})", s, flags=re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except Exception:
                return None
        return None

def call_llm_batch(texts, prompt_template, model="gpt-4o-mini"):
    """
    texts: list[str]
    returns: list[dict]
    Uses OpenAI client and ensures `meta` is JSON-serializable.
    """
    responses = []
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in env")
    client = OpenAI(api_key=api_key)

    for t in texts:
        prompt = prompt_template.replace("{text}", t)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role":"user","content":prompt}],
                temperature=0,
                max_tokens=300,
            )

            # extract text safely (new client returns objects with .message.content)
            try:
                txt = resp.choices[0].message.content
            except Exception:
                # fallback to string representation
                txt = str(resp)

            parsed = safe_parse_json(txt)

            # normalize usage/meta into a plain dict that pyarrow can serialize
            meta_raw = None
            try:
                meta_raw = resp.usage if hasattr(resp, "usage") else (resp["usage"] if isinstance(resp, dict) and "usage" in resp else None)
            except Exception:
                meta_raw = None

            # Convert to JSON-serializable dict: dump with default=str then load back to dict
            try:
                meta = json.loads(json.dumps(meta_raw, default=str))
            except Exception:
                # worst-case: store a string
                meta = str(meta_raw)

            responses.append({"raw": txt, "parsed": parsed, "meta": meta})
        except Exception as e:
            responses.append({"raw": None, "parsed": None, "error": str(e)})
        time.sleep(0.35)
    return responses

def main(args):
    tickets_path = Path(args.tickets)
    labels_path = Path(args.labels)
    prompt_path = Path(args.prompt)
    out_dir = Path(args.out_dir)
    model = args.model
    batch = int(args.batch)

    out_dir.mkdir(parents=True, exist_ok=True)
    # load data
    print("[+] loading tickets:", tickets_path)
    df_t = pd.read_parquet(tickets_path)
    print("[+] loading labels:", labels_path)
    df_lab = pd.read_csv(labels_path, low_memory=False)

    # unify by ticket_id if present
    if "ticket_id" in df_lab.columns and "ticket_id" in df_t.columns:
        df = df_lab.merge(df_t[["ticket_id","clean_text"]], on="ticket_id", how="left")
    else:
        # join by index on labeled subset only
        df = df_lab.copy()
        # ensure clean_text exists
        if "clean_text" not in df.columns:
            if "clean_text" in df_t.columns:
                df = df.merge(df_t[["clean_text"]], left_index=True, right_index=True, how="left")
            else:
                raise SystemExit("No clean_text available in labels or tickets")
    # load prompt
    prompt_template = prompt_path.read_text(encoding="utf-8")

    results = []
    n = df.shape[0]
    print(f"[+] Will call LLM for {n} labeled tickets (batched {batch})")
    for i in range(0, n, batch):
        batch_df = df.iloc[i:i+batch]
        texts = batch_df["clean_text"].fillna("").astype(str).tolist()
        print(f"[+] Calling batch {i}..{i+len(texts)-1}")
        batch_resps = call_llm_batch(texts, prompt_template, model=model)
        for idx, row in enumerate(batch_df.itertuples(index=False)):
            r = batch_resps[idx]
            out = {
                "index": int(i+idx),
                "ticket_id": row.ticket_id if "ticket_id" in df.columns else None,
                "input_text": texts[idx],
                "llm_raw": r.get("raw"),
                "llm_parsed": r.get("parsed"),
                "llm_error": r.get("error") if "error" in r else None,
                "meta": r.get("meta")
            }
            results.append(out)
    # build DataFrame
    df_res = pd.DataFrame(results)
    # normalize parsed JSON into columns
    def extract_field(parsed, key):
        if not isinstance(parsed, dict): return None
        return parsed.get(key)
    df_res["complaint_type_pred"] = df_res["llm_parsed"].apply(lambda x: extract_field(x,"complaint_type") if x else None)
    df_res["severity_pred"] = df_res["llm_parsed"].apply(lambda x: extract_field(x,"severity") if x else None)
    df_res["sentiment_pred"] = df_res["llm_parsed"].apply(lambda x: extract_field(x,"sentiment") if x else None)
    df_res["root_cause_pred"] = df_res["llm_parsed"].apply(lambda x: extract_field(x,"root_cause") if x else None)
    df_res["action_required_pred"] = df_res["llm_parsed"].apply(lambda x: extract_field(x,"action_required") if x else None)

    # save raw results
    df_res.to_parquet(out_dir / "llm_responses.parquet", index=False)
    df_res.to_csv(out_dir / "llm_responses.csv", index=False)

    # evaluation: compare to human labels for the rows where ground-truth exists
    eval_df = df.merge(df_res[["index","complaint_type_pred","severity_pred","sentiment_pred","action_required_pred"]], left_index=True, right_on="index", how="left")
    # ensure column names for truth exist
    truth_cols = {}
    for k in ["label_complaint_type","label_severity","label_sentiment","label_action_required"]:
        truth_cols[k] = k if k in eval_df.columns else None

    metrics = {}
    from sklearn.metrics import classification_report, f1_score, accuracy_score
    # complaint type
    if truth_cols["label_complaint_type"]:
        y_true = eval_df["label_complaint_type"].fillna("NULL")
        y_pred = eval_df["complaint_type_pred"].fillna("NULL")
        metrics["complaint_report"] = classification_report(y_true, y_pred, output_dict=True)
        metrics["complaint_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["complaint_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    if truth_cols["label_severity"]:
        y_true = eval_df["label_severity"].fillna("NULL")
        y_pred = eval_df["severity_pred"].fillna("NULL")
        metrics["severity_report"] = classification_report(y_true, y_pred, output_dict=True)
        metrics["severity_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["severity_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))
    if truth_cols["label_sentiment"]:
        y_true = eval_df["label_sentiment"].fillna("NULL")
        y_pred = eval_df["sentiment_pred"].fillna("NULL")
        metrics["sentiment_report"] = classification_report(y_true, y_pred, output_dict=True)
        metrics["sentiment_acc"] = float(accuracy_score(y_true, y_pred))
        metrics["sentiment_macro_f1"] = float(f1_score(y_true, y_pred, average="macro"))

    # save metrics
    with open(out_dir / "llm_summary.json","w") as fh:
        json.dump(metrics, fh, indent=2)

    # also save a CSV table for per-sample comparison
    compare_cols = ["ticket_id","clean_text","label_complaint_type","complaint_type_pred","label_severity","severity_pred","label_sentiment","sentiment_pred","label_action_required","action_required_pred","root_cause_pred"]
    # take columns present
    cmp = eval_df[[c for c in compare_cols if c in eval_df.columns or c in eval_df.columns or c in eval_df.columns]]
    cmp.to_csv(out_dir / "llm_evaluation.csv", index=False)
    print("[+] Done. Artifacts in", out_dir)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--tickets", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--prompt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--batch", default=20)
    args = p.parse_args()
    main(args)

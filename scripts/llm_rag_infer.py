#!/usr/bin/env python3
"""
scripts/llm_rag_infer.py

Run Retrieval-Augmented Generation (RAG) inference over the full tickets dataset.

Saves enriched tickets to out_dir/tickets_enriched.parquet and tickets_enriched.csv.

Features:
 - Uses existing embeddings (data/embeddings) and NearestNeighbors retrieval.
 - Robust extraction of function_call.arguments for OpenAI responses.
 - Per-batch checkpointing to out_dir/partial/ so runs are resumable.
 - --limit to test on subset.
 - --cheap-mode to reduce k and shorten prompts for cost control.

Example dry-run (100 rows):
python3 scripts/llm_rag_infer.py \
  --tickets data/clean/tickets.parquet \
  --emb_dir data/embeddings \
  --out_dir results/llm_full \
  --k 5 --batch 20 --limit 100

Example full run:
python3 scripts/llm_rag_infer.py \
  --tickets data/clean/tickets.parquet \
  --emb_dir data/embeddings \
  --out_dir results/llm_full \
  --k 5 --batch 20
"""
import os
import json
import time
import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from openai import OpenAI

# -------------------------
# Prompt fragments
# -------------------------
PROMPT_HEADER_FULL = """You are a ticket classification assistant. Output ONLY a single JSON object with fields:
{{"complaint_type","severity","sentiment","root_cause","action_required"}}.
Allowed complaint_type: ["Billing","Login","Delivery","Product Defect","Performance","Customer Service","Other"].
Allowed severity: ["Low","Medium","High"].
Allowed sentiment: ["Positive","Neutral","Negative"].
action_required: "Yes" or "No".
root_cause: short phrase (1-6 words) or "Unknown".

Provide concise answers. Use reasoning only if necessary. Return only the structured JSON via a function call (no extra text).
"""

# Shorter prompt for cheap mode (less context)
PROMPT_HEADER_CHEAP = """You are a structured classification assistant for customer support tickets.

Your task is to classify each ticket into exactly ONE complaint type, ONE severity level, and ONE sentiment.

---

ALLOWED COMPLAINT TYPES:
- Product Defect
- Login
- Billing
- Customer Service
- Delivery
- Performance
- Other

Definitions:

Product Defect:
Hardware malfunction, physical issue, device not working, crashes, overheating, broken components.

Login:
Authentication issues, password errors, account locked, cannot sign in.

Billing:
Charges, refunds, payment failures, incorrect billing, subscription issues.

Customer Service:
Complaints about support agents, poor service, long wait times.

Delivery:
Shipping delays, missing packages, damaged deliveries.

Performance:
App slow, lag, latency, timeout, responsiveness issues.

Other:
Anything that does not clearly match the above.

---

ALLOWED SEVERITY:
- High: Core functionality broken, account inaccessible, payment failed, data loss.
- Medium: Product issue but system partially usable.
- Low: Minor inconvenience or informational request.

---

ALLOWED SENTIMENT:
- Negative
- Neutral
- Positive

---

IMPORTANT RULES:
- Choose ONLY from allowed complaint types.
- Do NOT invent new categories.
- If unsure between two, choose the most specific one.
- If ticket clearly describes hardware malfunction, it MUST be Product Defect.
- If ticket mentions login/authentication failure, it MUST be Login.
- If ticket mentions charges/refund/payment, it MUST be Billing.

---

Return ONLY valid JSON with the following format:

{
  "complaint_type": "...",
  "severity": "...",
  "sentiment": "...",
  "root_cause": "...",
  "action_required": "Yes" or "No"
}

Do not include explanations.
"""

EXAMPLE_SNIPPET = """Ticket:
```{text}```
Output:
{json}
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
    vec_path = emb_dir / "embeddings.npy"
    meta_path = emb_dir / "meta.csv"
    texts_path = emb_dir / "clean_texts.txt"
    if not vec_path.exists() or not meta_path.exists() or not texts_path.exists():
        raise FileNotFoundError(f"Missing embeddings assets in {emb_dir}. Run build_embeddings first.")
    vectors = np.load(vec_path)
    meta = pd.read_csv(meta_path)
    texts = texts_path.read_text(encoding="utf-8").splitlines()
    return vectors, meta, texts

def build_index(vectors):
    nn = NearestNeighbors(n_neighbors=min(10, len(vectors)), metric="cosine")
    nn.fit(vectors)
    return nn

def canonicalize_complaint(label: Optional[str]) -> str:
    import re
    if not label or not isinstance(label, str):
        return "Other"
    s = label.strip().lower()
    if re.search(r"bill|payment|refund|invoice|charge", s):
        return "Billing"
    if re.search(r"login|auth|password|signin|sign-in|authentication", s):
        return "Login"
    if re.search(r"deliver|shipment|tracking|order not|not arrived|not received", s):
        return "Delivery"
    if re.search(r"product|defect|bug|crash|error|ui|issue", s):
        return "Product Defect"
    if re.search(r"perf|latency|slow|lag|timeout|responsive", s):
        return "Performance"
    if re.search(r"customer|support|service|agent|rep", s):
        return "Customer Service"
    if "/" in label or "|" in label:
        parts = [p.strip() for p in label.split("/") + label.split("|")]
        for p in parts:
            mapped = canonicalize_complaint(p)
            if mapped != "Other":
                return mapped
    return "Other"

def canonicalize_simple(val, allowed):
    if not isinstance(val, str):
        return "Unknown"
    v = val.strip()
    return v if v in allowed else "Unknown"

def robust_extract_function_arguments(resp):
    """Defensive extractor for function_call.arguments across SDK shapes and raw dict strings."""
    import json, ast

    # Try to extract from resp.to_dict() first (most consistent)
    try:
        d = resp.to_dict() if hasattr(resp, "to_dict") else (resp if isinstance(resp, dict) else None)
        if isinstance(d, dict):
            choices = d.get("choices") or d.get("choices", [])
            if choices and isinstance(choices, (list, tuple)):
                msg = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
                # message may include function_call
                fc = msg.get("function_call") if isinstance(msg, dict) else None
                if fc:
                    args = fc.get("arguments") if isinstance(fc, dict) else None
                    if isinstance(args, str):
                        try:
                            return json.loads(args)
                        except Exception:
                            try:
                                return ast.literal_eval(args)
                            except Exception:
                                return None
                    if isinstance(args, dict):
                        return args
    except Exception:
        pass

    # Next try attribute-style access (SDK object)
    try:
        # resp.choices[0].message.function_call.arguments (various shapes)
        c = getattr(resp, "choices", None)
        if c:
            msg = c[0].message if hasattr(c[0], "message") else None
            if msg:
                fc = getattr(msg, "function_call", None)
                if fc is not None:
                    args = getattr(fc, "arguments", None)
                    if isinstance(args, str):
                        try:
                            return json.loads(args)
                        except Exception:
                            try:
                                return ast.literal_eval(args)
                            except Exception:
                                return None
                    if isinstance(args, dict):
                        return args
                # sometimes message itself is a dict-like
                if isinstance(msg, dict):
                    fc = msg.get("function_call")
                    if fc:
                        args = fc.get("arguments")
                        if isinstance(args, str):
                            try:
                                return json.loads(args)
                            except Exception:
                                try:
                                    return ast.literal_eval(args)
                                except Exception:
                                    return None
                        if isinstance(args, dict):
                            return args
    except Exception:
        pass

    # Last resort: try to parse resp as string and search for "function_call" manually
    try:
        s = str(resp)
        # crude find of JSON substring
        start = s.find('"function_call"')
        if start != -1:
            # attempt to locate first subsequent '{' and parse JSON chunk
            brace = s.find('{', start)
            if brace != -1:
                cand = s[brace:]
                # try progressively shorter slices up to some length
                for L in (2000,1500,1000,800,500):
                    try:
                        j = json.loads(cand[:L])
                        # dig for arguments
                        choices = j.get("choices", [])
                        if choices:
                            msg = choices[0].get("message", {})
                            fc = msg.get("function_call", {})
                            args = fc.get("arguments")
                            if isinstance(args, str):
                                try:
                                    return json.loads(args)
                                except Exception:
                                    try:
                                        return ast.literal_eval(args)
                                    except Exception:
                                        return None
                            if isinstance(args, dict):
                                return args
                    except Exception:
                        continue
    except Exception:
        pass

    return None

# -------------------------
# Main inference loop
# -------------------------
def main(args):
    tickets_path = Path(args.tickets)
    emb_dir = Path(args.emb_dir)
    out_dir = Path(args.out_dir)
    k = int(args.k)
    batch = int(args.batch)
    limit = int(args.limit) if args.limit and int(args.limit) > 0 else None
    cheap_mode = args.cheap_mode
    model = args.model
    emb_model = args.embedding_model

    out_dir.mkdir(parents=True, exist_ok=True)
    partial_dir = out_dir / "partial"
    partial_dir.mkdir(parents=True, exist_ok=True)

    # load tickets (all)
    print("[+] Loading tickets:", tickets_path)
    df_t = pd.read_parquet(tickets_path).reset_index(drop=True)
    if limit:
        df_t = df_t.iloc[:limit].reset_index(drop=True)
        print(f"[+] LIMIT enabled: processing first {limit} rows")

    # load embeddings for retrieval (from labeled examples)
    print("[+] Loading label embeddings from", emb_dir)
    vectors, meta, lab_texts = load_embeddings(emb_dir)
    nn = build_index(vectors)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Set OPENAI_API_KEY in env (export OPENAI_API_KEY=...)")
    client = OpenAI(api_key=api_key)

    n = df_t.shape[0]
    print(f"[+] Processing {n} tickets (batch={batch}, k={k}, cheap_mode={cheap_mode})")

    processed_indices = set()
    # resume: find existing partial files and mark indices as processed
    for p in sorted(partial_dir.glob("partial_*.parquet")):
        try:
            d = pd.read_parquet(p)
            processed_indices.update(d["__infer_index"].astype(int).tolist())
            print(f"[+] Found checkpoint {p.name} with {len(d)} rows; marking indexes done")
        except Exception:
            continue

    results_rows = []

    # main loop
    for start in range(0, n, batch):
        end = min(start + batch, n)
        idxs = list(range(start, end))
        # skip batch if fully processed
        if all(i in processed_indices for i in idxs):
            print(f"[+] Skipping batch {start}..{end-1} (already processed)")
            continue

        batch_df = df_t.iloc[start:end].reset_index(drop=False)
        # 'index' column from reset_index is the original row number -> keep it as __infer_index
        batch_df = batch_df.rename(columns={"index": "__infer_index"})
        q_texts = batch_df["clean_text"].fillna("").astype(str).tolist()

        # embed queries
        try:
            resp_emb = client.embeddings.create(model=emb_model, input=q_texts)
            q_embs = [r.embedding for r in resp_emb.data]
        except Exception as e:
            print("[!] Embedding call failed for batch", start, "error:", e)
            # if embeddings fail, skip this batch but write placeholder rows
            partial_out = []
            for j, txt in enumerate(q_texts):
                ix = int(batch_df.loc[j, "__infer_index"])
                partial_out.append({
                    "__infer_index": ix,
                    "ticket_id": batch_df.loc[j].get("ticket_id") if "ticket_id" in batch_df.columns else None,
                    "input_text": txt,
                    "complaint_pred_raw": None,
                    "complaint_pred": "Other",
                    "severity_pred": "Unknown",
                    "sentiment_pred": "Unknown",
                    "action_required_pred": "Unknown",
                    "root_cause_pred": "Unknown",
                    "raw_response": None,
                    "error": str(e)
                })
            pd.DataFrame(partial_out).to_parquet(partial_dir / f"partial_{start}_{end-1}.parquet", index=False)
            print(f"[!] Wrote placeholder partial for batch {start}..{end-1}")
            # mark processed indices and continue
            processed_indices.update(idxs)
            continue

        batch_out = []
        for j, qv in enumerate(q_embs):
            global_idx = int(batch_df.loc[j, "__infer_index"])
            txt = q_texts[j]
            # retrieval
            try:
                dists, ids = nn.kneighbors([qv], n_neighbors=min(k, len(vectors)))
                ids = ids[0].tolist()
            except Exception as e:
                print("[!] NN lookup failed at idx", global_idx, "error:", e)
                ids = []

            few_shot = ""
            # Build few-shot examples from retrieved ids (limit the size to control prompt length)
            for ex_id in ids:
                try:
                    meta_row = meta.iloc[ex_id]
                    row_index = int(meta_row["row_index"])
                    ex_text = lab_texts[ex_id]
                    # read things from meta/labels is limited here; include minimal example JSON
                    # We'll avoid adding verbose root_cause in examples to reduce prompt size
                    ex_json = {
                        "complaint_type": str(meta_row.get("label_complaint_type", meta_row.get("label", "Other"))) if "label_complaint_type" in meta_row else str(meta_row.get("label","Other")),
                        "severity": str(meta_row.get("label_severity","Unknown")) if "label_severity" in meta_row else "Unknown",
                        "sentiment": str(meta_row.get("label_sentiment","Unknown")) if "label_sentiment" in meta_row else "Unknown",
                        "root_cause": "Unknown",
                        "action_required": "Yes"
                    }
                    few_shot += EXAMPLE_SNIPPET.format(text=ex_text.replace("```",""), json=json.dumps(ex_json))
                except Exception:
                    continue

            prompt_header = PROMPT_HEADER_CHEAP if cheap_mode else PROMPT_HEADER_FULL
            prompt = prompt_header + "\n" + few_shot + "\nNow classify:\n```" + txt.replace("```","") + "```\n"

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
                # log the error and write a safe fallback
                batch_out.append({
                    "__infer_index": global_idx,
                    "ticket_id": batch_df.loc[j].get("ticket_id") if "ticket_id" in batch_df.columns else None,
                    "input_text": txt,
                    "complaint_pred_raw": None,
                    "complaint_pred": "Other",
                    "severity_pred": "Unknown",
                    "sentiment_pred": "Unknown",
                    "action_required_pred": "Unknown",
                    "root_cause_pred": "Unknown",
                    "raw_response": None,
                    "error": str(e)
                })
                # small sleep and continue
                time.sleep(0.25)
                continue

            parsed = robust_extract_function_arguments(resp)

            complaint_raw = parsed.get("complaint_type") if parsed and parsed.get("complaint_type") else None
            complaint_pred = canonicalize_complaint(complaint_raw)
            severity_pred = canonicalize_simple(parsed.get("severity") if parsed and parsed.get("severity") else None, ["Low","Medium","High"])
            sentiment_pred = canonicalize_simple(parsed.get("sentiment") if parsed and parsed.get("sentiment") else None, ["Positive","Neutral","Negative"])
            action_pred = canonicalize_simple(parsed.get("action_required") if parsed and parsed.get("action_required") else None, ["Yes","No"])
            root_pred = parsed.get("root_cause") if parsed and parsed.get("root_cause") else "Unknown"

            # serializable raw_response
            try:
                raw_response = resp.to_dict() if hasattr(resp, "to_dict") else str(resp)
            except Exception:
                raw_response = str(resp)

            batch_out.append({
                "__infer_index": global_idx,
                "ticket_id": batch_df.loc[j].get("ticket_id") if "ticket_id" in batch_df.columns else None,
                "input_text": txt,
                "complaint_pred_raw": complaint_raw,
                "complaint_pred": complaint_pred,
                "severity_pred": severity_pred,
                "sentiment_pred": sentiment_pred,
                "action_required_pred": action_pred,
                "root_cause_pred": root_pred,
                "raw_response": raw_response
            })

            # throttle
            time.sleep(0.2 if not cheap_mode else 0.15)

        # write partial for this batch
        df_partial = pd.DataFrame(batch_out)
        partial_path = partial_dir / f"partial_{start}_{end-1}.parquet"
        df_partial.to_parquet(partial_path, index=False)
        print(f"[+] Wrote partial results for batch {start}..{end-1} ({len(df_partial)} rows) -> {partial_path.name}")

        processed_indices.update(idxs)

    # assemble all partials in order
    parts = sorted(partial_dir.glob("partial_*.parquet"))
    all_rows = []
    for p in parts:
        try:
            d = pd.read_parquet(p)
            all_rows.append(d)
        except Exception:
            continue
    if all_rows:
        df_out = pd.concat(all_rows, ignore_index=True).sort_values("__infer_index").reset_index(drop=True)
    else:
        df_out = pd.DataFrame(columns=[
            "__infer_index","ticket_id","input_text","complaint_pred_raw","complaint_pred",
            "severity_pred","sentiment_pred","action_required_pred","root_cause_pred","raw_response"
        ])

    # join predictions back to original tickets (by index)
    df_out = df_out.rename(columns={"__infer_index":"index_tmp"})
    df_t_full = pd.read_parquet(tickets_path).reset_index(drop=True)
    # if limit was used, df_t was subset; we already processed indices relative to the subset; ensure join logic consistent:
    if limit:
        # subset indices were 0..limit-1
        df_t_use = df_t_full.iloc[:limit].reset_index(drop=True)
    else:
        df_t_use = df_t_full.reset_index(drop=True)

    # merge by position/index
    df_t_use = df_t_use.reset_index().rename(columns={"index":"index_tmp"})
    enriched = df_t_use.merge(df_out, on="index_tmp", how="left", suffixes=("","_pred"))
    # rename prediction columns to nice names
    enriched = enriched.rename(columns={
        "complaint_pred":"pred_complaint_type",
        "severity_pred":"pred_severity",
        "sentiment_pred":"pred_sentiment",
        "action_required_pred":"pred_action_required",
        "root_cause_pred":"pred_root_cause",
        "complaint_pred_raw":"pred_complaint_raw"
    })

    # drop helper column
    enriched = enriched.drop(columns=["index_tmp"], errors="ignore")

    # save final artifacts
    out_parquet = out_dir / "tickets_enriched.parquet"
    out_csv = out_dir / "tickets_enriched.csv"
    enriched.to_parquet(out_parquet, index=False)
    enriched.to_csv(out_csv, index=False)
    # write progress summary
    summary = {
        "n_input_tickets": int(df_t_full.shape[0]) if not limit else int(limit),
        "n_processed": int(df_out.shape[0]),
        "out_parquet": str(out_parquet),
        "out_csv": str(out_csv)
    }
    with open(out_dir / "inference_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    print("[+] Finished. Results saved to:", out_parquet, out_csv)
    print("[+] Summary:", summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickets", required=True, help="Path to cleaned tickets parquet")
    parser.add_argument("--emb_dir", required=True, help="Path to embeddings (embeddings.npy, meta.csv, clean_texts.txt)")
    parser.add_argument("--out_dir", required=True, help="Directory to write results")
    parser.add_argument("--k", type=int, default=5, help="Top-k retrieved examples")
    parser.add_argument("--batch", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--limit", type=int, default=0, help="If >0, only process this many rows (dry-run)")
    parser.add_argument("--model", default="gpt-4o-mini", help="LLM model to use for classification")
    parser.add_argument("--embedding_model", default="text-embedding-3-small", help="Embedding model")
    parser.add_argument("--cheap_mode", action="store_true", help="Use cheaper prompt / smaller throttle")
    args = parser.parse_args()
    main(args)
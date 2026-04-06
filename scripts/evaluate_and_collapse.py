#!/usr/bin/env python3
"""
scripts/evaluate_and_collapse.py

Usage:
python3 scripts/evaluate_and_collapse.py \
  --labels data/labels/human_labeled_clean.csv \
  --out_dir results/eval_reports \
  --zero_shot results/llm/llm_responses.parquet \
  --rag results/llm_rag/llm_rag_responses.parquet \
  --rag_k5 results/llm_rag_k5/llm_rag_responses.parquet \
  --baseline results/baseline/baseline_preds.csv

Any of the prediction files may be absent; the script will compute metrics for the ones it finds.

Outputs:
- results/eval_reports/summary_all.json
- results/eval_reports/<run>_classification_report.json
- results/eval_reports/<run>_confusion.csv
"""
import argparse, json, os
from pathlib import Path
import pandas as pd
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix

CANONICAL_LABELS = [
    "Product Defect",
    "Login",
    "Billing",
    "Customer Service",
    "Delivery",
    "Performance",
    "Other"
]

def collapse_label(label: str):
    """Map many variants to canonical labels."""
    if not isinstance(label, str):
        return "Other"
    s = label.strip().lower()
    # billing/payment
    if any(tok in s for tok in ["bill", "payment", "refund", "invoice", "charge"]):
        return "Billing"
    # login/auth
    if any(tok in s for tok in ["login", "auth", "password", "signin", "sign-in", "authentication"]):
        return "Login"
    # delivery/shipping
    if any(tok in s for tok in ["deliver", "shipment", "tracking", "order not", "not arrived", "not received"]):
        return "Delivery"
    # performance
    if any(tok in s for tok in ["perf", "latency", "slow", "lag", "timeout", "responsive"]):
        return "Performance"
    # customer service
    if any(tok in s for tok in ["customer", "support", "service", "agent", "rep", "representative"]):
        return "Customer Service"
    # product defect
    if any(tok in s for tok in ["product", "defect", "bug", "crash", "error", "ui", "issue"]):
        return "Product Defect"
    # else Other
    return "Other"

def safe_load_preds(path: Path, default_pred_col_candidates):
    """
    Load predictions from a parquet/csv path and try to identify the complaint-type column.
    Returns DataFrame with a column 'complaint_pred_raw' if found.
    """
    if not path.exists():
        return None
    if path.suffix in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)
    # candidate column names commonly produced by scripts
    for cand in default_pred_col_candidates:
        if cand in df.columns:
            df = df.rename(columns={cand: "complaint_pred_raw"})
            return df
    # If the file already uses 'complaint_pred' or 'label_complaint_type', adapt
    if "complaint_pred" in df.columns:
        df = df.rename(columns={"complaint_pred":"complaint_pred_raw"})
        return df
    if "label_complaint_type" in df.columns:
        df = df.rename(columns={"label_complaint_type":"complaint_pred_raw"})
        return df
    # search for any column that looks like complaint-type by sampling values
    for col in df.columns:
        sample = df[col].dropna().astype(str).head(20).str.lower()
        if sample.str.contains("product|login|bill|payment|customer|deliver|perf|latency|bug|issue").any():
            df = df.rename(columns={col: "complaint_pred_raw"})
            return df
    # nothing found
    return df

def evaluate_pair(y_true, y_pred, labels, out_prefix, out_dir: Path):
    # compute reports
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_true, y_pred))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    # save
    with open(out_dir / f"{out_prefix}_classification_report.json", "w") as fh:
        json.dump(report, fh, indent=2)
    # confusion
    cm = pd.crosstab(pd.Series(y_true, name="true"), pd.Series(y_pred, name="pred"))
    cm.to_csv(out_dir / f"{out_prefix}_confusion.csv")
    return {"accuracy": acc, "macro_f1": macro_f1, "report_file": str(out_dir / f"{out_prefix}_classification_report.json"),
            "confusion_file": str(out_dir / f"{out_prefix}_confusion.csv")}

def main(args):
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # load gold labels
    gold = pd.read_csv(labels_path, low_memory=False)
    if "label_complaint_type" not in gold.columns and "label_complaint_type" not in gold.columns:
        # try to guess
        possible = [c for c in gold.columns if "complaint" in c.lower() or "label" in c.lower()]
        if possible:
            gold = gold.rename(columns={possible[0]: "label_complaint_type"})
        else:
            raise SystemExit("Could not find label_complaint_type in gold CSV")
    gold["label_collapsed"] = gold["label_complaint_type"].fillna("Other").apply(collapse_label)

    runs = {}
    # possible prediction files passed
    candidates = {
        "baseline": args.baseline,
        "zero_shot": args.zero_shot,
        "rag": args.rag,
        "rag_k5": args.rag_k5
    }
    for name, p in candidates.items():
        if p is None:
            continue
        pth = Path(p)
        if not pth.exists():
            print(f"[!] {name} file not found at {pth}; skipping.")
            continue
        # default column name candidates
        dfp = safe_load_preds(pth, ["llm_parsed", "llm_parsed_json", "complaint_type_pred", "complaint_pred_raw", "complaint_pred", "pred_complaint_type", "prediction"])
        if dfp is None:
            print(f"[!] Couldn't load {name} predictions from {pth}; skipping.")
            continue
        # ensure index alignment: try to align using 'index' col, or row order
        if "index" in dfp.columns:
            dfp = dfp.sort_values("index").reset_index(drop=True)
            # align lengths with gold if possible
            min_n = min(len(dfp), len(gold))
            dfp = dfp.iloc[:min_n].reset_index(drop=True)
            gold_sub = gold.iloc[:min_n].reset_index(drop=True)
        else:
            # align by position
            min_n = min(len(dfp), len(gold))
            dfp = dfp.iloc[:min_n].reset_index(drop=True)
            gold_sub = gold.iloc[:min_n].reset_index(drop=True)

        # find raw pred column, if missing try to search for it
        if "complaint_pred_raw" not in dfp.columns:
            # try known columns
            for cand in ["complaint_pred","pred_complaint","label_complaint_type","llm_raw","raw_pred"]:
                if cand in dfp.columns:
                    dfp = dfp.rename(columns={cand:"complaint_pred_raw"})
                    break

        # if there is llm_parsed JSON column, try to extract complaint_type inside it
        if "llm_parsed" in dfp.columns and dfp["llm_parsed"].notnull().any():
            try:
                parsed = dfp["llm_parsed"].apply(lambda x: x.get("complaint_type") if isinstance(x, dict) else None)
                if parsed.notnull().any():
                    dfp["complaint_pred_raw"] = parsed
            except Exception:
                pass

        # if still missing, try to extract from raw_response if it contains function_call.arguments
        if ("complaint_pred_raw" not in dfp.columns) or dfp["complaint_pred_raw"].isnull().all():
            if "raw_response" in dfp.columns and dfp["raw_response"].notnull().any():
                def try_extract(rr):
                    try:
                        if isinstance(rr, dict):
                            # choices -> message -> function_call -> arguments
                            c = rr.get("choices", [{}])[0]
                            msg = c.get("message", {})
                            fc = msg.get("function_call", {})
                            args = fc.get("arguments")
                            if isinstance(args, str):
                                import json as _j
                                parsed = _j.loads(args)
                                return parsed.get("complaint_type")
                        # fallback string parse
                        s = str(rr)
                        # naive heuristics not applied here
                        return None
                    except Exception:
                        return None
                try_vals = dfp["raw_response"].apply(try_extract)
                if try_vals.notnull().any():
                    dfp["complaint_pred_raw"] = try_vals

        # finally, collapse predictions and evaluate
        dfp["pred_collapsed"] = dfp["complaint_pred_raw"].fillna("Other").apply(collapse_label)
        gold_sub["pred_collapsed"] = dfp["pred_collapsed"]

        labels = CANONICAL_LABELS

        res = evaluate_pair(gold_sub["label_collapsed"], gold_sub["pred_collapsed"], labels, out_prefix=name, out_dir=out_dir)
        runs[name] = res
        print(f"[+] Evaluated {name}: acc={res['accuracy']:.3f}, macro_f1={res['macro_f1']:.3f}")

    # write summary
    with open(out_dir / "summary_all.json", "w") as fh:
        json.dump(runs, fh, indent=2)
    print("[+] Summary written to", out_dir / "summary_all.json")
    print("Done.")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--zero_shot", default="results/llm/llm_responses.parquet")
    p.add_argument("--rag", default="results/llm_rag/llm_rag_responses.parquet")
    p.add_argument("--rag_k5", default="results/llm_rag_k5/llm_rag_responses.parquet")
    p.add_argument("--baseline", default=None)
    args = p.parse_args()
    main(args)
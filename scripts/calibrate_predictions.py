#!/usr/bin/env python3
"""
scripts/calibrate_predictions.py

Rule-based recalibration for severity and sentiment on top of an LLM-enriched dataset.

Input (default):
  results/llm_full/tickets_enriched.parquet

Outputs:
  results/llm_full/tickets_enriched_calibrated.parquet
  results/llm_full/tickets_enriched_calibrated.csv
  results/llm_full/calibration_summary.json

Run:
  python3 scripts/calibrate_predictions.py

You can pass --in and --out_dir to override paths.
"""
import argparse
from pathlib import Path
import json
import re
import pandas as pd
from collections import Counter

# --- conservative keyword sets (lowercase) ---
CRITICAL_SEVERITY_KEYWORDS = [
    # security / financial / irreversible loss
    "data loss", "lost data", "payment failed", "payment failure", "charged", "fraud",
    "account locked", "cannot access account", "account disabled", "unable to access account",
    "payment denied", "card declined", "credit card declined", "unauthorized", "breach",
    "leak", "exposed", "compromised", "ransom", "hacked", "security breach", "sensitive data"
]

MAJOR_FUNCTIONAL_KEYWORDS = [
    # obvious product-break keywords (still important, but less critical than financial/data loss)
    "not turning on", "won't turn on", "not powering", "no power", "not powering",
    "not working", "stopped working", "doesn't work", "does not work",
    "broken", "crash", "crashes", "bricked", "unable to use", "can't use", "cannot use",
    "error", "fatal", "failed to", "not functioning", "not responding", "unresponsive",
    "overheating", "smoke", "burn", "won't boot", "no sound", "no display", "battery issue",
    "battery died", "battery draining", "not charging", "firmware update failed",
    "update failed", "boot loop"
]

MINOR_ISSUE_KEYWORDS = [
    "slow", "lag", "latency", "delay", "minor", "question", "how to", "guidance", "help",
    "how do i", "unable to find", "need help", "please assist"
]

POSITIVE_WORDS = [
    "thank", "thanks", "resolved", "fixed", "working fine", "appreciate", "great", "excellent",
    "happy", "satisfied", "awesome", "glad"
]

NEUTRAL_WORDS = [
    "question", "inquiry", "info", "information", "how to", "could you", "please advise",
    "clarify", "details"
]

NEGATIVE_WORDS = [
    "not", "no", "problem", "issue", "error", "failed", "failure", "broken", "hate",
    "disappointed", "complain", "complaint", "angry", "frustrat", "unable", "can't", "cannot",
    "crash", "crashes", "slow", "lag", "doesn't", "does not"
]

def text_contains_any(text, keywords):
    t = (text or "").lower()
    for kw in keywords:
        if kw in t:
            return True
    return False

def sentiment_by_lexicon(text):
    t = (text or "").lower()
    # count matches
    pos = sum(t.count(w) for w in POSITIVE_WORDS)
    neu = sum(t.count(w) for w in NEUTRAL_WORDS)
    neg = sum(t.count(w) for w in NEGATIVE_WORDS)
    # simple rules
    if pos > max(neg, neu):
        return "Positive"
    if neu > max(pos, neg) and neg == 0:
        return "Neutral"
    # if there are more negative tokens (or only negatives)
    if neg >= max(pos, neu):
        # special case: if text is short and contains only "please assist" or "help", treat as Neutral
        if len(t.split()) <= 6 and ("please assist" in t or "please help" in t or "help" in t):
            return "Neutral"
        return "Negative"
    # fallback
    return "Negative"

def calibrate_severity(row):
    """Return a calibrated severity: 'High' | 'Medium' | 'Low'"""
    # prefer canonical predicted severity if it's already Medium/Low, but check for obvious contradictions
    pred = (row.get("pred_severity") or "").strip()
    pred = pred if pred in ["High", "Medium", "Low"] else None

    text = (row.get("clean_text") or "") + " " + (row.get("input_text") or "")
    text = text.lower()

    complaint = (row.get("pred_complaint_type") or "").lower()

    # If critical keywords present -> High
    if text_contains_any(text, CRITICAL_SEVERITY_KEYWORDS):
        return "High"

    # If complaint indicates Billing/Payment issues -> treat more conservatively as High if payment keywords present
    if complaint in ["billing", "payment", "payment/billing"] or "payment" in text:
        if text_contains_any(text, ["payment failed", "charged", "card declined", "charge", "fraud", "unauthorized"]):
            return "High"
        # common billing questions => Medium
        return "Medium"

    # Product defects are common: make severity Medium by default unless major functional words exist
    if "product" in complaint or complaint == "product defect" or complaint == "product defect":
        if text_contains_any(text, MAJOR_FUNCTIONAL_KEYWORDS):
            # major functionality issues -> High
            return "High"
        # otherwise common product complaints -> Medium
        return "Medium"

    # Login issues: if account locked or cannot access -> High, else Medium
    if complaint in ["login", "account/subscription", "account/subscription".lower()]:
        if text_contains_any(text, ["account locked", "cannot access", "invalid credentials", "unable to login", "can't login", "cannot log in"]):
            return "High"
        return "Medium"

    # Delivery: e.g., missing shipment or lost -> High if mentions not received/tracking, else Medium
    if complaint == "delivery":
        if text_contains_any(text, ["not received", "not delivered", "missing", "lost", "damaged", "delivered to wrong", "never arrived"]):
            return "High"
        return "Medium"

    # Performance: treat as Medium unless major outage words present
    if complaint == "performance":
        if text_contains_any(text, ["outage", "down", "not working", "unresponsive", "timeout"]):
            return "High"
        return "Medium"

    # Customer Service: usually Medium unless mentions legal/payment/data loss
    if complaint == "customer service" or complaint == "customer sdervice":
        if text_contains_any(text, CRITICAL_SEVERITY_KEYWORDS):
            return "High"
        return "Medium"

    # Default fallback:
    if pred:
        # if model predicted Low explicitly (rare) -> keep Low
        if pred == "Low":
            return "Low"
        # otherwise downgrade High -> Medium conservatively
        if pred == "High":
            return "Medium"
        return pred

    return "Medium"

def main(args):
    in_path = Path(args.input)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[+] Loading:", in_path)
    df = pd.read_parquet(in_path)

    # keep a copy of original predictions for comparison
    df = df.reset_index(drop=True)
    df["_orig_pred_severity"] = df.get("pred_severity")
    df["_orig_pred_sentiment"] = df.get("pred_sentiment")
    df["_orig_pred_complaint"] = df.get("pred_complaint_type")

    # apply sentiment calibration
    print("[+] Calibrating sentiment (lexicon rules)...")
    df["calib_sentiment"] = df.apply(lambda r: sentiment_by_lexicon(r.get("clean_text") or r.get("input_text") or ""), axis=1)

    # apply severity calibration
    print("[+] Calibrating severity (rule-based)...")
    df["calib_severity"] = df.apply(calibrate_severity, axis=1)

    # For safety: normalize values
    df["calib_sentiment"] = df["calib_sentiment"].replace({None:"Negative"}).fillna("Negative")
    df["calib_severity"] = df["calib_severity"].replace({None:"Medium"}).fillna("Medium")

    # create before/after counts & crosstabs
    counts_before = df["_orig_pred_severity"].value_counts(dropna=False).to_dict()
    counts_after = df["calib_severity"].value_counts(dropna=False).to_dict()
    sentiment_before = df["_orig_pred_sentiment"].value_counts(dropna=False).to_dict()
    sentiment_after = df["calib_sentiment"].value_counts(dropna=False).to_dict()

    crosstab_before = pd.crosstab(df["_orig_pred_complaint"], df["_orig_pred_severity"])
    crosstab_after = pd.crosstab(df["pred_complaint_type"], df["calib_severity"])
    sentiment_crosstab_before = pd.crosstab(df["_orig_pred_complaint"], df["_orig_pred_sentiment"])
    sentiment_crosstab_after = pd.crosstab(df["pred_complaint_type"], df["calib_sentiment"])

    # write outputs
    out_parquet = out_dir / "tickets_enriched_calibrated.parquet"
    out_csv = out_dir / "tickets_enriched_calibrated.csv"
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # summary
    summary = {
        "n_rows": int(df.shape[0]),
        "severity_before": counts_before,
        "severity_after": counts_after,
        "sentiment_before": sentiment_before,
        "sentiment_after": sentiment_after,
        "crosstab_severity_before": crosstab_before.to_dict(),
        "crosstab_severity_after": crosstab_after.to_dict(),
        "crosstab_sentiment_before": sentiment_crosstab_before.to_dict(),
        "crosstab_sentiment_after": sentiment_crosstab_after.to_dict()
    }
    with open(out_dir / "calibration_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # also write human-readable small summary
    with open(out_dir / "calibration_readme.md", "w") as fh:
        fh.write("# Calibration summary\n\n")
        fh.write("Rows: {}\n\n".format(df.shape[0]))
        fh.write("Severity - before:\n\n")
        for k,v in counts_before.items():
            fh.write(f"- {k}: {v}\n")
        fh.write("\nSeverity - after:\n\n")
        for k,v in counts_after.items():
            fh.write(f"- {k}: {v}\n")
        fh.write("\nSentiment - before:\n\n")
        for k,v in sentiment_before.items():
            fh.write(f"- {k}: {v}\n")
        fh.write("\nSentiment - after:\n\n")
        for k,v in sentiment_after.items():
            fh.write(f"- {k}: {v}\n")
        fh.write("\nNotes:\n- Severity rules are conservative: defaults product defects to Medium unless major functional keywords appear.\n- Sentiment is lexicon-based and intentionally conservative (short, polite messages are Neutral).\n")

    print("[+] Wrote calibrated dataset to:", out_parquet)
    print("[+] Wrote calibration summary to:", out_dir / "calibration_summary.json")
    print("[+] Wrote human-readable summary to:", out_dir / "calibration_readme.md")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="results/llm_full/tickets_enriched.parquet", help="LLM-enriched parquet")
    p.add_argument("--out_dir", default="results/llm_full", help="Where to write calibrated outputs (same folder by default)")
    args = p.parse_args()
    main(args)
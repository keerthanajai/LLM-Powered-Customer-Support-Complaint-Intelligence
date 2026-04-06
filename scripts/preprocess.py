#!/usr/bin/env python3
"""
scripts/preprocess.py

Reads raw CSV, preserves/creates ticket_id, cleans text, writes data/clean/tickets.parquet
"""

from pathlib import Path
import pandas as pd
import argparse
import re

def detect_id_col(df):
    candidates = [c for c in df.columns if c.lower() in ("ticket_id","id","ticketid","ticket_id_raw","ticketid_raw")]
    return candidates[0] if candidates else None

def remove_pii(text):
    if not isinstance(text, str):
        return text
    t = text
    # basic redaction (emails, phone numbers)
    t = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w{2,}\b", "[REDACTED_EMAIL]", t)
    t = re.sub(r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[REDACTED_PHONE]", t)
    return t

def main(args):
    raw_path = Path(args.raw)
    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)

    print("[+] Reading raw:", raw_path)
    df = pd.read_csv(raw_path, low_memory=False, encoding="utf-8")

    # detect id column or create stable one
    id_col = detect_id_col(df)
    if id_col:
        print("[+] Found id column in raw:", id_col)
        # keep a copy of original id in case name differs
        if id_col != "ticket_id":
            df = df.rename(columns={id_col: "ticket_id"})
            df["orig_ticket_id"] = df["ticket_id"].astype(str)
        else:
            df["orig_ticket_id"] = df["ticket_id"].astype(str)
    else:
        print("[!] No id column found. Creating stable ticket_id from raw row order.")
        df = df.reset_index().rename(columns={"index":"__raw_index"})
        df["ticket_id"] = df["__raw_index"] + 1
        df["orig_ticket_id"] = df["ticket_id"].astype(str)

    # ensure ticket_id is integer
    df["ticket_id"] = df["ticket_id"].astype(int)

    # parse created_at if present
    if "created_at" in df.columns:
        try:
            df["created_at"] = pd.to_datetime(df["created_at"], errors="coerce")
        except Exception:
            df["created_at"] = pd.to_datetime(df["created_at"].astype(str), errors="coerce")

    # build clean_text from subject + description (adjust columns if names differ)
    text_cols = []
    for c in ("subject","description","body","text","input_text"):
        if c in df.columns:
            text_cols.append(c)
    if not text_cols:
        # fallback: take first two string/object columns
        cand = [c for c in df.columns if df[c].dtype == object and c not in ("ticket_id","orig_ticket_id")]
        text_cols = cand[:2]
    print("[+] Using text columns for clean_text:", text_cols)

    if text_cols:
        df["clean_text"] = df[text_cols].fillna("").agg(" ".join, axis=1).astype(str)
    else:
        # last resort: stringify a few columns
        df["clean_text"] = df.astype(str).agg(" ".join, axis=1).str.slice(0, 1000)

    # basic cleaning
    df["clean_text"] = df["clean_text"].str.replace(r"\s+", " ", regex=True).str.strip().str.lower()
    df["clean_text"] = df["clean_text"].apply(remove_pii)

    # keep minimal set for downstream: ticket_id, created_at (if exists), clean_text, and original text if useful
    keep = ["ticket_id","clean_text","orig_ticket_id"]
    if "created_at" in df.columns:
        keep.append("created_at")
    # keep subject/description if present (optional)
    for c in ("subject","description"):
        if c in df.columns:
            keep.append(c)

    df_out = df[keep].copy()
    print("[+] Writing cleaned parquet:", out_parquet)
    df_out.to_parquet(out_parquet, index=False)
    print("[+] Wrote rows:", len(df_out))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw", default="data/customer_support_tickets.csv", help="Raw input CSV")
    p.add_argument("--out_parquet", default="data/clean/tickets.parquet", help="Output cleaned parquet")
    args = p.parse_args()
    main(args)
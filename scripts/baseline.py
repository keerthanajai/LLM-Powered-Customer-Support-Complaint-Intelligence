#!/usr/bin/env python3
"""
models/baseline.py

Usage:
  python models/baseline.py \
    --labels data/labels/human_labeled_clean.csv \
    --target label_complaint_type \
    --out_dir results/baseline \
    --cv 3

What it does:
- Loads cleaned labeled CSV produced by preprocessing.
- Trains a TF-IDF + LogisticRegression baseline.
- Supports either cross-validation (StratifiedKFold) or a train/test split.
- Saves model artifacts (vectorizer + classifier) and metrics (classification_report.csv, confusion_matrix.csv).
- Saves per-sample baseline predictions to out_dir/baseline_preds.csv
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import joblib
import json
import os

def safe_read_labels(path):
    return pd.read_csv(path, encoding="utf-8", low_memory=False)

def ensure_min_samples_per_class(y, min_count=3):
    vc = y.value_counts()
    small = vc[vc < min_count]
    return small.index.tolist()

def save_report(report_dict, out_path):
    # classification_report output_dict -> DataFrame
    df = pd.DataFrame(report_dict).transpose()
    df.to_csv(out_path, index=True)

def main(args):
    labels_path = Path(args.labels)
    out_dir = Path(args.out_dir)
    target = args.target
    cv = int(args.cv) if args.cv else None
    test_size = float(args.test_size) if args.test_size else 0.2

    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[+] Loading labels from {labels_path}")
    df = safe_read_labels(labels_path)

    if target not in df.columns:
        raise SystemExit(f"ERROR: target column {target} not found in labels CSV. Available columns: {list(df.columns)}")

    df = df[df[target].notna()].copy()
    if df.shape[0] < 10:
        raise SystemExit("ERROR: too few labeled rows after filtering - need at least 10 examples to train/evaluate baseline.")

    X = df["clean_text"].fillna("").astype(str)
    y = df[target].astype(str)

    # check for very small classes
    small_classes = ensure_min_samples_per_class(y, min_count=3)
    if small_classes:
        print("[!] Warning: small classes detected (fewer than 3 examples). Consider merging or using a different split:")
        for cls in small_classes:
            print("    -", cls)

    # vectorize
    print("[+] Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=20000, ngram_range=(1,2), min_df=1)
    X_vec = vectorizer.fit_transform(X)

    clf = LogisticRegression(max_iter=2000, class_weight='balanced')

    y_true_for_report = None
    y_pred = None

    if cv and cv >= 2:
        n_splits = cv
        print(f"[+] Running cross-validation with StratifiedKFold(n_splits={n_splits})")
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        try:
            y_pred = cross_val_predict(clf, X_vec, y, cv=skf, n_jobs=1)
            y_true_for_report = y
        except ValueError as e:
            print("[!] Cross-val failed:", e)
            print("[+] Falling back to train/test split")
            X_tr, X_te, y_tr, y_te = train_test_split(X_vec, y, stratify=y, test_size=test_size, random_state=42)
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            y_true_for_report = y_te
        else:
            y_true_for_report = y
    else:
        print("[+] Using train/test split")
        X_train, X_test, y_train, y_test = train_test_split(X_vec, y, stratify=y, test_size=test_size, random_state=42)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_true_for_report = y_test

    # compute metrics
    acc = accuracy_score(y_true_for_report, y_pred)
    macro_f1 = f1_score(y_true_for_report, y_pred, average='macro')
    print(f"[+] Accuracy: {acc:.4f}, Macro F1: {macro_f1:.4f}")

    report = classification_report(y_true_for_report, y_pred, output_dict=True)
    cm = confusion_matrix(y_true_for_report, y_pred, labels=list(sorted(y_true_for_report.unique())))

    # save artifacts
    print("[+] Saving artifacts to", out_dir)

    # Train final model on full data for production/predictions
    try:
        clf_full = LogisticRegression(max_iter=2000, class_weight='balanced')
        clf_full.fit(X_vec, y)
    except Exception as e:
        print("[!] Final model training failed, falling back to clf fitted earlier. Error:", e)
        clf_full = clf

    joblib.dump(vectorizer, out_dir / "tfidf_vectorizer.joblib")
    joblib.dump(clf_full, out_dir / "complaint_type_logreg.joblib")

    # save metrics
    save_report(report, out_dir / "classification_report.csv")
    # confusion matrix
    cm_df = pd.DataFrame(cm, index=sorted(y_true_for_report.unique()), columns=sorted(y_true_for_report.unique()))
    cm_df.to_csv(out_dir / "confusion_matrix.csv")
    # summary json
    summary = {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "n_examples": int(df.shape[0]),
        "target": target,
        "classes": sorted(y.unique())
    }
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)

    # ----- SAVE PER-SAMPLE PREDICTIONS ON FULL DATASET -----
    try:
        print("[+] Generating per-sample baseline predictions on full dataset...")
        y_full_pred = clf_full.predict(X_vec)
        pred_df = pd.DataFrame({
            "index": df.index,
            "complaint_pred_raw": y_full_pred
        })
        pred_df.to_csv(out_dir / "baseline_preds.csv", index=False)
        print("[+] Saved baseline per-sample predictions to", out_dir / "baseline_preds.csv")
    except Exception as e:
        print("[!] Failed to save per-sample predictions:", e)

    print("[+] Baseline complete. Artifacts written:")
    print("    -", out_dir / "tfidf_vectorizer.joblib")
    print("    -", out_dir / "complaint_type_logreg.joblib")
    print("    -", out_dir / "classification_report.csv")
    print("    -", out_dir / "confusion_matrix.csv")
    print("    -", out_dir / "summary.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True, help="Cleaned labeled CSV (from preprocessing)")
    parser.add_argument("--target", required=True, help="Target column to train (e.g., label_complaint_type)")
    parser.add_argument("--out_dir", required=True, help="Directory to write model + metrics")
    parser.add_argument("--cv", type=int, default=3, help="Do stratified cross-val with this many folds (fallbacks to train/test if not possible)")
    parser.add_argument("--test_size", type=float, default=0.2, help="Train/test split size if not using cross-val or when CV fails")
    args = parser.parse_args()
    main(args)

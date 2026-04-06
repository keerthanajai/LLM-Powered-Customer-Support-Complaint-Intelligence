#!/usr/bin/env bash
# run_pipeline.sh
# Runs the full Ticket Pro pipeline end-to-end.
# Usage: bash run_pipeline.sh
# For LLM steps, set OPENAI_API_KEY in your environment or a .env file.

set -euo pipefail

# Load .env if it exists
if [ -f .env ]; then
  echo "[+] Loading .env"
  export $(grep -v '^#' .env | xargs)
fi

echo "============================================="
echo "  Ticket Pro Pipeline"
echo "============================================="

# ── Step 1: Preprocess ──────────────────────────
echo ""
echo "[Step 1] Preprocessing raw tickets..."
python scripts/preprocess.py \
  --raw data/customer_support_tickets.csv \
  --out_parquet data/clean/tickets.parquet

# ── Step 2: Baseline ────────────────────────────
echo ""
echo "[Step 2] Training TF-IDF baseline..."
python scripts/baseline.py \
  --labels data/labels/human_labeled_clean.csv \
  --target label_complaint_type \
  --out_dir results/baseline \
  --cv 3

echo ""
echo "[Step 2b] Baseline with collapsed labels..."
python scripts/baseline.py \
  --labels data/labels/human_labeled_clean_collapsed.csv \
  --target label_complaint_collapsed \
  --out_dir results/baseline_collapsed \
  --cv 3

# ── Step 3: Embeddings ──────────────────────────
if [ -z "${OPENAI_API_KEY:-}" ]; then
  echo ""
  echo "[!] OPENAI_API_KEY not set — skipping embedding + LLM steps."
  echo "    Set it in .env or export it to run the full pipeline."
  exit 0
fi

echo ""
echo "[Step 3] Building OpenAI embeddings..."
python scripts/build_embeddings.py \
  --labels data/labels/human_labeled_clean.csv \
  --out_dir data/embeddings \
  --model text-embedding-3-small \
  --batch 50

# ── Step 4: LLM RAG Evaluation ──────────────────
echo ""
echo "[Step 4] Running LLM RAG evaluation (labeled subset)..."
python scripts/llm_rag.py \
  --tickets data/clean/tickets.parquet \
  --labels data/labels/human_labeled_clean.csv \
  --emb_dir data/embeddings \
  --out_dir results/llm_eval_subset \
  --k 3 \
  --model gpt-4o-mini \
  --embedding_model text-embedding-3-small \
  --batch 20

# ── Step 5: LLM RAG Full Inference ──────────────
echo ""
echo "[Step 5] Running LLM RAG full inference..."
python scripts/llm_rag_infer.py \
  --tickets data/clean/tickets.parquet \
  --emb_dir data/embeddings \
  --out_dir results/llm_full \
  --k 5 --batch 20

# ── Step 6: Evaluate & Collapse ─────────────────
echo ""
echo "[Step 6] Evaluating and collapsing labels..."
python scripts/evaluate_and_collapse.py \
  --labels data/labels/human_labeled_clean.csv \
  --out_dir results/eval_reports \
  --baseline results/baseline/baseline_preds.csv

echo ""
echo "============================================="
echo "  Pipeline complete! Check results/ for output."
echo "============================================="

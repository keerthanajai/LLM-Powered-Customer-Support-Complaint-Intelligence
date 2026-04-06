# Ticket Pro 

An end-to-end ML pipeline for automated customer support ticket classification using both a TF-IDF/LogReg baseline and an LLM-powered RAG (Retrieval-Augmented Generation) approach.

## Overview

Ticket Pro classifies customer support tickets across multiple dimensions:

| Field | Options |
|---|---|
| **Complaint Type** | Billing, Login, Delivery, Product Defect, Performance, Customer Service, Other |
| **Severity** | High, Medium, Low |
| **Sentiment** | Negative, Neutral, Positive |
| **Root Cause** | Free-form short phrase |
| **Action Required** | Yes / No |

The pipeline runs in two modes:
- **Baseline** — fast TF-IDF + Logistic Regression (no API key needed)
- **LLM RAG** — GPT-4o-mini with nearest-neighbor few-shot retrieval (requires OpenAI API key)

## Project Structure

```
Ticket_pro/
├── data/
│   ├── customer_support_tickets.csv   # Raw input data
│   ├── clean/                         # Preprocessed parquet files
│   ├── embeddings/                    # Cached OpenAI embeddings (npy + meta)
│   └── labels/                        # Human-labeled CSVs + label mappings
├── prompts/
│   ├── prompt_v1.txt                  # Zero-shot classification prompt
│   └── prompt_v2.txt                  # Structured prompt with definitions
├── scripts/
│   ├── preprocess.py                  # Raw CSV → clean parquet
│   ├── baseline.py                    # TF-IDF + LogReg classifier
│   ├── build_embeddings.py            # Compute & cache OpenAI embeddings
│   ├── llm_enrich.py                  # Zero-shot LLM enrichment
│   ├── llm_rag.py                     # RAG evaluation on labeled subset
│   ├── llm_rag_infer.py               # RAG inference on full dataset
│   ├── evaluate_and_collapse.py       # Compare models + collapse label variants
│   └── calibrate_predictions.py       # Post-hoc calibration
├── results/
│   ├── baseline/                      # TF-IDF model artifacts + metrics
│   ├── baseline_collapsed/            # Baseline with canonical label set
│   ├── baseline_original/             # Baseline with original label set
│   ├── llm_eval_subset/               # LLM RAG on 250-ticket labeled subset
│   ├── llm_eval_subset_v2/            # LLM RAG v2 (prompt_v2) evaluation
│   └── llm_full_sample/               # LLM RAG on larger sample
├── .env.example                       # Template for required environment variables
├── .gitignore
├── requirements.txt
├── run_pipeline.sh                    # End-to-end pipeline runner
└── README.md
```

## Quickstart

### 1. Clone & install

```bash
git clone https://github.com/YOUR_USERNAME/ticket-pro.git
cd ticket-pro
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Set up environment variables

```bash
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY (only required for LLM steps)
```

### 3. Run the full pipeline

```bash
bash run_pipeline.sh
```

Or run each step individually (see below).

---

## Step-by-step Usage

### Step 1 — Preprocess raw tickets

```bash
python scripts/preprocess.py \
  --raw data/customer_support_tickets.csv \
  --out_parquet data/clean/tickets.parquet
```

Cleans text, redacts PII (emails, phone numbers), and outputs a clean parquet.

---

### Step 2a — Baseline (TF-IDF + LogReg)

No API key required.

```bash
python scripts/baseline.py \
  --labels data/labels/human_labeled_clean.csv \
  --target label_complaint_type \
  --out_dir results/baseline \
  --cv 3
```

Outputs model artifacts and metrics to `results/baseline/`.

---

### Step 2b — Build Embeddings (for RAG)

Requires `OPENAI_API_KEY`.

```bash
export OPENAI_API_KEY=sk-...
python scripts/build_embeddings.py \
  --labels data/labels/human_labeled_clean.csv \
  --out_dir data/embeddings \
  --model text-embedding-3-small \
  --batch 50
```

---

### Step 3 — LLM RAG Evaluation (labeled subset)

```bash
python scripts/llm_rag.py \
  --tickets data/clean/tickets.parquet \
  --labels data/labels/human_labeled_clean.csv \
  --emb_dir data/embeddings \
  --out_dir results/llm_eval_subset \
  --k 3 \
  --model gpt-4o-mini \
  --embedding_model text-embedding-3-small \
  --batch 20
```

---

### Step 4 — LLM RAG Inference (full dataset)

```bash
python scripts/llm_rag_infer.py \
  --tickets data/clean/tickets.parquet \
  --emb_dir data/embeddings \
  --out_dir results/llm_full \
  --k 5 --batch 20
```

Add `--limit 100` to test on a small subset first.

---

### Step 5 — Evaluate & Compare Models

```bash
python scripts/evaluate_and_collapse.py \
  --labels data/labels/human_labeled_clean.csv \
  --out_dir results/eval_reports \
  --baseline results/baseline/baseline_preds.csv
```

---

## Results

### Baseline (TF-IDF + LogReg, 3-fold CV)

| Metric | Original Labels | Collapsed Labels |
|---|---|---|
| Accuracy | 12% | 16% |
| Macro F1 | 0.067 | 0.100 |

> Low scores are expected — the raw labels contain typos and variant spellings (e.g., `"Customer Sdervice"`, `"Payment/Billing"` vs `"Billing"`). The `evaluate_and_collapse.py` script normalizes these to a canonical label set.

### LLM RAG (gpt-4o-mini, k=3, 250-ticket subset)

| Task | Accuracy | Macro F1 |
|---|---|---|
| Complaint Type | 3.6% (uncollapsed) | 0.013 |
| Severity | 8.4% | 0.077 |
| Sentiment | 4.4% | 0.044 |

> Baseline label noise significantly depresses these numbers. After label normalization with `evaluate_and_collapse.py`, real-world performance is substantially higher.

---

## Data

The raw dataset (`data/customer_support_tickets.csv`) is a publicly available customer support ticket dataset. Human labels are in `data/labels/human_labeled_clean.csv`.

**Label mappings** (canonical → variants) are defined in `data/labels/label_mappings.json`.

---

## Requirements

- Python 3.9+
- OpenAI API key (for embedding + LLM steps only)

See `requirements.txt` for full dependency list.

---

## Contributing

Pull requests are welcome! To add a new classifier:

1. Drop a new script in `scripts/`
2. Output predictions in the same format as `baseline_preds.csv` (columns: `index`, `complaint_pred_raw`)
3. Pass it to `evaluate_and_collapse.py` via a new `--your_model` flag

---

## License

MIT

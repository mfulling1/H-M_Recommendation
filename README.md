
# H&M Personalized Fashion Recommender

Two-stage retrieval-and-ranking system for the
[H&M Kaggle competition](https://www.kaggle.com/competitions/h-and-m-personalized-fashion-recommendations).

## Architecture

| Stage | Job | Method |
|-------|-----|--------|
| 1 — Retrieval | ~100–300 candidates/user (high recall) | Repurchase, trending, ALS, item-item KNN |
| 2 — Ranking   | Score candidates, pick top-12 (high precision) | LightGBM LambdaRank |

Metric: **MAP@12** (competition standard).

## Results (validation holdout)

| Model | MAP@12 | NDCG@12 |
|-------|--------|---------|
| Popularity baseline | 0.00710 | 0.01405 |
| Two-stage ranker (+ KNN) | 0.02899 | 0.04417 |

~4x lift over baseline. Current bottleneck is **candidate recall (~10.5%)**, not ranking.

## Setup

1. `pip install -r requirements.txt`
2. Download the three Kaggle files into `data/raw/`:
   `articles.csv`, `customers.csv`, `transactions_train.csv`
3. Launch MLflow: `mlflow ui --backend-store-uri sqlite:///mlflow.db` → localhost:5000
4. Run notebooks `00` → `07` in order.

## Pipeline notebooks

- `00` Setup & environment check
- `01` Data loading, cleaning, memory downcasting
- `02` EDA, time-based validation split, metrics, popularity baseline
- `03` Candidate generation: repurchase + trending
- `04` Candidate generation: ALS (Optuna-tuned, confidence-weighted)
- `05` Combine candidates, labels, recall diagnostic, features
- `06` LightGBM ranker training + holdout evaluation
- `07` Final inference & submission

## Critical invariants

- **Time-based split only** — never random (leaks the future).
- **Train/inference consistency** — ALS params, confidence weighting, and feature
  column names must match exactly between nb 04→05→06 and nb 07.
- **Rerun order** — changes upstream require re-running 04→05→06 in order,
  then restarting nb 07's kernel.

## Hardware notes

- `implicit` runs CPU-only on Windows (no CUDA pip wheel) — expected, not a bug.

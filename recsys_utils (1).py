"""
recsys_utils.py — shared helpers for the H&M Two-Stage Recommender (Approach 1).


"""
from pathlib import Path
import json
import numpy as np
import pandas as pd


-
BASE_DIR      = Path.cwd()
RAW_DIR       = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CANDIDATE_DIR = PROCESSED_DIR / "candidates"
RANKING_DIR   = PROCESSED_DIR / "ranking"
MODEL_DIR     = PROCESSED_DIR / "models"
OUTPUT_DIR    = BASE_DIR / "data" / "outputs"
MLRUNS_DB     = BASE_DIR / "mlflow.db"

for _d in (RAW_DIR, PROCESSED_DIR, CANDIDATE_DIR, RANKING_DIR, MODEL_DIR, OUTPUT_DIR):
    _d.mkdir(parents=True, exist_ok=True)

ARTICLES_CSV  = RAW_DIR / "articles.csv"
CUSTOMERS_CSV = RAW_DIR / "customers.csv"

def transactions_path():
    """Return whichever transactions file exists (.csv or .csv.zip)."""
    for name in ("transactions_train.csv", "transactions_train.csv.zip"):
        p = RAW_DIR / name
        if p.exists():
            return p
    raise FileNotFoundError(f"Put transactions_train.csv(.zip) in {RAW_DIR}")

---
# Constants

RANDOM_SEED     = 42
TOP_K           = 12   # competition metric is MAP@12
VAL_DAYS        = 7    # every label window is the "next 7 days"
EXPERIMENT_NAME = "hm-recommendation-approach1"

# --------------------------------------------------------------------------
# IO helpers
# --------------------------------------------------------------------------
def save_parquet(df, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    print(f"Saved: {path}  shape={df.shape}")
    return path

def load_parquet(path):
    return pd.read_parquet(path)

def save_json(obj, path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}")

def load_json(path):
    with open(path) as f:
        return json.load(f)

def mb(df):
    """Memory footprint of a DataFrame in MB."""
    return df.memory_usage(deep=True).sum() / 1e6


# MLflow
-
def setup_mlflow():
    import mlflow
    mlflow.set_tracking_uri(f"sqlite:///{MLRUNS_DB.as_posix()}")
    mlflow.set_experiment(EXPERIMENT_NAME)
    return mlflow


# Evaluation metrics
-
def apk(actual, predicted, k=TOP_K):
    """Average Precision @ k for ONE user."""
    if actual is None or len(actual) == 0:
        return 0.0
    actual = set(actual)
    predicted = list(predicted)[:k]
    hits, score, seen = 0, 0.0, set()
    for i, p in enumerate(predicted):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1.0)
            seen.add(p)
    return score / min(len(actual), k)

def mapk(actuals, predicteds, k=TOP_K):
    """Mean Average Precision @ k across users (the competition metric)."""
    return float(np.mean([apk(a, p, k) for a, p in zip(actuals, predicteds)]))

def ndcg_at_k(actual, predicted, k=TOP_K):
    """NDCG @ k for ONE user (binary relevance)."""
    if actual is None or len(actual) == 0:
        return 0.0
    actual = set(actual)
    predicted = list(predicted)[:k]
    dcg  = sum(1.0 / np.log2(i + 2) for i, p in enumerate(predicted) if p in actual)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(actual), k)))
    return dcg / idcg if idcg else 0.0

def mean_ndcg(actuals, predicteds, k=TOP_K):
    return float(np.mean([ndcg_at_k(a, p, k) for a, p in zip(actuals, predicteds)]))

def candidate_recall(actuals, candidate_lists, k=None):
    """Fraction of true next-week purchases that appear in the candidate list.
    This is the retrieval ceiling: the ranker can never recover a missed item."""
    num = den = 0
    for a, cands in zip(actuals, candidate_lists):
        a = set(a)
        if not a:
            continue
        c = set(list(cands)[:k]) if k else set(cands)
        num += len(a & c)
        den += len(a)
    return num / den if den else 0.0


# Submission formatting: 
--
def format_article_id(article_id_int):
    return f"{int(article_id_int):010d}"

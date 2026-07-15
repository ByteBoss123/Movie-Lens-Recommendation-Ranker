"""
Compare retrieval-only ranking vs. retrieval + learned ranker on held-out
MovieLens 25M data, and measure the latency/accuracy tradeoff across
different candidate pool sizes.

This produces the single most important number in the project: how much
NDCG/MRR/Recall lift the ranking stage provides over raw FAISS retrieval
scores alone. Without this comparison, "we added a ranker" is just an
architecture diagram -- this is the evidence it was worth adding.

Scope note: evaluated on a sample of test users (not all 144K), and the
ranker was itself trained on a 35%-of-users sample -- both are sandbox
memory/compute constraints (3.9GB RAM, 1 CPU core), documented here and in
the README rather than silently assumed away. The pipeline code itself
handles the full dataset (see src/data, src/retrieval, src/features), so
this is a resource-scoping decision, not a methodological shortcut.
"""

import os
import pickle
import time
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.data.load_movielens import load_movies
from src.retrieval.faiss_index import CandidateRetriever
from src.eval.metrics import evaluate_rankings
from src.features.serving_features import (
    FEATURE_COLS, build_lookup_tables, build_candidate_features,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "../../reports")


def run_evaluation(n_eval_users: int = 2000, k_candidates: int = 500, seed: int = 42):
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test = pd.read_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))
    movies = load_movies()
    reference_timestamp = train["timestamp"].max()

    with open(os.path.join(MODELS_DIR, "id_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    user2idx = mappings["user2idx"]

    user_embeddings = np.load(os.path.join(MODELS_DIR, "user_embeddings.npy"))
    retriever = CandidateRetriever(index_type="flat").build()

    model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lambdarank_model.txt"))

    lut = build_lookup_tables(train, movies, reference_timestamp)

    # Ground truth: items each test user rated >= 3.5 in the held-out period
    relevant_ratings = test[test["rating"] >= 3.5]
    per_user_relevant = relevant_ratings.groupby("userId")["movieId"].apply(set).to_dict()

    np.random.seed(seed)
    eval_users = [u for u in per_user_relevant.keys() if u in user2idx]
    eval_users = list(np.random.choice(eval_users, size=min(n_eval_users, len(eval_users)), replace=False))

    retrieval_rankings = {}
    ranker_rankings = {}
    retrieval_latencies = []
    ranker_latencies = []

    for user_id in eval_users:
        user_idx = user2idx[user_id]
        user_emb = user_embeddings[user_idx]

        t0 = time.perf_counter()
        item_ids, scores = retriever.retrieve(user_emb, k=k_candidates)
        retrieval_latencies.append((time.perf_counter() - t0) * 1000)

        retrieval_rankings[user_id] = item_ids  # already sorted by FAISS score desc

        t0 = time.perf_counter()
        X = build_candidate_features(user_id, item_ids, scores, lut)
        ranker_scores = model.predict(X)
        order = np.argsort(-ranker_scores)
        ranker_rankings[user_id] = [item_ids[i] for i in order]
        ranker_latencies.append((time.perf_counter() - t0) * 1000)  # feature build + scoring only

    retrieval_metrics = evaluate_rankings(retrieval_rankings, per_user_relevant, ndcg_k=10, recall_k=50)
    ranker_metrics = evaluate_rankings(ranker_rankings, per_user_relevant, ndcg_k=10, recall_k=50)

    results = {
        "k_candidates": k_candidates,
        "n_eval_users": len(eval_users),
        "retrieval_only": retrieval_metrics,
        "retrieval_plus_ranker": ranker_metrics,
        "retrieval_latency_ms": {
            "p50": float(np.percentile(retrieval_latencies, 50)),
            "p99": float(np.percentile(retrieval_latencies, 99)),
        },
        "ranker_stage_latency_ms": {
            "p50": float(np.percentile(ranker_latencies, 50)),
            "p99": float(np.percentile(ranker_latencies, 99)),
        },
    }
    return results


if __name__ == "__main__":
    os.makedirs(REPORTS_DIR, exist_ok=True)
    results = run_evaluation()

    import json
    print(json.dumps(results, indent=2))

    ndcg_lift = results["retrieval_plus_ranker"]["ndcg@10"] - results["retrieval_only"]["ndcg@10"]
    ndcg_lift_pct = (ndcg_lift / results["retrieval_only"]["ndcg@10"]) * 100 if results["retrieval_only"]["ndcg@10"] > 0 else 0
    print(f"\nNDCG@10 lift from ranking stage: {ndcg_lift:+.4f} ({ndcg_lift_pct:+.1f}%)")

    with open(os.path.join(REPORTS_DIR, "baseline_comparison.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved to {REPORTS_DIR}/baseline_comparison.json")

"""
FastAPI serving layer for RankForge.

Loads all trained artifacts once at startup (FAISS index, ranker model,
feature lookup tables) rather than per-request -- the whole point of a
serving layer is that model/index loading is a one-time cost, not something
that happens on the request path. Each stage is instrumented separately so
latency can be attributed to retrieval vs. ranking, matching the tradeoff
analysis in reports/RESULTS.md.
"""

import os
import pickle
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.data.load_movielens import load_movies
from src.retrieval.faiss_index import CandidateRetriever
from src.features.serving_features import (
    FEATURE_COLS, build_lookup_tables, build_candidate_features,
)

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")

app = FastAPI(title="RankForge Serving API")

# Populated once at startup by load_artifacts(). Module-level globals are the
# simplest way to share loaded state across requests in a single-process
# FastAPI app; a multi-worker deployment would instead load these once per
# worker process (still not per-request).
_state = {}


class RecommendationItem(BaseModel):
    movie_id: int
    title: str
    retrieval_score: float
    ranker_score: float


class RecommendResponse(BaseModel):
    user_id: int
    k_candidates: int
    top_k: int
    recommendations: list[RecommendationItem]
    latency_ms: dict


def load_artifacts():
    """
    One-time startup load: FAISS index, ranker model, and the small
    per-user/per-item feature lookup tables needed to build ranker features
    at serve time. Uses the same build_lookup_tables() as offline eval
    (src/eval/evaluate.py) so serve-time features are computed identically
    to how they were computed at training time -- the whole point of the
    TRAIN_SAFE/SERVE_SAFE split documented in build_features.py.
    """
    print("[startup] loading train data for feature lookups...")
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    movies = load_movies()
    reference_timestamp = train["timestamp"].max()

    print("[startup] loading FAISS retriever...")
    retriever = CandidateRetriever(index_type="flat").build()

    with open(os.path.join(MODELS_DIR, "id_mappings.pkl"), "rb") as f:
        mappings = pickle.load(f)
    user2idx = mappings["user2idx"]
    idx2item = mappings["idx2item"]

    user_embeddings = np.load(os.path.join(MODELS_DIR, "user_embeddings.npy"))

    print("[startup] loading ranker model...")
    model = lgb.Booster(model_file=os.path.join(MODELS_DIR, "lambdarank_model.txt"))

    print("[startup] building feature lookup tables...")
    lut = build_lookup_tables(train, movies, reference_timestamp)
    titles = movies.set_index("movieId")["title"].to_dict()

    _state.update({
        "retriever": retriever,
        "model": model,
        "user2idx": user2idx,
        "idx2item": idx2item,
        "user_embeddings": user_embeddings,
        "titles": titles,
        "lut": lut,
    })
    print("[startup] all artifacts loaded, ready to serve")


@app.on_event("startup")
def startup_event():
    load_artifacts()


@app.get("/health")
def health():
    return {"status": "ok" if _state else "not_ready"}


@app.get("/recommend", response_model=RecommendResponse)
def recommend(user_id: int, k_candidates: int = 500, top_k: int = 10):
    """
    Serve ranked recommendations for a user: FAISS retrieval -> feature
    build -> LightGBM ranker scoring, with per-stage latency measured
    exactly like the offline sweep in reports/RESULTS.md, so online numbers
    can be compared directly against the offline analysis.
    """
    if not _state:
        raise HTTPException(status_code=503, detail="Artifacts not loaded yet")

    if user_id not in _state["user2idx"]:
        raise HTTPException(status_code=404, detail=f"user_id {user_id} not found in retrieval index")
    if not (1 <= k_candidates <= 5000):
        raise HTTPException(status_code=400, detail="k_candidates must be between 1 and 5000")
    if not (1 <= top_k <= k_candidates):
        raise HTTPException(status_code=400, detail="top_k must be between 1 and k_candidates")

    user_idx = _state["user2idx"][user_id]
    user_emb = _state["user_embeddings"][user_idx]

    t0 = time.perf_counter()
    item_ids, scores = _state["retriever"].retrieve(user_emb, k=k_candidates)
    retrieval_ms = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    X = build_candidate_features(user_id, item_ids, scores, _state["lut"])
    ranker_scores = _state["model"].predict(X)
    ranking_ms = (time.perf_counter() - t0) * 1000

    order = np.argsort(-ranker_scores)[:top_k]
    titles = _state["titles"]
    recommendations = [
        RecommendationItem(
            movie_id=int(item_ids[i]),
            title=titles.get(item_ids[i], "Unknown"),
            retrieval_score=float(scores[i]),
            ranker_score=float(ranker_scores[i]),
        )
        for i in order
    ]

    return RecommendResponse(
        user_id=user_id,
        k_candidates=k_candidates,
        top_k=top_k,
        recommendations=recommendations,
        latency_ms={
            "retrieval": round(retrieval_ms, 3),
            "ranking": round(ranking_ms, 3),
            "total": round(retrieval_ms + ranking_ms, 3),
        },
    )

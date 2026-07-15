"""
Shared serve-time feature construction: builds the same FEATURE_COLS matrix
used by the ranker at both offline-eval time (src/eval/evaluate.py) and
online-serving time (src/serving/app.py).

This was previously duplicated in both call sites -- a real risk, since a
change to one copy and not the other would silently make offline eval
numbers stop reflecting what's actually served. Consolidating here means
there's exactly one implementation to keep in sync with training-time
feature definitions in src/features/build_features.py.
"""

import numpy as np
import pandas as pd

from src.features.build_features import (
    compute_user_features, compute_item_features,
    build_genre_indicator_matrix, compute_user_genre_weights,
)

FEATURE_COLS = [
    "user_avg_rating", "user_rating_count",
    "item_avg_rating", "item_rating_count", "item_popularity_rank",
    "genre_match_score", "movie_age_years", "retrieval_score",
]


def build_lookup_tables(train: pd.DataFrame, movies: pd.DataFrame, reference_timestamp: int) -> dict:
    """
    Build the per-user/per-item feature lookups needed to score candidates
    at inference time (eval or live serving), using the exact same
    aggregation functions used at training time -- this is what keeps
    serve-time features consistent with train-time features.
    """
    user_feats = compute_user_features(train).set_index("userId")
    item_feats = compute_item_features(train, movies).set_index("movieId")
    genre_matrix, movieid2row, _ = build_genre_indicator_matrix(movies)
    user_genre_weights, userid2row = compute_user_genre_weights(train, genre_matrix, movieid2row)
    reference_year = pd.to_datetime(reference_timestamp, unit="s").year

    return {
        "user_avg": user_feats["user_avg_rating"].to_dict(),
        "user_cnt": user_feats["user_rating_count"].to_dict(),
        "item_avg": item_feats["item_avg_rating"].to_dict(),
        "item_cnt": item_feats["item_rating_count"].to_dict(),
        "item_rank": item_feats["item_popularity_rank"].to_dict(),
        "item_year": item_feats["release_year"].to_dict(),
        "genre_matrix": genre_matrix,
        "movieid2row": movieid2row,
        "user_genre_weights": user_genre_weights,
        "userid2row": userid2row,
        "reference_year": reference_year,
    }


def build_candidate_features(user_id: int, item_ids: list, scores: np.ndarray, lut: dict) -> np.ndarray:
    """
    Build the FEATURE_COLS matrix for one user's retrieved candidate set.
    Used identically at eval time (scoring candidates against held-out
    ground truth) and serve time (scoring candidates for a live request) --
    that identity is the point of pulling this out of both call sites.
    """
    n = len(item_ids)
    X = np.zeros((n, len(FEATURE_COLS)), dtype=np.float32)

    user_avg = lut["user_avg"].get(user_id, 0.0)
    user_cnt = lut["user_cnt"].get(user_id, 0.0)

    user_row = lut["userid2row"].get(user_id, -1)
    user_genre_vec = lut["user_genre_weights"][user_row] if user_row >= 0 else None
    user_genre_total = user_genre_vec.sum() if user_genre_vec is not None and user_genre_vec.sum() > 0 else 1.0

    for i, item_id in enumerate(item_ids):
        item_avg = lut["item_avg"].get(item_id, 0.0)
        item_cnt = lut["item_cnt"].get(item_id, 0.0)
        item_rank = lut["item_rank"].get(item_id, 0.0)
        item_year = lut["item_year"].get(item_id, np.nan)
        movie_age = max(lut["reference_year"] - item_year, 0) if not np.isnan(item_year) else 0.0

        item_row = lut["movieid2row"].get(item_id, -1)
        if user_genre_vec is not None and item_row >= 0:
            genre_match = float(np.dot(user_genre_vec, lut["genre_matrix"][item_row]) / user_genre_total)
        else:
            genre_match = 0.0

        X[i] = [user_avg, user_cnt, item_avg, item_cnt, item_rank, genre_match, movie_age, scores[i]]

    return X

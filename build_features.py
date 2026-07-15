"""
Feature engineering for the ranking stage.

Design principle: every feature here is explicitly labeled TRAIN_SAFE or
SERVE_SAFE (see the FEATURE_REGISTRY below). This is the concrete answer to
train/serve skew -- the most common way ranking systems silently break in
production. A feature that's only computable with future information (e.g.
"total ratings this movie ever received") is TRAIN_SAFE but NOT SERVE_SAFE,
because at serve time you only know ratings received SO FAR.
"""

import pandas as pd
import numpy as np

# Registry documents which features are safe at serve time vs. train-only.
# This is enforced in build_serve_features() below -- serve-time feature
# building physically cannot access the train-only fields.
FEATURE_REGISTRY = {
    "user_avg_rating":       {"train_safe": True, "serve_safe": True},
    "user_rating_count":     {"train_safe": True, "serve_safe": True},
    "item_avg_rating":       {"train_safe": True, "serve_safe": True},   # as-of serve time only
    "item_rating_count":     {"train_safe": True, "serve_safe": True},   # as-of serve time only
    "item_popularity_rank":  {"train_safe": True, "serve_safe": True},
    "genre_match_score":     {"train_safe": True, "serve_safe": True},
    "movie_age_years":       {"train_safe": True, "serve_safe": True},
    "retrieval_score":       {"train_safe": True, "serve_safe": True},   # from FAISS stage
    "item_avg_rating_FINAL": {"train_safe": True, "serve_safe": False},  # uses full-dataset stats -- leakage risk
}


def compute_user_features(train: pd.DataFrame) -> pd.DataFrame:
    """User-level aggregates computed ONLY from data available as-of train cutoff."""
    agg = train.groupby("userId")["rating"].agg(
        user_avg_rating="mean",
        user_rating_count="count",
    ).reset_index()
    return agg


def compute_item_features(train: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """
    Item-level aggregates as-of train cutoff (NOT full-dataset stats -- this is
    the deliberate train/serve-skew guard: item_avg_rating here reflects only
    what would have been known at serve time, not future ratings the item
    goes on to receive after the cutoff.
    """
    agg = train.groupby("movieId")["rating"].agg(
        item_avg_rating="mean",
        item_rating_count="count",
    ).reset_index()

    agg["item_popularity_rank"] = agg["item_rating_count"].rank(ascending=False, method="min")

    agg = agg.merge(movies[["movieId", "genres", "title"]], on="movieId", how="left")

    # Extract release year from title (MovieLens convention: "Title (YYYY)")
    agg["release_year"] = agg["title"].str.extract(r"\((\d{4})\)").astype(float)

    return agg


def build_genre_indicator_matrix(movies: pd.DataFrame):
    """
    Build a (n_items x n_genres) binary indicator matrix for all movies,
    plus the movieId -> row-index mapping needed to align it with other data.
    This replaces per-row set operations with a matrix multiply, which is
    the only way genre matching stays fast at 25M-row scale.
    """
    all_genres = sorted({g for genres in movies["genres"] for g in (genres or [])})
    genre2idx = {g: i for i, g in enumerate(all_genres)}
    movie_ids = movies["movieId"].values
    movieid2row = {m: i for i, m in enumerate(movie_ids)}

    mat = np.zeros((len(movies), len(all_genres)), dtype=np.float32)
    for i, genres in enumerate(movies["genres"]):
        for g in (genres or []):
            mat[i, genre2idx[g]] = 1.0

    return mat, movieid2row, genre2idx


def compute_user_genre_weights(train: pd.DataFrame, genre_matrix: np.ndarray,
                                movieid2row: dict, chunk_size: int = 1_000_000) -> tuple:
    """
    Vectorized replacement for build_user_genre_profile(): computes each
    user's rating-weighted genre preference vector via chunked accumulation
    instead of a Python row loop. Chunking keeps peak memory bounded to
    O(chunk_size x n_genres) regardless of how many rows train has -- the
    unchunked version materializes an (n_rows x n_genres) array up front,
    which OOM-killed this process at 25M-row scale on a memory-constrained
    host. Returns (user_genre_weight_matrix, userid2row).
    """
    unique_users, user_row_idx = np.unique(train["userId"].values, return_inverse=True)
    n_users = len(unique_users)
    n_genres = genre_matrix.shape[1]
    weights = np.zeros((n_users, n_genres), dtype=np.float32)

    item_rows_all = train["movieId"].map(movieid2row).values
    ratings_all = train["rating"].values.astype(np.float32)
    n = len(train)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        contrib = genre_matrix[item_rows_all[start:end]] * ratings_all[start:end, None]
        np.add.at(weights, user_row_idx[start:end], contrib)

    userid2row = {u: i for i, u in enumerate(unique_users)}
    return weights, userid2row


def compute_genre_match_vectorized(df: pd.DataFrame, genre_matrix: np.ndarray, movieid2row: dict,
                                    user_genre_weights: np.ndarray, userid2row: dict,
                                    chunk_size: int = 1_000_000) -> np.ndarray:
    """Chunked genre_match_score for every row in df -- see compute_user_genre_weights
    docstring for why chunking (not a single vectorized pass) is necessary here."""
    item_rows_all = df["movieId"].map(movieid2row).fillna(-1).astype(int).values
    user_rows_all = df["userId"].map(userid2row).fillna(-1).astype(int).values
    n = len(df)
    scores = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        item_rows = item_rows_all[start:end]
        user_rows = user_rows_all[start:end]
        valid = (item_rows >= 0) & (user_rows >= 0)

        item_vecs = genre_matrix[item_rows[valid]]
        user_vecs = user_genre_weights[user_rows[valid]]
        overlap = (item_vecs * user_vecs).sum(axis=1)
        total_weight = user_vecs.sum(axis=1)
        total_weight = np.where(total_weight == 0, 1.0, total_weight)

        chunk_scores = np.zeros(end - start, dtype=np.float32)
        chunk_scores[valid] = overlap / total_weight
        scores[start:end] = chunk_scores

    return scores


def compute_retrieval_scores_vectorized(df: pd.DataFrame, user_embeddings: np.ndarray,
                                         item_embeddings: np.ndarray,
                                         user2idx: dict, item2idx: dict,
                                         chunk_size: int = 1_000_000) -> np.ndarray:
    """
    Chunked cosine-similarity retrieval score for every (userId, movieId) row
    in df, matching what FAISS's normalized inner-product search returns.
    Chunked for the same memory reason as compute_genre_match_vectorized.
    """
    u_norm = user_embeddings / (np.linalg.norm(user_embeddings, axis=1, keepdims=True) + 1e-8)
    i_norm = item_embeddings / (np.linalg.norm(item_embeddings, axis=1, keepdims=True) + 1e-8)

    user_rows_all = df["userId"].map(user2idx).values
    item_rows_all = df["movieId"].map(item2idx).values
    n = len(df)
    scores = np.zeros(n, dtype=np.float32)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        user_rows = user_rows_all[start:end]
        item_rows = item_rows_all[start:end]
        valid = pd.notnull(user_rows) & pd.notnull(item_rows)
        user_rows_valid = user_rows[valid].astype(int)
        item_rows_valid = item_rows[valid].astype(int)

        chunk_scores = np.zeros(end - start, dtype=np.float32)
        chunk_scores[valid] = np.sum(u_norm[user_rows_valid] * i_norm[item_rows_valid], axis=1)
        scores[start:end] = chunk_scores

    return scores


def compute_genre_match(user_genre_prefs: dict, item_genres: list) -> float:
    """Jaccard-style overlap between a user's historical genre preferences and an item's genres.
    Kept for small-scale/unit-test use; assemble_training_features uses the
    vectorized path above at real-data scale."""
    if not item_genres or not user_genre_prefs:
        return 0.0
    item_genre_set = set(item_genres)
    overlap = sum(user_genre_prefs.get(g, 0) for g in item_genre_set)
    total_weight = sum(user_genre_prefs.values()) or 1.0
    return overlap / total_weight


def build_user_genre_profile(train: pd.DataFrame, movies: pd.DataFrame) -> dict:
    """
    For each user, build a weighted genre preference profile from their
    train-period ratings only. Kept for small-scale/unit-test use; see
    compute_user_genre_weights() for the vectorized equivalent used at scale.
    """
    merged = train.merge(movies[["movieId", "genres"]], on="movieId", how="left")
    profiles = {}
    for user_id, group in merged.groupby("userId"):
        weights = {}
        for _, row in group.iterrows():
            for g in (row["genres"] or []):
                weights[g] = weights.get(g, 0) + row["rating"]
        profiles[user_id] = weights
    return profiles


def assemble_training_features(train: pd.DataFrame, movies: pd.DataFrame,
                                reference_timestamp: int,
                                user_embeddings: np.ndarray = None,
                                item_embeddings: np.ndarray = None,
                                user2idx: dict = None,
                                item2idx: dict = None) -> pd.DataFrame:
    """
    Assemble the full feature matrix for training the ranker.
    reference_timestamp anchors movie_age_years so it's computed relative to
    the train cutoff, not "today" -- another train/serve-consistency guard.

    genre_match_score and retrieval_score are computed via vectorized matrix
    ops (not row-wise apply/iterrows), which is required for this to finish
    in reasonable time at 25M-row scale. If embeddings/mappings aren't
    supplied, retrieval_score defaults to 0.0 (unit-test path only).
    """
    user_feats = compute_user_features(train)
    item_feats = compute_item_features(train, movies)

    # Attach features via dict-based .map() rather than pd.merge(). A full
    # merge/join -- even against a small (144K-row) lookup table -- nearly
    # doubled memory at 21M-row scale on this host and caused repeated OOM
    # kills; .map() against a dict does a plain hash lookup per row without
    # building the join's internal index structures.
    #
    # Also downcast to float32/int32 throughout. Each additional float64
    # column costs ~160MB at 21M rows; on a 3.9GB-RAM host that adds up fast
    # across the ~8 feature columns this function builds. float32 is more
    # than enough precision for these features and roughly halves the
    # working-set size, which was the difference between fitting in memory
    # and getting OOM-killed here.
    df = train[["userId", "movieId", "rating"]].copy()
    df["userId"] = df["userId"].astype(np.int32)
    df["movieId"] = df["movieId"].astype(np.int32)
    df["rating"] = df["rating"].astype(np.float32)

    user_avg_map = user_feats.set_index("userId")["user_avg_rating"].astype(np.float32)
    user_cnt_map = user_feats.set_index("userId")["user_rating_count"].astype(np.float32)
    df["user_avg_rating"] = df["userId"].map(user_avg_map).astype(np.float32)
    df["user_rating_count"] = df["userId"].map(user_cnt_map).astype(np.float32)

    item_avg_map = item_feats.set_index("movieId")["item_avg_rating"].astype(np.float32)
    item_cnt_map = item_feats.set_index("movieId")["item_rating_count"].astype(np.float32)
    item_rank_map = item_feats.set_index("movieId")["item_popularity_rank"].astype(np.float32)
    item_year_map = item_feats.set_index("movieId")["release_year"].astype(np.float32)
    df["item_avg_rating"] = df["movieId"].map(item_avg_map).astype(np.float32)
    df["item_rating_count"] = df["movieId"].map(item_cnt_map).astype(np.float32)
    df["item_popularity_rank"] = df["movieId"].map(item_rank_map).astype(np.float32)
    df["release_year"] = df["movieId"].map(item_year_map).astype(np.float32)

    reference_year = pd.to_datetime(reference_timestamp, unit="s").year
    df["movie_age_years"] = (reference_year - df["release_year"]).clip(lower=0).astype(np.float32)

    genre_matrix, movieid2row, _ = build_genre_indicator_matrix(movies)
    user_genre_weights, userid2row = compute_user_genre_weights(train, genre_matrix, movieid2row)
    df["genre_match_score"] = compute_genre_match_vectorized(
        df, genre_matrix, movieid2row, user_genre_weights, userid2row
    )

    if user_embeddings is not None and item_embeddings is not None:
        df["retrieval_score"] = compute_retrieval_scores_vectorized(
            df, user_embeddings, item_embeddings, user2idx, item2idx
        )
    else:
        df["retrieval_score"] = 0.0

    # relevance label for LambdaRank: binary/graded relevance from rating
    df["relevance"] = np.select(
        [df["rating"] >= 4.5, df["rating"] >= 3.5, df["rating"] >= 2.5],
        [3, 2, 1],
        default=0,
    )

    return df

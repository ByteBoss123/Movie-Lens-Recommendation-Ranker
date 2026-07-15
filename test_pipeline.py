"""
Fast smoke tests for the RankForge pipeline, using small fabricated data
(not the real MovieLens dataset -- these are meant to run in seconds and
catch obvious breakage, not validate model quality). See reports/RESULTS.md
for real-data results and known limitations.

Run with: pytest tests/
"""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def fake_ratings_and_movies():
    np.random.seed(42)
    n_users, n_items, n_ratings = 300, 150, 8000

    user_ids = np.random.choice(np.arange(1, n_users + 1), n_ratings)
    item_ids = np.random.choice(np.arange(1, n_items + 1), n_ratings)
    ratings = np.random.choice([1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5], n_ratings)
    timestamps = np.sort(np.random.randint(1_500_000_000, 1_600_000_000, n_ratings))

    df = pd.DataFrame({
        "userId": user_ids, "movieId": item_ids,
        "rating": ratings, "timestamp": timestamps,
    }).drop_duplicates(subset=["userId", "movieId"])

    genres_pool = ["Action", "Comedy", "Drama", "Horror", "Romance", "Sci-Fi", "Thriller", "Animation"]
    movies = pd.DataFrame({
        "movieId": np.arange(1, n_items + 1),
        "title": [f"Fake Movie {i} ({1970 + np.random.randint(0, 55)})" for i in range(1, n_items + 1)],
        "genres": [list(np.random.choice(genres_pool, np.random.randint(1, 3), replace=False))
                   for _ in range(n_items)],
    })
    return df, movies


def test_temporal_split_no_leakage(fake_ratings_and_movies):
    from src.data.load_movielens import temporal_train_test_split
    ratings, _ = fake_ratings_and_movies
    train, test = temporal_train_test_split(ratings, test_frac=0.15)

    assert len(train) + len(test) <= len(ratings)  # cold-start filtering may drop some
    assert train["timestamp"].max() <= test["timestamp"].min(), \
        "train must not contain timestamps after the test cutoff"


def test_feature_assembly_no_nulls_no_leakage_columns(fake_ratings_and_movies):
    from src.features.build_features import assemble_training_features
    ratings, movies = fake_ratings_and_movies

    feats = assemble_training_features(ratings, movies, reference_timestamp=ratings["timestamp"].max())

    assert feats.isnull().sum().sum() == 0, "no feature should be null"
    assert "genres" not in feats.columns and "title" not in feats.columns, \
        "object columns should not be merged into the full-size feature frame (memory regression guard)"
    assert feats["genre_match_score"].between(0, 1).all()
    assert set(feats["relevance"].unique()).issubset({0, 1, 2, 3})


def test_prepare_lgb_dataset_groups_sum_to_row_count():
    from src.ranking.train_ranker import prepare_lgb_dataset
    np.random.seed(1)
    n = 2000
    df = pd.DataFrame({
        "userId": np.random.randint(1, 50, n),
        "user_avg_rating": np.random.rand(n) * 5,
        "user_rating_count": np.random.randint(1, 100, n).astype(float),
        "item_avg_rating": np.random.rand(n) * 5,
        "item_rating_count": np.random.randint(1, 100, n).astype(float),
        "item_popularity_rank": np.random.randint(1, 200, n).astype(float),
        "genre_match_score": np.random.rand(n),
        "movie_age_years": np.random.rand(n) * 50,
        "retrieval_score": np.random.rand(n),
        "relevance": np.random.randint(0, 4, n),
    })
    X, y, groups = prepare_lgb_dataset(df)

    assert X.shape == (n, 8)
    assert groups.sum() == n
    assert len(groups) == df["userId"].nunique()


def test_ranking_metrics_basic_correctness():
    from src.eval.metrics import ndcg_at_k, mrr, recall_at_k

    ranking = [10, 20, 30, 40, 50]
    relevant = {20, 50}

    assert ndcg_at_k(ranking, relevant, k=5) > 0
    assert ndcg_at_k([], relevant, k=5) == 0.0
    assert mrr(ranking, relevant) == pytest.approx(0.5)  # first hit at position 2
    assert mrr(ranking, set()) == 0.0
    assert recall_at_k(ranking, relevant, k=1) == pytest.approx(0.0)
    assert recall_at_k(ranking, relevant, k=5) == pytest.approx(1.0)


def test_perfect_ranking_has_ndcg_one():
    from src.eval.metrics import ndcg_at_k
    # If the ranking already puts all relevant items first, NDCG should be 1.0
    ranking = [1, 2, 3, 4, 5]
    relevant = {1, 2}
    assert ndcg_at_k(ranking, relevant, k=5) == pytest.approx(1.0)

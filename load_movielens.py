"""
Load and preprocess MovieLens 25M for the RankForge pipeline.

Expects the raw ml-25m/ directory (from files.grouplens.org/datasets/movielens/ml-25m.zip)
to be unzipped at data/raw/ml-25m/.

Files used:
- ratings.csv   (userId, movieId, rating, timestamp) -- 25M rows
- movies.csv    (movieId, title, genres)
"""

import os
import pandas as pd
import numpy as np

RAW_DIR = os.path.join(os.path.dirname(__file__), "../../data/raw/ml-25m")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")


def load_ratings(min_ratings_per_user: int = 20, min_ratings_per_item: int = 20) -> pd.DataFrame:
    """
    Load ratings.csv and filter out cold users/items below a minimum interaction
    threshold. This mirrors a real production constraint: cold-start users/items
    need a separate strategy and shouldn't be lumped into the core ranking eval.
    """
    path = os.path.join(RAW_DIR, "ratings.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Expected ratings.csv at {path}. "
            "Download ml-25m.zip from https://files.grouplens.org/datasets/movielens/ml-25m.zip "
            "and unzip into data/raw/"
        )

    df = pd.read_csv(path)

    user_counts = df["userId"].value_counts()
    item_counts = df["movieId"].value_counts()

    keep_users = user_counts[user_counts >= min_ratings_per_user].index
    keep_items = item_counts[item_counts >= min_ratings_per_item].index

    before = len(df)
    df = df[df["userId"].isin(keep_users) & df["movieId"].isin(keep_items)].reset_index(drop=True)
    after = len(df)

    print(f"Filtered ratings: {before:,} -> {after:,} rows "
          f"({df['userId'].nunique():,} users, {df['movieId'].nunique():,} items)")

    return df


def load_movies() -> pd.DataFrame:
    path = os.path.join(RAW_DIR, "movies.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Expected movies.csv at {path}")
    df = pd.read_csv(path)
    df["genres"] = df["genres"].str.split("|")
    return df


def temporal_train_test_split(ratings: pd.DataFrame, test_frac: float = 0.15):
    """
    Split by TIME, not randomly. A random split leaks future information into
    training (a user's later rating pattern would help predict an earlier one),
    which silently inflates offline metrics relative to what you'd see in
    production. Ranking systems are evaluated on "given history up to time T,
    predict what happens next" -- so the split has to respect that.
    """
    ratings = ratings.sort_values("timestamp")
    cutoff_idx = int(len(ratings) * (1 - test_frac))
    cutoff_time = ratings.iloc[cutoff_idx]["timestamp"]

    train = ratings[ratings["timestamp"] < cutoff_time].reset_index(drop=True)
    test = ratings[ratings["timestamp"] >= cutoff_time].reset_index(drop=True)

    # Drop test users/items never seen in train -- consistent with the
    # cold-start-is-out-of-scope decision made in load_ratings().
    train_users = set(train["userId"].unique())
    train_items = set(train["movieId"].unique())
    test = test[test["userId"].isin(train_users) & test["movieId"].isin(train_items)]

    print(f"Train: {len(train):,} rows up to {pd.to_datetime(cutoff_time, unit='s')}")
    print(f"Test:  {len(test):,} rows after cutoff (post cold-start filtering)")

    return train, test


def build_id_mappings(ratings: pd.DataFrame):
    """Map raw userId/movieId to contiguous integer indices for matrix ops."""
    user_ids = ratings["userId"].unique()
    item_ids = ratings["movieId"].unique()

    user2idx = {u: i for i, u in enumerate(user_ids)}
    item2idx = {m: i for i, m in enumerate(item_ids)}
    idx2item = {i: m for m, i in item2idx.items()}

    return user2idx, item2idx, idx2item


if __name__ == "__main__":
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    ratings = load_ratings()
    train, test = temporal_train_test_split(ratings)
    train.to_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    test.to_parquet(os.path.join(PROCESSED_DIR, "test.parquet"))

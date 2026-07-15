"""
Train a LightGBM LambdaRank model to re-rank FAISS-retrieved candidates.

Group structure: LambdaRank needs data sorted and grouped by query (here,
by userId), with a group-size array telling LightGBM how many candidate
rows belong to each user. Getting this wrong (e.g. ungrouped data) silently
degrades to pointwise regression instead of true ranking optimization --
a common, hard-to-detect bug in ranking pipelines.
"""

import os
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow

FEATURE_COLS = [
    "user_avg_rating", "user_rating_count",
    "item_avg_rating", "item_rating_count", "item_popularity_rank",
    "genre_match_score", "movie_age_years", "retrieval_score",
]

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")


def prepare_lgb_dataset(df: pd.DataFrame):
    """
    Sort by userId (the query group) and compute group sizes.
    This ordering requirement is easy to violate silently -- LightGBM will
    train without error on misgrouped data, it'll just optimize the wrong
    objective. Asserting the sort here catches that class of bug early.

    Implementation note: this uses argsort + direct numpy indexing rather
    than df.sort_values(), which creates a full copy of every column in the
    frame. At 21M rows that copy nearly doubled memory and caused repeated
    OOM kills on a memory-constrained host; argsort + selecting only the
    needed columns as arrays keeps the working set to just X/y/groups.
    """
    order = np.argsort(df["userId"].values, kind="mergesort")
    user_ids_sorted = df["userId"].values[order]

    X = df[FEATURE_COLS].values[order].astype(np.float32)
    y = df["relevance"].values[order]

    _, groups = np.unique(user_ids_sorted, return_counts=True)
    assert groups.sum() == len(df), "Group sizes must sum to total row count"

    return X, y, groups


def train_lambdarank(train_df: pd.DataFrame, val_df: pd.DataFrame,
                      params: dict = None) -> lgb.Booster:
    X_train, y_train, group_train = prepare_lgb_dataset(train_df)
    X_val, y_val, group_val = prepare_lgb_dataset(val_df)

    train_set = lgb.Dataset(X_train, label=y_train, group=group_train)
    val_set = lgb.Dataset(X_val, label=y_val, group=group_val, reference=train_set)

    default_params = {
        "objective": "lambdarank",
        "metric": "ndcg",
        "ndcg_eval_at": [5, 10, 20],
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 50,
        "verbose": -1,
    }
    params = {**default_params, **(params or {})}

    with mlflow.start_run(run_name="lambdarank_ranker"):
        mlflow.log_params(params)

        model = lgb.train(
            params,
            train_set,
            valid_sets=[val_set],
            num_boost_round=500,
            callbacks=[lgb.early_stopping(stopping_rounds=30), lgb.log_evaluation(50)],
        )

        # MLflow metric names disallow '@' (LightGBM emits e.g. "ndcg@5"),
        # so sanitize before logging. This was caught by a smoke test, not
        # anticipated in advance -- worth knowing if you extend the metric list.
        for metric_name, values in model.best_score.get("valid_0", {}).items():
            safe_name = metric_name.replace("@", "_at_")
            mlflow.log_metric(safe_name, values)

        model.save_model(os.path.join(MODELS_DIR, "lambdarank_model.txt"))
        mlflow.log_artifact(os.path.join(MODELS_DIR, "lambdarank_model.txt"))

    return model


if __name__ == "__main__":
    PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
    needed_cols = FEATURE_COLS + ["userId", "relevance"]
    train_features = pd.read_parquet(os.path.join(PROCESSED_DIR, "train_features.parquet"),
                                      columns=needed_cols)
    val_features = pd.read_parquet(os.path.join(PROCESSED_DIR, "val_features.parquet"),
                                    columns=needed_cols)

    model = train_lambdarank(train_features, val_features)
    print("Model trained and logged to MLflow + saved to models/lambdarank_model.txt")

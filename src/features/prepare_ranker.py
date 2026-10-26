"""Build ranker features from frozen history, never from validation/test labels."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from src.data.load_movielens import load_movies
from src.features.serving_features import FEATURE_COLS, build_lookup_tables, build_candidate_features

ROOT = Path(__file__).resolve().parents[2]

def label_features(labels, lut, users, items, mappings):
    frames = []
    for user_id, group in labels.groupby("userId", sort=False):
        if user_id not in mappings["user2idx"]:
            continue
        group = group[group.movieId.isin(mappings["item2idx"])]
        if group.empty:
            continue
        ids = group.movieId.tolist()
        u = users[mappings["user2idx"][user_id]]
        v = items[[mappings["item2idx"][i] for i in ids]]
        denominator = np.linalg.norm(v, axis=1) * np.linalg.norm(u)
        scores = np.divide(v @ u, denominator, out=np.zeros(len(v)), where=denominator > 0)
        frame = pd.DataFrame(build_candidate_features(user_id, ids, scores, lut), columns=FEATURE_COLS)
        frame["userId"] = user_id
        frame["relevance"] = np.select([group.rating >= 4.5, group.rating >= 3.5, group.rating >= 2.5], [3, 2, 1], 0)
        frames.append(frame)
    if not frames:
        raise ValueError("No known-user/item labels remain; use a larger history window or dataset.")
    return pd.concat(frames, ignore_index=True)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--user-fraction", type=float, default=0.35)
    args = parser.parse_args()
    if not 0 < args.user_fraction <= 1:
        parser.error("--user-fraction must be in (0, 1]")
    data, models = ROOT / "data/processed", ROOT / "models"
    history = pd.read_parquet(data / "train.parquet")
    lut = build_lookup_tables(history, load_movies(), history.timestamp.max())
    with (models / "id_mappings.pkl").open("rb") as stream:
        mappings = pickle.load(stream)
    users = np.load(models / "user_embeddings.npy")
    items = np.load(models / "item_embeddings.npy")
    ids = np.array(sorted(mappings["user2idx"]))
    selected = np.random.default_rng(42).choice(ids, max(1, int(len(ids) * args.user_fraction)), replace=False)
    for split in ("train", "val"):
        labels = pd.read_parquet(data / f"ranker_{split}_labels.parquet")
        labels = labels[labels.userId.isin(selected)]
        features = label_features(labels, lut, users, items, mappings)
        features.to_parquet(data / f"{split}_features.parquet", index=False)
        print(f"{split}: {len(features):,} rows, {features.userId.nunique():,} users")

if __name__ == "__main__":
    main()

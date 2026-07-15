"""
Train ALS matrix factorization to produce user/item embeddings.
These embeddings feed the FAISS retrieval index in build_faiss_index.py.

Why ALS over a two-tower NN for the retrieval stage:
ALS is the standard, well-understood baseline for implicit-feedback retrieval
and trains fast enough to iterate on at 25M-row scale without a GPU. A
two-tower encoder is a reasonable v2 (documented in reports/future_work.md)
but isn't necessary to demonstrate the retrieval -> ranking architecture,
which is the actual point of this project.
"""

import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from implicit.als import AlternatingLeastSquares

from src.data.load_movielens import build_id_mappings

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "../../data/processed")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")


def build_interaction_matrix(train: pd.DataFrame, user2idx: dict, item2idx: dict) -> sp.csr_matrix:
    """
    Build a user x item confidence matrix from explicit ratings.
    Ratings >= 3.5 are treated as positive implicit signal (a common,
    defensible threshold for MovieLens' 0.5-5.0 scale); confidence scales
    with how far above that threshold the rating is.
    """
    rows = train["userId"].map(user2idx)
    cols = train["movieId"].map(item2idx)

    confidence = np.where(train["rating"] >= 3.5, 1.0 + (train["rating"] - 3.5) * 2, 0.0)

    mask = confidence > 0
    matrix = sp.csr_matrix(
        (confidence[mask], (rows[mask], cols[mask])),
        shape=(len(user2idx), len(item2idx)),
    )
    return matrix


def train_als(interaction_matrix: sp.csr_matrix, factors: int = 64, iterations: int = 15,
              regularization: float = 0.05):
    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
    )
    model.fit(interaction_matrix)
    return model


if __name__ == "__main__":
    train = pd.read_parquet(os.path.join(PROCESSED_DIR, "train.parquet"))
    user2idx, item2idx, idx2item = build_id_mappings(train)

    interaction_matrix = build_interaction_matrix(train, user2idx, item2idx)
    print(f"Interaction matrix: {interaction_matrix.shape}, "
          f"{interaction_matrix.nnz:,} nonzero entries")

    model = train_als(interaction_matrix)

    os.makedirs(MODELS_DIR, exist_ok=True)
    np.save(os.path.join(MODELS_DIR, "item_embeddings.npy"), model.item_factors)
    np.save(os.path.join(MODELS_DIR, "user_embeddings.npy"), model.user_factors)

    import pickle
    with open(os.path.join(MODELS_DIR, "id_mappings.pkl"), "wb") as f:
        pickle.dump({"user2idx": user2idx, "item2idx": item2idx, "idx2item": idx2item}, f)

    print("Saved embeddings and ID mappings to models/")

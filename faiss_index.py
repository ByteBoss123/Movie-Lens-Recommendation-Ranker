"""
Build a FAISS ANN index over item embeddings and retrieve top-N candidates
for a given user embedding.

Uses IndexFlatIP (exact inner-product search) as the baseline, with an
IVF-based approximate index as the latency-optimized alternative -- the
choice between them is exactly the accuracy/latency tradeoff explored in
reports/latency_analysis.md.
"""

import os
import pickle
import numpy as np
import faiss

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../../models")


class CandidateRetriever:
    def __init__(self, index_type: str = "flat", nlist: int = 100):
        self.index_type = index_type
        self.nlist = nlist
        self.index = None
        self.item_embeddings = None
        self.idx2item = None

    def build(self):
        self.item_embeddings = np.load(
            os.path.join(MODELS_DIR, "item_embeddings.npy")
        ).astype("float32")

        with open(os.path.join(MODELS_DIR, "id_mappings.pkl"), "rb") as f:
            mappings = pickle.load(f)
        self.idx2item = mappings["idx2item"]

        dim = self.item_embeddings.shape[1]

        # Normalize for cosine similarity via inner product
        faiss.normalize_L2(self.item_embeddings)

        if self.index_type == "flat":
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.item_embeddings)
        elif self.index_type == "ivf":
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, self.nlist, faiss.METRIC_INNER_PRODUCT)
            self.index.train(self.item_embeddings)
            self.index.add(self.item_embeddings)
            self.index.nprobe = 10  # tunable: higher = more accurate, slower
        else:
            raise ValueError(f"Unknown index_type: {self.index_type}")

        return self

    def retrieve(self, user_embedding: np.ndarray, k: int = 500):
        """Return top-k candidate item indices and scores for a user embedding."""
        query = user_embedding.astype("float32").reshape(1, -1)
        faiss.normalize_L2(query)
        scores, indices = self.index.search(query, k)
        item_ids = [self.idx2item[i] for i in indices[0] if i != -1]
        return item_ids, scores[0]


if __name__ == "__main__":
    retriever = CandidateRetriever(index_type="flat").build()
    print(f"FAISS index built: {retriever.index.ntotal:,} items indexed")

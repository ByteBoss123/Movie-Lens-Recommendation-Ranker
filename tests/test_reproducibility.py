import numpy as np
import pandas as pd
import pytest
from src.features.prepare_ranker import label_features
from src.features.serving_features import FEATURE_COLS, build_lookup_tables

def test_future_labels_cannot_change_features():
    history = pd.DataFrame({'userId': [1, 1, 2, 2], 'movieId': [10, 20, 10, 20],
                            'rating': [4., 3., 2., 5.], 'timestamp': [1, 2, 3, 4]})
    movies = pd.DataFrame({'movieId': [10, 20], 'title': ['A (2000)', 'B (2001)'],
                           'genres': [['Drama'], ['Comedy']]})
    lut = build_lookup_tables(history, movies, 1600000000)
    labels = pd.DataFrame({'userId': [1, 2], 'movieId': [20, 10], 'rating': [5., 1.]})
    mappings = {'user2idx': {1: 0, 2: 1}, 'item2idx': {10: 0, 20: 1}}
    embeddings = np.eye(2, dtype=np.float32)
    first = label_features(labels, lut, embeddings, embeddings, mappings)
    labels['rating'] = [1., 5.]
    second = label_features(labels, lut, embeddings, embeddings, mappings)
    pd.testing.assert_frame_equal(first[FEATURE_COLS], second[FEATURE_COLS])
    assert first.relevance.tolist() != second.relevance.tolist()

def test_retrieval_returns_matching_ids_and_scores_for_small_catalog():
    import faiss
    from src.retrieval.faiss_index import CandidateRetriever
    retriever = CandidateRetriever()
    retriever.index = faiss.IndexFlatIP(2)
    retriever.index.add(np.eye(2, dtype=np.float32))
    retriever.idx2item = {0: 10, 1: 20}
    ids, scores = retriever.retrieve(np.array([1., 0.]), k=500)
    assert len(ids) == len(scores) == 2
    assert np.isfinite(scores).all()

def test_api_imports_from_clean_package():
    from src.serving.app import app
    assert '/recommend' in {route.path for route in app.routes}

def test_ndcg_penalizes_relevant_items_missing_from_candidates():
    from src.eval.metrics import ndcg_at_k
    expected = 1 / (1 + 1 / np.log2(3))
    assert ndcg_at_k([1, 9], {1, 2}, 2) == pytest.approx(expected)
    # Adding a relevant item below k must not change the top-k metric.
    assert ndcg_at_k([1, 9, 2], {1, 2}, 2) == pytest.approx(expected)

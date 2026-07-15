"""
Standard ranking evaluation metrics.

These operate on a single ranked list at a time (a list of item IDs in
predicted order, plus the set of ground-truth relevant items) rather than
on the whole dataset at once, so they compose cleanly with per-user
evaluation loops without needing the full candidate matrix in memory.
"""

import numpy as np


def dcg_at_k(relevance_in_rank_order: list, k: int) -> float:
    """Discounted cumulative gain for a binary/graded relevance list, already
    ordered by the ranking being evaluated (position 0 = top-ranked)."""
    relevance_in_rank_order = relevance_in_rank_order[:k]
    if not relevance_in_rank_order:
        return 0.0
    gains = np.array(relevance_in_rank_order, dtype=float)
    discounts = np.log2(np.arange(2, len(gains) + 2))
    return float(np.sum(gains / discounts))


def ndcg_at_k(ranked_item_ids: list, relevant_item_ids: set, k: int) -> float:
    """
    Normalized DCG@k for one user/query.
    ranked_item_ids: items in the order the system ranked them (best first).
    relevant_item_ids: ground-truth relevant items (e.g. held-out ratings >= 3.5).
    """
    relevance = [1.0 if item in relevant_item_ids else 0.0 for item in ranked_item_ids]
    actual_dcg = dcg_at_k(relevance, k)

    ideal_relevance = sorted(relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


def mrr(ranked_item_ids: list, relevant_item_ids: set) -> float:
    """Reciprocal rank of the first relevant item found; 0 if none appear."""
    for i, item in enumerate(ranked_item_ids):
        if item in relevant_item_ids:
            return 1.0 / (i + 1)
    return 0.0


def recall_at_k(ranked_item_ids: list, relevant_item_ids: set, k: int) -> float:
    """Fraction of ground-truth relevant items that appear in the top-k."""
    if not relevant_item_ids:
        return 0.0
    top_k = set(ranked_item_ids[:k])
    hits = len(top_k & relevant_item_ids)
    return hits / len(relevant_item_ids)


def evaluate_rankings(per_user_rankings: dict, per_user_relevant: dict,
                       ndcg_k: int = 10, recall_k: int = 50) -> dict:
    """
    Aggregate metrics across many users.
    per_user_rankings: {userId: [item_id, item_id, ...]} in ranked order.
    per_user_relevant: {userId: set(item_id, ...)} ground-truth relevant items.
    Users with no relevant items in the held-out period are skipped (there's
    nothing to evaluate against for them).
    """
    ndcgs, mrrs, recalls = [], [], []

    for user_id, ranking in per_user_rankings.items():
        relevant = per_user_relevant.get(user_id, set())
        if not relevant:
            continue
        ndcgs.append(ndcg_at_k(ranking, relevant, ndcg_k))
        mrrs.append(mrr(ranking, relevant))
        recalls.append(recall_at_k(ranking, relevant, recall_k))

    return {
        f"ndcg@{ndcg_k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "mrr": float(np.mean(mrrs)) if mrrs else 0.0,
        f"recall@{recall_k}": float(np.mean(recalls)) if recalls else 0.0,
        "n_users_evaluated": len(ndcgs),
    }

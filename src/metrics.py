"""
ShopSense Recommender Evaluation Metrics
========================================
Standard ranking and classification metrics for recommendation systems:
- Precision@K
- Recall@K
- NDCG@K (Normalized Discounted Cumulative Gain)
- Catalog Coverage@K
"""

import math
from typing import Dict, List, Set, Sequence, Any
import numpy as np


def precision_at_k(actual: Sequence[Any], recommended: Sequence[Any], k: int = 10) -> float:
    """
    Computes Precision@K for a single user.
    Precision@K = (# of relevant items in top-K recommendations) / K
    """
    if k <= 0 or not recommended:
        return 0.0
    rec_k = recommended[:k]
    actual_set = set(actual)
    hits = sum(1 for item in rec_k if item in actual_set)
    return float(hits / k)


def recall_at_k(actual: Sequence[Any], recommended: Sequence[Any], k: int = 10) -> float:
    """
    Computes Recall@K for a single user.
    Recall@K = (# of relevant items in top-K recommendations) / (total # of actual relevant items)
    
    Note: Uses the FULL actual set in the denominator (no artificial truncation bug).
    """
    if not actual or not recommended or k <= 0:
        return 0.0
    rec_k = recommended[:k]
    actual_set = set(actual)
    hits = sum(1 for item in rec_k if item in actual_set)
    return float(hits / len(actual_set))


def ndcg_at_k(actual: Sequence[Any], recommended: Sequence[Any], k: int = 10) -> float:
    """
    Computes NDCG@K for a single user.
    DCG@K = sum_{r=1}^K [ I(rec[r] in actual) / log2(r + 1) ]
    IDCG@K = sum_{r=1}^{min(K, |actual|)} [ 1 / log2(r + 1) ]
    NDCG@K = DCG@K / IDCG@K
    """
    if not actual or not recommended or k <= 0:
        return 0.0

    actual_set = set(actual)
    rec_k = recommended[:k]

    dcg = 0.0
    for rank, item in enumerate(rec_k, start=1):
        if item in actual_set:
            dcg += 1.0 / math.log2(rank + 1)

    ideal_hits = min(len(actual_set), k)
    if ideal_hits == 0:
        return 0.0

    idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0.0 else 0.0


def catalog_coverage_at_k(
    recommendations_dict: Dict[Any, Sequence[Any]],
    catalog_items: Set[Any],
    k: int = 10,
) -> float:
    """
    Computes Catalog Coverage@K:
    Percentage of catalog items that appear at least once across all users' top-K recommendations.
    """
    if not catalog_items:
        return 0.0
    recommended_set = set()
    for recs in recommendations_dict.values():
        recommended_set.update(recs[:k])
    
    covered_items = recommended_set.intersection(catalog_items)
    return float(len(covered_items) / len(catalog_items)) * 100.0


def evaluate_recommendations(
    actuals_dict: Dict[Any, Sequence[Any]],
    recommendations_dict: Dict[Any, Sequence[Any]],
    k: int = 10,
) -> Dict[str, float]:
    """
    Computes mean Precision@K, Recall@K, NDCG@K across all evaluated users.
    """
    precisions, recalls, ndcgs = [], [], []

    for user_id, actual_items in actuals_dict.items():
        recs = recommendations_dict.get(user_id, [])
        precisions.append(precision_at_k(actual_items, recs, k=k))
        recalls.append(recall_at_k(actual_items, recs, k=k))
        ndcgs.append(ndcg_at_k(actual_items, recs, k=k))

    return {
        f"precision@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"recall@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "n_users_evaluated": len(actuals_dict),
    }

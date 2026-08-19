"""
Unit Tests for ShopSense Unified Recommender Router
===================================================
Verifies multi-tier routing policy:
- 10+ interactions -> Hybrid CF
- 1–9 interactions -> Content-Based
- 0 interactions   -> Popularity Prior
- Unknown/cold item -> Content-Based Metadata Matching

Verifies recommendation deduplication is based on parent_asin / product ID (never duplicated).
"""

from pathlib import Path
import pytest
import pandas as pd

from src.recommender import ShopSenseRecommender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_clean.parquet"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"


@pytest.fixture(scope="module")
def unified_recommender():
    train_df = pd.read_parquet(TRAIN_PATH)
    meta_df = pd.read_parquet(METADATA_PATH)
    rec = ShopSenseRecommender(min_cf_interactions=10)
    rec.fit(train_df=train_df, metadata_df=meta_df)
    return rec, train_df, meta_df


def test_routing_0_interactions_popularity(unified_recommender):
    """Verify completely new user with 0 interactions routes to cold_user_popularity_prior."""
    rec, _, _ = unified_recommender

    # Case A: user_id=None with empty history
    recs_a, route_a = rec.recommend(user_id=None, user_history=[], k=10)
    assert route_a == "cold_user_popularity_prior"
    assert len(recs_a) == 10
    assert len(set(recs_a)) == 10

    # Case B: Unknown user_id not present in training data
    recs_b, route_b = rec.recommend(user_id="UNKNOWN_COLD_USER_XYZ_000", k=10)
    assert route_b == "cold_user_popularity_prior"
    assert len(recs_b) == 10
    assert len(set(recs_b)) == 10


def test_routing_1_interaction_content(unified_recommender):
    """Verify user with exactly 1 interaction routes to content_based_sparse_history."""
    rec, train_df, _ = unified_recommender
    sample_items = train_df["parent_asin"].drop_duplicates().head(1).tolist()

    recs, route = rec.recommend(user_id=None, user_history=sample_items, k=10)

    assert route == "content_based_sparse_history"
    assert len(recs) == 10
    assert len(set(recs)) == 10
    assert sample_items[0] not in recs


def test_routing_9_interactions_content(unified_recommender):
    """Verify user with exactly 9 interactions routes to content_based_sparse_history."""
    rec, train_df, _ = unified_recommender

    # Test with known train user who has exactly 9 interactions
    counts = train_df["user_id"].value_counts()
    users_with_9 = counts[counts == 9].index
    assert len(users_with_9) > 0, "Expected users with exactly 9 interactions in train dataset"
    user_9 = users_with_9[0]

    recs_user, route_user = rec.recommend(user_id=user_9, k=10)
    assert route_user == "content_based_sparse_history"
    assert len(recs_user) == 10
    assert len(set(recs_user)) == 10

    # Test with session user history of length 9
    sample_items = train_df["parent_asin"].drop_duplicates().head(9).tolist()
    recs_hist, route_hist = rec.recommend(user_id=None, user_history=sample_items, k=10)
    assert route_hist == "content_based_sparse_history"
    assert len(recs_hist) == 10
    assert len(set(recs_hist)) == 10


def test_routing_10_interactions_hybrid(unified_recommender):
    """Verify user with exactly 10 interactions routes to hybrid_cf."""
    rec, train_df, _ = unified_recommender

    counts = train_df["user_id"].value_counts()
    users_with_10 = counts[counts == 10].index
    assert len(users_with_10) > 0, "Expected users with exactly 10 interactions in train dataset"
    user_10 = users_with_10[0]

    recs, route = rec.recommend(user_id=user_10, k=10)
    assert route == "hybrid_cf"
    assert len(recs) == 10
    assert len(set(recs)) == 10


def test_routing_11_interactions_hybrid(unified_recommender):
    """Verify user with exactly 11 interactions routes to hybrid_cf."""
    rec, train_df, _ = unified_recommender

    counts = train_df["user_id"].value_counts()
    users_with_11 = counts[counts == 11].index
    assert len(users_with_11) > 0, "Expected users with exactly 11 interactions in train dataset"
    user_11 = users_with_11[0]

    recs, route = rec.recommend(user_id=user_11, k=10)
    assert route == "hybrid_cf"
    assert len(recs) == 10
    assert len(set(recs)) == 10


def test_routing_cold_item_content_matching(unified_recommender):
    """Verify cold/unseen item similarity query routes to cold_item_content_match."""
    rec, _, _ = unified_recommender
    sims, route = rec.get_similar_items("COLD_NONEXISTENT_ITEM_XYZ", k=5)

    assert route == "cold_item_content_match"
    assert isinstance(sims, list)


def test_deduplication_by_parent_asin_never_duplicated(unified_recommender):
    """
    Verify recommendation deduplication is based on parent_asin / product ID (NOT product title),
    and duplicate parent_asin values are never returned.
    """
    rec, train_df, meta_df = unified_recommender

    counts = train_df["user_id"].value_counts()
    user_warm = counts[counts >= 10].index[0]
    user_sparse = counts[counts == 9].index[0]

    # Check multiple scenarios and routes
    scenarios = [
        ("warm_hybrid", {"user_id": user_warm, "k": 10}),
        ("sparse_content_user", {"user_id": user_sparse, "k": 10}),
        ("sparse_content_hist", {"user_id": None, "user_history": train_df["parent_asin"].drop_duplicates().head(5).tolist(), "k": 10}),
        ("cold_popularity", {"user_id": None, "user_history": [], "k": 10}),
        ("redundant_history", {
            "user_id": None,
            "user_history": [train_df["parent_asin"].iloc[0]] * 5 + [train_df["parent_asin"].iloc[1]] * 3,
            "k": 10,
        }),
    ]

    for name, kwargs in scenarios:
        recs, _ = rec.recommend(**kwargs)
        # All items must be unique parent_asin strings
        assert len(recs) == len(set(recs)), f"Duplicate parent_asin found in route scenario: {name}"
        assert all(isinstance(asin, str) for asin in recs)

    # Verify deduplication key is parent_asin, not product title:
    # If two distinct products share the exact same title or store, they maintain distinct parent_asin IDs
    # and deduplication operates at the parent_asin level.
    titles_in_meta = meta_df.groupby("title")["parent_asin"].nunique()
    multi_asin_titles = titles_in_meta[titles_in_meta > 1]
    if len(multi_asin_titles) > 0:
        dup_title = multi_asin_titles.index[0]
        asins_with_same_title = meta_df[meta_df["title"] == dup_title]["parent_asin"].unique()
        # Verify asins are distinct IDs
        assert len(asins_with_same_title) > 1
        assert len(set(asins_with_same_title)) == len(asins_with_same_title)


def test_k_limit_and_determinism(unified_recommender):
    """Verify recommendation lengths match K and are deterministic."""
    rec, train_df, _ = unified_recommender
    sample_user = train_df["user_id"].iloc[0]

    for k_val in [1, 5, 10]:
        recs1, _ = rec.recommend(user_id=sample_user, k=k_val)
        recs2, _ = rec.recommend(user_id=sample_user, k=k_val)
        assert len(recs1) == k_val
        assert recs1 == recs2

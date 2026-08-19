"""
Unit Tests for ShopSense Hybrid Recommendation System
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.hybrid import HybridRecommender
from src.baseline import PopularityRecommender
from src.user_cf import UserBasedCF
from src.item_cf import ItemBasedCF

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"


@pytest.fixture(scope="module")
def train_data():
    return pd.read_parquet(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_data():
    return pd.read_parquet(TEST_PATH)


@pytest.fixture(scope="module")
def hybrid_model(train_data):
    model = HybridRecommender(w_pop=0.6, w_user=0.2, w_item=0.2)
    model.fit(train_data)
    return model


def test_weight_validation():
    """Verify weight validation and normalization."""
    # Valid non-normalized weights
    h = HybridRecommender(w_pop=6, w_user=2, w_item=2)
    assert pytest.approx(h.w_pop, 0.01) == 0.6
    assert pytest.approx(h.w_user, 0.01) == 0.2
    assert pytest.approx(h.w_item, 0.01) == 0.2
    assert pytest.approx(h.w_pop + h.w_user + h.w_item, 0.001) == 1.0

    # Negative weights should raise error
    with pytest.raises(ValueError):
        HybridRecommender(w_pop=-0.5, w_user=0.5, w_item=0.5)

    # All-zero weights should raise error
    with pytest.raises(ValueError):
        HybridRecommender(w_pop=0.0, w_user=0.0, w_item=0.0)


def test_score_normalization(hybrid_model):
    """Verify Min-Max score normalization maps values to [0, 1]."""
    raw_scores = {"item_A": 100.0, "item_B": 50.0, "item_C": 0.0}
    norm_scores = hybrid_model._normalize_scores(raw_scores)

    assert norm_scores["item_A"] == 1.0
    assert norm_scores["item_B"] == 0.5
    assert norm_scores["item_C"] == 0.0

    # Empty dictionary check
    assert hybrid_model._normalize_scores({}) == {}


def test_no_test_leakage(hybrid_model, test_data):
    """Verify hybrid catalog has zero knowledge of test-only items."""
    train_catalog = hybrid_model.catalog_items
    test_only_items = set(test_data["parent_asin"].unique()) - train_catalog

    for item in test_only_items:
        assert item not in hybrid_model.pop_scores_norm
        assert item not in hybrid_model.catalog_items


def test_seen_item_exclusion(hybrid_model, train_data):
    """Verify that hybrid recommender never recommends items the user already rated in train."""
    sample_users = train_data["user_id"].drop_duplicates().head(50).tolist()

    for uid in sample_users:
        seen = hybrid_model.user_seen_items.get(uid, set())
        recs = hybrid_model.recommend(user_id=uid, k=10, exclude_seen=True)
        assert len(set(recs).intersection(seen)) == 0, f"User {uid} was recommended a seen item!"


def test_deterministic_recommendations(hybrid_model, train_data):
    """Verify recommendations are deterministic on repeated runs."""
    sample_user = train_data["user_id"].iloc[0]
    recs1 = hybrid_model.recommend(user_id=sample_user, k=10)
    recs2 = hybrid_model.recommend(user_id=sample_user, k=10)
    assert recs1 == recs2


def test_no_duplicate_recommendations(hybrid_model, train_data):
    """Verify recommendation lists contain strictly unique items."""
    sample_users = train_data["user_id"].drop_duplicates().head(20).tolist()
    for uid in sample_users:
        recs = hybrid_model.recommend(user_id=uid, k=15)
        assert len(recs) == len(set(recs)), f"Duplicate items found in recs for {uid}"


def test_k_limit_respected(hybrid_model, train_data):
    """Verify K parameter controls recommendation list length."""
    sample_user = train_data["user_id"].iloc[0]
    for k_val in [1, 5, 10, 20]:
        recs = hybrid_model.recommend(user_id=sample_user, k=k_val)
        assert len(recs) == k_val


def test_unknown_user_handling(hybrid_model):
    """Verify unknown user gracefully gets popularity recommendations."""
    unknown_user = "COLD_USER_NONEXISTENT_999"
    recs = hybrid_model.recommend(user_id=unknown_user, k=10)
    assert len(recs) == 10
    assert recs == hybrid_model.pop_model.recommend(user_id=unknown_user, k=10)


def test_predict_rating_bounds(hybrid_model, train_data):
    """Verify predicted ratings are always clipped to [1.0, 5.0]."""
    sample_rows = train_data.head(30)
    for _, row in sample_rows.iterrows():
        pred = hybrid_model.predict_rating(row["user_id"], row["parent_asin"])
        assert 1.0 <= pred <= 5.0
        assert isinstance(pred, float)

    # Test unknown user/item pair
    pred_unknown = hybrid_model.predict_rating("FAKE_USER", "FAKE_ITEM")
    assert 1.0 <= pred_unknown <= 5.0

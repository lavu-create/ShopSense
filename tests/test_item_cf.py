"""
Unit Tests for ShopSense Item-Based Collaborative Filtering Recommender
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.item_cf import ItemBasedCF
from src.matrix_builder import UserItemMatrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = PROJECT_ROOT / "data" / "processed" / "test.parquet"


@pytest.fixture(scope="module")
def train_data():
    return pd.read_parquet(TRAIN_PATH)


@pytest.fixture(scope="module")
def test_data():
    if TEST_PATH.exists():
        return pd.read_parquet(TEST_PATH)
    return pd.DataFrame({
        "user_id": ["TEST_USER_001", "TEST_USER_002"],
        "parent_asin": ["TEST_UNSEEN_ITEM_001", "TEST_UNSEEN_ITEM_002"],
        "rating": [5.0, 4.0],
        "timestamp": [1700000000000, 1700000001000],
    })


@pytest.fixture(scope="module")
def item_cf_model(train_data):
    model = ItemBasedCF(n_neighbors=10, similarity_method="cosine")
    model.fit(train_data)
    return model


def test_sparse_item_similarity_construction(item_cf_model, train_data):
    """Verify sparse similarity matrix dimensions and properties."""
    assert item_cf_model.S_sparse is not None
    assert item_cf_model.S_sparse.shape == (item_cf_model.matrix.n_items, item_cf_model.matrix.n_items)
    assert item_cf_model.S_sparse.nnz > 0
    # Diagonal must be 0 (no self-similarity)
    diag = item_cf_model.S_sparse.diagonal()
    assert np.all(diag == 0.0)


def test_item_mappings(item_cf_model, train_data):
    """Verify item index mappings match train catalog."""
    matrix = item_cf_model.matrix
    assert matrix.n_items == train_data["parent_asin"].nunique()
    for idx, item in enumerate(matrix.idx_to_item[:100]):
        assert matrix.item_to_idx[item] == idx


def test_train_only_model_construction(item_cf_model, test_data):
    """Verify item-CF has zero knowledge of test-only items."""
    train_catalog = item_cf_model.catalog_items
    test_only_items = set(test_data["parent_asin"].unique()) - train_catalog

    for item in test_only_items:
        assert item not in item_cf_model.matrix.item_to_idx
        assert item not in item_cf_model.catalog_items


def test_seen_item_exclusion(item_cf_model, train_data):
    """Verify that item-based CF never recommends an item the user already rated in train."""
    sample_users = train_data["user_id"].drop_duplicates().head(50).tolist()

    for uid in sample_users:
        seen = item_cf_model.user_seen_items.get(uid, set())
        recs = item_cf_model.recommend(user_id=uid, k=10, exclude_seen=True)
        assert len(set(recs).intersection(seen)) == 0, f"User {uid} was recommended a seen item!"


def test_deterministic_recommendations(item_cf_model, train_data):
    """Verify recommendations are deterministic on repeated runs."""
    sample_user = train_data["user_id"].iloc[0]
    recs1 = item_cf_model.recommend(user_id=sample_user, k=10)
    recs2 = item_cf_model.recommend(user_id=sample_user, k=10)
    assert recs1 == recs2


def test_no_duplicate_recommendations(item_cf_model, train_data):
    """Verify recommendation lists contain strictly unique items."""
    sample_users = train_data["user_id"].drop_duplicates().head(20).tolist()
    for uid in sample_users:
        recs = item_cf_model.recommend(user_id=uid, k=15)
        assert len(recs) == len(set(recs)), f"Duplicate items found in recs for {uid}"


def test_k_limit_respected(item_cf_model, train_data):
    """Verify K parameter controls recommendation list length."""
    sample_user = train_data["user_id"].iloc[0]
    for k_val in [1, 5, 10, 20]:
        recs = item_cf_model.recommend(user_id=sample_user, k=k_val)
        assert len(recs) == k_val


def test_unknown_user_fallback(item_cf_model):
    """Verify unknown user gracefully falls back to popularity without errors."""
    unknown_user = "COLD_USER_NONEXISTENT_999"
    recs = item_cf_model.recommend(user_id=unknown_user, k=10)
    assert len(recs) == 10
    assert recs == item_cf_model.fallback_model.recommend(user_id=unknown_user, k=10)


def test_get_similar_items(item_cf_model, train_data):
    """Verify similar items retrieval returns positive similarity and excludes self."""
    sample_item = train_data["parent_asin"].iloc[0]
    similar_items = item_cf_model.get_similar_items(item_id=sample_item, n_neighbors=5)

    if similar_items:
        for sim_id, sim_score in similar_items:
            assert sim_id != sample_item
            assert sim_score > 0.0


def test_predicted_ratings_range(item_cf_model, train_data):
    """Verify predicted ratings are always clipped to [1.0, 5.0]."""
    sample_rows = train_data.head(30)
    for _, row in sample_rows.iterrows():
        pred = item_cf_model.predict_rating(row["user_id"], row["parent_asin"])
        assert 1.0 <= pred <= 5.0
        assert isinstance(pred, float)

    # Test on completely unseen user/item pair
    pred_unknown = item_cf_model.predict_rating("FAKE_USER", "FAKE_ITEM")
    assert 1.0 <= pred_unknown <= 5.0

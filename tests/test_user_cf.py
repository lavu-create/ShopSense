"""
Unit Tests for ShopSense User-Based Collaborative Filtering Recommender
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.user_cf import UserBasedCF
from src.matrix_builder import UserItemMatrix

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
def user_cf_model(train_data):
    model = UserBasedCF(n_neighbors=20)
    model.fit(train_data)
    return model


def test_sparse_matrix_construction(user_cf_model, train_data):
    """Verify sparse matrix properties and dimensions."""
    matrix = user_cf_model.matrix
    assert matrix.R is not None
    assert matrix.R_norm is not None
    assert matrix.n_users == train_data["user_id"].nunique()
    assert matrix.n_items == train_data["parent_asin"].nunique()
    assert matrix.R.shape == (matrix.n_users, matrix.n_items)
    assert matrix.R.nnz > 0


def test_user_item_bidirectional_mappings(user_cf_model):
    """Verify bidirectional index mappings are consistent."""
    matrix = user_cf_model.matrix
    for idx, uid in enumerate(matrix.idx_to_user[:100]):
        assert matrix.user_to_idx[uid] == idx

    for idx, item in enumerate(matrix.idx_to_item[:100]):
        assert matrix.item_to_idx[item] == idx


def test_no_test_leakage(user_cf_model, test_data):
    """Verify model contains no items or knowledge from test-only set."""
    train_catalog = user_cf_model.catalog_items
    test_only_items = set(test_data["parent_asin"].unique()) - train_catalog

    for item in test_only_items:
        assert item not in user_cf_model.matrix.item_to_idx
        assert item not in user_cf_model.catalog_items


def test_seen_item_exclusion(user_cf_model, train_data):
    """Verify that user-based CF never recommends an item the user already rated in train."""
    sample_users = train_data["user_id"].drop_duplicates().head(50).tolist()

    for uid in sample_users:
        seen = user_cf_model.user_seen_items.get(uid, set())
        recs = user_cf_model.recommend(user_id=uid, k=10, exclude_seen=True)
        assert len(set(recs).intersection(seen)) == 0, f"User {uid} was recommended a seen item!"


def test_deterministic_recommendations(user_cf_model, train_data):
    """Verify recommendations are deterministic on repeated runs."""
    sample_user = train_data["user_id"].iloc[0]
    recs1 = user_cf_model.recommend(user_id=sample_user, k=10)
    recs2 = user_cf_model.recommend(user_id=sample_user, k=10)
    assert recs1 == recs2


def test_no_duplicate_recommendations(user_cf_model, train_data):
    """Verify recommendation lists contain strictly unique items."""
    sample_users = train_data["user_id"].drop_duplicates().head(20).tolist()
    for uid in sample_users:
        recs = user_cf_model.recommend(user_id=uid, k=15)
        assert len(recs) == len(set(recs)), f"Duplicate items found in recs for {uid}"


def test_k_limit_respected(user_cf_model, train_data):
    """Verify K parameter controls recommendation list length."""
    sample_user = train_data["user_id"].iloc[0]
    for k_val in [1, 5, 10, 20]:
        recs = user_cf_model.recommend(user_id=sample_user, k=k_val)
        assert len(recs) == k_val


def test_unknown_user_fallback(user_cf_model):
    """Verify unknown user gracefully falls back to popularity without errors."""
    unknown_user = "COLD_USER_NONEXISTENT_999"
    recs = user_cf_model.recommend(user_id=unknown_user, k=10)
    assert len(recs) == 10
    assert recs == user_cf_model.fallback_model.recommend(user_id=unknown_user, k=10)


def test_get_similar_users(user_cf_model, train_data):
    """Verify neighbor retrieval returns positive similarity and excludes self."""
    sample_user = train_data["user_id"].iloc[0]
    neighbors = user_cf_model.get_similar_users(user_id=sample_user, n_neighbors=10)
    
    if neighbors:
        for neighbor_id, sim in neighbors:
            assert neighbor_id != sample_user
            assert sim > 0.0


def test_predicted_ratings_range(user_cf_model, train_data):
    """Verify predicted ratings are always clipped to [1.0, 5.0]."""
    sample_rows = train_data.head(30)
    for _, row in sample_rows.iterrows():
        pred = user_cf_model.predict_rating(row["user_id"], row["parent_asin"])
        assert 1.0 <= pred <= 5.0
        assert isinstance(pred, float)

    # Test on completely unseen user/item pair
    pred_unknown = user_cf_model.predict_rating("FAKE_USER", "FAKE_ITEM")
    assert 1.0 <= pred_unknown <= 5.0

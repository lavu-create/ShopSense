"""
Unit Tests for ShopSense Popularity Recommender Baseline
"""

import pytest
import pandas as pd
from pathlib import Path
from src.baseline import PopularityRecommender

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
def fitted_model(train_data):
    model = PopularityRecommender(m_threshold=5)
    model.fit(train_data)
    return model


def test_fit_uses_train_only(fitted_model, test_data):
    """Verify the recommender only knows items from train catalog."""
    train_catalog = fitted_model.catalog_items
    
    # Identify items present in test but not train
    test_only_items = set(test_data["parent_asin"].unique()) - train_catalog
    
    # Ensure no test-only item is in the model's catalog or ranked list
    for item in test_only_items:
        assert item not in train_catalog
        assert item not in fitted_model.ranked_item_ids


def test_no_seen_items_recommended(fitted_model, train_data):
    """Verify that a user is never recommended items they already rated in train."""
    # Test on a sample of 50 users
    sample_users = train_data["user_id"].drop_duplicates().head(50).tolist()
    
    for uid in sample_users:
        seen = fitted_model.user_seen_items.get(uid, set())
        recs = fitted_model.recommend(user_id=uid, k=10, exclude_seen=True)
        
        # Intersection between seen and recommended must be empty
        assert len(set(recs).intersection(seen)) == 0, f"User {uid} was recommended a seen item!"


def test_recommendations_are_deterministic(fitted_model, train_data):
    """Verify recommendations for the same user are identical on repeated calls."""
    sample_user = train_data["user_id"].iloc[0]
    
    recs_1 = fitted_model.recommend(user_id=sample_user, k=10)
    recs_2 = fitted_model.recommend(user_id=sample_user, k=10)
    
    assert recs_1 == recs_2


def test_no_duplicate_products_in_recommendations(fitted_model, train_data):
    """Verify recommendation list contains strictly unique products."""
    sample_users = train_data["user_id"].drop_duplicates().head(20).tolist()
    
    for uid in sample_users:
        recs = fitted_model.recommend(user_id=uid, k=20)
        assert len(recs) == len(set(recs)), f"Duplicate items found in recs for {uid}"


def test_k_parameter_respected(fitted_model, train_data):
    """Verify that the length of the recommendation list matches K."""
    sample_user = train_data["user_id"].iloc[0]
    
    for k_val in [1, 5, 10, 25]:
        recs = fitted_model.recommend(user_id=sample_user, k=k_val)
        assert len(recs) == k_val


def test_unknown_user_fallback(fitted_model):
    """Verify that an unseen/cold-start user gets the global top-K without errors."""
    unknown_user = "UNKNOWN_NEW_USER_XYZ"
    recs = fitted_model.recommend(user_id=unknown_user, k=10)
    
    assert len(recs) == 10
    # For unknown user with no seen history, recs should be the absolute top 10 ranked items
    assert recs == fitted_model.ranked_item_ids[:10]

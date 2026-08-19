"""
Unit Tests for ShopSense Content-Based Recommender
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path

from src.content_based import ContentBasedRecommender

PROJECT_ROOT = Path(__file__).resolve().parent.parent
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_clean.parquet"
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"


@pytest.fixture(scope="module")
def sample_meta():
    meta_df = pd.read_parquet(METADATA_PATH)
    return meta_df.head(1000).copy()


@pytest.fixture(scope="module")
def content_model(sample_meta):
    model = ContentBasedRecommender(max_features=1000)
    model.fit(sample_meta)
    return model


def test_tfidf_matrix_properties(content_model, sample_meta):
    """Verify sparse TF-IDF matrix construction."""
    assert content_model.X_tfidf is not None
    assert content_model.X_tfidf.shape[0] == sample_meta["parent_asin"].nunique()
    assert content_model.X_tfidf.shape[1] <= 1000
    assert content_model.X_tfidf.nnz > 0


def test_similar_items_retrieval(content_model, sample_meta):
    """Verify similar items are returned with positive similarity and exclude self."""
    target_item = sample_meta["parent_asin"].iloc[0]
    sims = content_model.get_similar_items(target_item, k=5)

    if sims:
        for sim_id, sim_score in sims:
            assert sim_id != target_item
            assert sim_score > 0.0


def test_recommend_for_user_history(content_model, sample_meta):
    """Verify recommendations from user history exclude seen items and return unique items."""
    history = sample_meta["parent_asin"].head(3).tolist()
    recs = content_model.recommend_for_user_history(history, k=5, exclude_seen=True)

    assert len(recs) <= 5
    assert len(set(recs)) == len(recs)
    for h in history:
        assert h not in recs


def test_missing_metadata_handling(content_model):
    """Verify graceful handling when product metadata contains NaNs."""
    corrupted_df = pd.DataFrame([
        {"parent_asin": "ITEM_NAN_1", "title": None, "main_category": None, "store": None, "categories_flat": None},
        {"parent_asin": "ITEM_NAN_2", "title": "Valid Guitar", "main_category": "Musical Instruments", "store": "Fender", "categories_flat": "Guitars"},
    ])
    model = ContentBasedRecommender()
    model.fit(corrupted_df)
    assert model.is_fitted
    sims = model.get_similar_items("ITEM_NAN_2", k=1)
    assert isinstance(sims, list)


def test_cold_user_recommendations(content_model):
    """Verify cold user fallback returns recommendations without exceptions."""
    recs = content_model.recommend_for_cold_user(k=5)
    assert isinstance(recs, list)

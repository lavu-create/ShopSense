"""
Unit tests for ShopSense Preprocessor & Chronological Split
"""

import pytest
import pandas as pd
from pathlib import Path
from src.preprocessor import verify_split_integrity

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def test_data_files_exist():
    """Checks full benchmark dataset files if present locally."""
    if not (PROCESSED_DIR / "test.parquet").exists() or not (PROCESSED_DIR / "interactions_filtered.parquet").exists():
        pytest.skip(
            "Full benchmark artifacts (interactions_filtered.parquet, test.parquet) "
            "are excluded from Git (.gitignore). Run src.data_loader and src.preprocessor "
            "to regenerate full dataset."
        )
    assert (PROCESSED_DIR / "interactions_filtered.parquet").exists()
    assert (PROCESSED_DIR / "train.parquet").exists()
    assert (PROCESSED_DIR / "test.parquet").exists()


def test_split_assertions():
    """Validates split integrity and record counts on full benchmark dataset if present."""
    if not (PROCESSED_DIR / "test.parquet").exists() or not (PROCESSED_DIR / "interactions_filtered.parquet").exists():
        pytest.skip(
            "Full benchmark artifacts (interactions_filtered.parquet, test.parquet) "
            "are excluded from Git (.gitignore). Full benchmark assertions skipped."
        )
    filt_df = pd.read_parquet(PROCESSED_DIR / "interactions_filtered.parquet")
    train_df = pd.read_parquet(PROCESSED_DIR / "train.parquet")
    test_df = pd.read_parquet(PROCESSED_DIR / "test.parquet")

    # Run verification suite
    stats = verify_split_integrity(filt_df, train_df, test_df)
    assert stats["all_passed"] is True
    assert len(train_df) == 136814
    assert len(test_df) == 39314
    assert len(filt_df) == 176128
    assert train_df["user_id"].nunique() == 10678
    assert test_df["user_id"].nunique() == 10678


def test_synthetic_k_core_and_chronological_split():
    """Unit test: Verifies k-core pruning logic, chronological ordering, and integrity assertions on synthetic data."""
    from src.preprocessor import apply_iterative_k_core, chronological_per_user_split

    records = []
    base_ts = 1600000000000
    items = [f"ITEM_{i}" for i in range(5)]

    # Users 1 and 2: 10 interactions each across items 0-4
    for u in ["USER_1", "USER_2"]:
        for i in range(10):
            records.append({
                "user_id": u,
                "parent_asin": items[i % 5],
                "rating": 5.0,
                "timestamp": base_ts + i * 1000,
            })
    # User 3: only 2 interactions (should be pruned by min_user_k=5)
    for i in range(2):
        records.append({
            "user_id": "USER_3",
            "parent_asin": "ITEM_0",
            "rating": 4.0,
            "timestamp": base_ts + i * 1000,
        })

    raw_df = pd.DataFrame(records)

    # 1. Test k-core pruning
    filtered_df, iters = apply_iterative_k_core(raw_df, min_user_k=5, min_item_k=2)
    assert "USER_3" not in filtered_df["user_id"].unique()
    assert len(filtered_df["user_id"].unique()) == 2
    assert len(filtered_df) == 20

    # 2. Test chronological split 80/20
    train_df, test_df = chronological_per_user_split(filtered_df, train_ratio=0.80)
    assert len(train_df) == 16  # 8 per user
    assert len(test_df) == 4    # 2 per user

    # 3. Test integrity verification suite
    stats = verify_split_integrity(filtered_df, train_df, test_df)
    assert stats["all_passed"] is True
    assert stats["n_warm_interactions"] == 4
    assert stats["n_cold_interactions"] == 0

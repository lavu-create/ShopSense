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
    assert (PROCESSED_DIR / "interactions_filtered.parquet").exists()
    assert (PROCESSED_DIR / "train.parquet").exists()
    assert (PROCESSED_DIR / "test.parquet").exists()


def test_split_assertions():
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

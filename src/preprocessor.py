"""
ShopSense Dataset Preprocessor & Leakage-Safe Chronological Splitter
===================================================================
Applies iterative k-core filtering (Configuration D: Users >= 10, Items >= 5)
and creates a leakage-safe, per-user chronological train/test split (80/20).

Outputs:
  - data/processed/interactions_filtered.parquet
  - data/processed/train.parquet
  - data/processed/test.parquet
  - data/samples/train_sample.csv
  - data/samples/test_sample.csv

Usage:
  python -m src.preprocessor
"""

import math
import time
from pathlib import Path
from typing import Dict, Tuple, Any

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

INPUT_REVIEWS_PARQUET = PROCESSED_DIR / "reviews_clean.parquet"
OUTPUT_FILTERED_PARQUET = PROCESSED_DIR / "interactions_filtered.parquet"
OUTPUT_TRAIN_PARQUET = PROCESSED_DIR / "train.parquet"
OUTPUT_TEST_PARQUET = PROCESSED_DIR / "test.parquet"

# Thresholds for Configuration D
MIN_USER_INTERACTIONS = 10
MIN_ITEM_INTERACTIONS = 5
TRAIN_RATIO = 0.80


# ---------------------------------------------------------------------------
# Iterative k-Core Filtering
# ---------------------------------------------------------------------------

def apply_iterative_k_core(
    df: pd.DataFrame,
    min_user_k: int = MIN_USER_INTERACTIONS,
    min_item_k: int = MIN_ITEM_INTERACTIONS,
) -> Tuple[pd.DataFrame, int]:
    """
    Iteratively filter users and items until all remaining users have >= min_user_k
    interactions and all remaining items have >= min_item_k interactions.
    """
    print(f"\n🔄 Running iterative k-core pruning (Users >= {min_user_k}, Items >= {min_item_k})...")
    curr = df.copy()
    iteration = 0
    start_time = time.time()

    while True:
        iteration += 1
        n_start = len(curr)

        # 1. Filter users
        user_counts = curr["user_id"].value_counts()
        valid_users = user_counts[user_counts >= min_user_k].index
        curr = curr[curr["user_id"].isin(valid_users)]

        # 2. Filter items
        item_counts = curr["parent_asin"].value_counts()
        valid_items = item_counts[item_counts >= min_item_k].index
        curr = curr[curr["parent_asin"].isin(valid_items)]

        n_end = len(curr)
        print(f"   Iteration {iteration:2d}: {n_end:>10,} interactions remaining (dropped {n_start - n_end:,})")

        if n_start == n_end:
            break

    elapsed = time.time() - start_time
    print(f"  ✓ Converged in {iteration} iterations ({elapsed:.2f}s)")
    
    # Ensure timestamp_dt is present
    if "timestamp_dt" not in curr.columns and "timestamp" in curr.columns:
        curr["timestamp_dt"] = pd.to_datetime(curr["timestamp"], unit="ms", errors="coerce")

    return curr.reset_index(drop=True), iteration


# ---------------------------------------------------------------------------
# Chronological Train/Test Split
# ---------------------------------------------------------------------------

def chronological_per_user_split(
    df: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits interactions chronologically per user:
      - Sorts each user's history by timestamp ascending
      - First 80% -> train
      - Remaining 20% -> test
    Guarantees no temporal leakage and that every user has history in train.
    """
    print(f"\n⏳ Creating per-user chronological split ({int(train_ratio*100)}/{int((1-train_ratio)*100)})...")
    start_time = time.time()

    # Ensure timestamp_dt is present
    if "timestamp_dt" not in df.columns and "timestamp" in df.columns:
        df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")

    # Sort globally by user_id, timestamp ascending, with stable parent_asin tiebreaker
    df_sorted = df.sort_values(
        by=["user_id", "timestamp", "parent_asin"],
        ascending=[True, True, True]
    ).reset_index(drop=True)

    # Compute per-user ranks and cumulative counts
    # cumcount() gives 0, 1, 2, ..., n-1
    df_sorted["user_rank"] = df_sorted.groupby("user_id").cumcount()
    user_sizes = df_sorted.groupby("user_id")["user_id"].transform("count")

    # Train cutoff index: for n=10, 10*0.8=8 -> ranks 0..7 (8 items) go to train, ranks 8..9 (2 items) go to test
    df_sorted["train_cutoff"] = (user_sizes * train_ratio).apply(math.floor).astype(int)

    # Boolean mask for train
    is_train = df_sorted["user_rank"] < df_sorted["train_cutoff"]

    cols_to_keep = ["user_id", "parent_asin", "rating", "timestamp", "timestamp_dt"]
    train_df = df_sorted[is_train][cols_to_keep].reset_index(drop=True)
    test_df = df_sorted[~is_train][cols_to_keep].reset_index(drop=True)

    elapsed = time.time() - start_time
    print(f"  ✓ Split complete in {elapsed:.2f}s")
    print(f"    Train: {len(train_df):,} interactions")
    print(f"    Test:  {len(test_df):,} interactions")

    return train_df, test_df


# ---------------------------------------------------------------------------
# Strict Verification & Assertions
# ---------------------------------------------------------------------------

def verify_split_integrity(
    full_filtered_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Runs strict integrity assertions on the chronological split:
    1. Total rows match
    2. Every user in filtered dataset is in train
    3. Every user in test exists in train
    4. Chronological integrity: min(test_ts) >= max(train_ts) per user
    5. Minimum historical interaction constraint: every user has >= 8 train interactions
    6. Cold-start / unseen item measurement
    """
    print("\n🔍 Verifying split integrity and running assertions...")

    # Check 1: Row count conservation
    assert len(train_df) + len(test_df) == len(full_filtered_df), "Row counts do not sum to total!"
    print("  ✓ [Pass] Row counts perfectly conserved (Train + Test == Filtered Total)")

    # Check 2: User conservation
    filtered_users = set(full_filtered_df["user_id"].unique())
    train_users = set(train_df["user_id"].unique())
    test_users = set(test_df["user_id"].unique())

    assert train_users == filtered_users, "Some filtered users are missing from training set!"
    assert test_users.issubset(train_users), "Test set contains users not present in training set!"
    print(f"  ✓ [Pass] All {len(train_users):,} users present in training set")
    print(f"  ✓ [Pass] All {len(test_users):,} test users exist in training set")

    # Check 3: Minimum train interactions per user
    min_train_per_user = train_df.groupby("user_id").size().min()
    min_test_per_user = test_df.groupby("user_id").size().min()
    assert min_train_per_user >= int(MIN_USER_INTERACTIONS * TRAIN_RATIO), f"User has fewer than {int(MIN_USER_INTERACTIONS * TRAIN_RATIO)} train items!"
    assert min_test_per_user >= 1, "A test user has 0 test items!"
    print(f"  ✓ [Pass] Min train interactions/user = {min_train_per_user}, min test interactions/user = {min_test_per_user}")

    # Check 4: Chronological ordering per user
    train_max_ts = train_df.groupby("user_id")["timestamp"].max().rename("max_train_ts")
    test_min_ts = test_df.groupby("user_id")["timestamp"].min().rename("min_test_ts")
    merged_ts = pd.concat([train_max_ts, test_min_ts], axis=1).dropna()

    time_violations = (merged_ts["min_test_ts"] < merged_ts["max_train_ts"]).sum()
    assert time_violations == 0, f"Detected {time_violations} temporal leakage violations!"
    print(f"  ✓ [Pass] Chronological ordering validated: 0 temporal leakage violations across {len(merged_ts):,} users")

    # Check 5: Cold-start / unseen item measurement
    train_items = set(train_df["parent_asin"].unique())
    test_items = set(test_df["parent_asin"].unique())

    test_item_in_train = test_df["parent_asin"].isin(train_items)
    n_warm_interactions = test_item_in_train.sum()
    n_cold_interactions = (~test_item_in_train).sum()
    pct_cold_interactions = (n_cold_interactions / len(test_df)) * 100.0

    cold_items_set = test_items - train_items
    print(f"\n❄️ Cold-Start Item Analysis in Test Set:")
    print(f"  - Test interactions with catalog-trained items: {n_warm_interactions:,} ({100-pct_cold_interactions:.2f}%)")
    print(f"  - Test interactions with UNSEEN items:           {n_cold_interactions:,} ({pct_cold_interactions:.2f}%)")
    print(f"  - Number of distinct unseen items in test:       {len(cold_items_set):,} / {len(test_items):,}")

    return {
        "all_passed": True,
        "n_warm_interactions": int(n_warm_interactions),
        "n_cold_interactions": int(n_cold_interactions),
        "pct_cold_interactions": float(pct_cold_interactions),
        "n_cold_items": len(cold_items_set),
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def generate_split_report(
    raw_df: pd.DataFrame,
    filtered_df: pd.DataFrame,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    cold_stats: Dict[str, Any],
):
    """Prints the comprehensive dataset split and profile report."""
    print("\n" + "=" * 70)
    print("📊 SHOPSENSE MODELING DATASET & SPLIT REPORT")
    print("=" * 70)

    # 1. Before Filtering
    n_raw_int = len(raw_df)
    n_raw_u = raw_df["user_id"].nunique()
    n_raw_i = raw_df["parent_asin"].nunique()

    print("\n1. BEFORE FILTERING (Full Cleaned Dataset):")
    print(f"   Interactions:           {n_raw_int:>12,}")
    print(f"   Unique Users:           {n_raw_u:>12,}")
    print(f"   Unique Products:        {n_raw_i:>12,}")
    print(f"   Sparsity:               {100.0 - (n_raw_int / (n_raw_u * n_raw_i)) * 100.0:>11.6f}%")

    # 2. After Filtering
    n_filt_int = len(filtered_df)
    n_filt_u = filtered_df["user_id"].nunique()
    n_filt_i = filtered_df["parent_asin"].nunique()
    sparsity_filt = (1.0 - (n_filt_int / (n_filt_u * n_filt_i))) * 100.0

    print(f"\n2. AFTER FILTERING (Configuration D: Users >= {MIN_USER_INTERACTIONS}, Items >= {MIN_ITEM_INTERACTIONS}):")
    print(f"   Interactions:           {n_filt_int:>12,}  ({n_filt_int/n_raw_int*100:.2f}% of original)")
    print(f"   Unique Users:           {n_filt_u:>12,}  ({n_filt_u/n_raw_u*100:.2f}% of original)")
    print(f"   Unique Products:        {n_filt_i:>12,}  ({n_filt_i/n_raw_i*100:.2f}% of original)")
    print(f"   User-Item Sparsity:     {sparsity_filt:>11.4f}%")
    print(f"   Avg Interactions / User:{n_filt_int / n_filt_u:>12.2f}")
    print(f"   Avg Interactions / Item:{n_filt_int / n_filt_i:>12.2f}")

    # 3. Train Set
    n_train_int = len(train_df)
    n_train_u = train_df["user_id"].nunique()
    n_train_i = train_df["parent_asin"].nunique()
    train_ratings = train_df["rating"]

    print("\n3. TRAIN SET (Oldest 80% per user):")
    print(f"   Interactions:           {n_train_int:>12,}  ({n_train_int/n_filt_int*100:.2f}% of filtered)")
    print(f"   Unique Users:           {n_train_u:>12,}  (100.00% user coverage)")
    print(f"   Unique Products:        {n_train_i:>12,}  ({n_train_i/n_filt_i*100:.2f}% catalog coverage)")
    print(f"   Avg Interactions / User:{n_train_int / n_train_u:>12.2f}")
    print(f"   Rating Mean ± Std:      {train_ratings.mean():.2f} ± {train_ratings.std():.2f}")
    print(f"   Rating Distribution:")
    for r, cnt in train_ratings.value_counts().sort_index().items():
        pct = cnt / n_train_int * 100
        print(f"     {r:.0f} ★ : {cnt:>8,} ({pct:5.1f}%)  {'█' * int(pct/2)}")

    # 4. Test Set
    n_test_int = len(test_df)
    n_test_u = test_df["user_id"].nunique()
    n_test_i = test_df["parent_asin"].nunique()
    test_ratings = test_df["rating"]

    print("\n4. TEST SET (Newest 20% per user):")
    print(f"   Interactions:           {n_test_int:>12,}  ({n_test_int/n_filt_int*100:.2f}% of filtered)")
    print(f"   Unique Users:           {n_test_u:>12,}  (100.00% user coverage)")
    print(f"   Unique Products:        {n_test_i:>12,}")
    print(f"   Avg Interactions / User:{n_test_int / n_test_u:>12.2f}")
    print(f"   Rating Mean ± Std:      {test_ratings.mean():.2f} ± {test_ratings.std():.2f}")
    print(f"   Rating Distribution:")
    for r, cnt in test_ratings.value_counts().sort_index().items():
        pct = cnt / n_test_int * 100
        print(f"     {r:.0f} ★ : {cnt:>8,} ({pct:5.1f}%)  {'█' * int(pct/2)}")

    # 5. Cold-Start Test Items
    print("\n5. COLD-START / UNSEEN TEST ITEMS:")
    print(f"   Test Interactions with Warm Items: {cold_stats['n_warm_interactions']:>10,} ({100-cold_stats['pct_cold_interactions']:5.2f}%)")
    print(f"   Test Interactions with Cold Items: {cold_stats['n_cold_interactions']:>10,} ({cold_stats['pct_cold_interactions']:5.2f}%)")
    print(f"   Unseen Products in Test:           {cold_stats['n_cold_items']:>10,} / {n_test_i:,}")

    print("\n" + "=" * 70)


# ---------------------------------------------------------------------------
# Main Pipeline
# ---------------------------------------------------------------------------

def run_preprocessing_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Main execution function for Step 5."""
    print("🚀 ShopSense Step 5: Dataset Preprocessing & Chronological Split Pipeline")
    print(f"   Input file:  {INPUT_REVIEWS_PARQUET}")

    # Load cleaned reviews
    if not INPUT_REVIEWS_PARQUET.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_REVIEWS_PARQUET}. Run data_loader first.")

    raw_df = pd.read_parquet(INPUT_REVIEWS_PARQUET)

    # 1. Apply Iterative k-Core (Config D)
    filtered_df, iters = apply_iterative_k_core(
        raw_df,
        min_user_k=MIN_USER_INTERACTIONS,
        min_item_k=MIN_ITEM_INTERACTIONS,
    )

    # 2. Save Filtered Dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    filtered_df.to_parquet(OUTPUT_FILTERED_PARQUET, index=False)
    print(f"  ✓ Saved: {OUTPUT_FILTERED_PARQUET.name} ({OUTPUT_FILTERED_PARQUET.stat().st_size / (1024*1024):.1f} MB)")

    # 3. Chronological Split
    train_df, test_df = chronological_per_user_split(filtered_df, train_ratio=TRAIN_RATIO)

    # 4. Verification & Assertions
    cold_stats = verify_split_integrity(filtered_df, train_df, test_df)

    # 5. Save Train & Test Datasets
    train_df.to_parquet(OUTPUT_TRAIN_PARQUET, index=False)
    test_df.to_parquet(OUTPUT_TEST_PARQUET, index=False)
    print(f"  ✓ Saved: {OUTPUT_TRAIN_PARQUET.name} ({OUTPUT_TRAIN_PARQUET.stat().st_size / (1024*1024):.1f} MB)")
    print(f"  ✓ Saved: {OUTPUT_TEST_PARQUET.name} ({OUTPUT_TEST_PARQUET.stat().st_size / (1024*1024):.1f} MB)")

    # 6. Save Development Samples
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    train_sample = train_df.sample(n=min(5000, len(train_df)), random_state=42)
    test_sample = test_df.sample(n=min(2000, len(test_df)), random_state=42)
    cols_to_save = [c for c in train_sample.columns if c != "timestamp_dt"]
    train_sample[cols_to_save].to_csv(SAMPLES_DIR / "train_sample.csv", index=False)
    test_sample[cols_to_save].to_csv(SAMPLES_DIR / "test_sample.csv", index=False)
    print(f"  ✓ Saved: data/samples/train_sample.csv & test_sample.csv")

    # 7. Generate Full Report
    generate_split_report(raw_df, filtered_df, train_df, test_df, cold_stats)

    return filtered_df, train_df, test_df


if __name__ == "__main__":
    run_preprocessing_pipeline()

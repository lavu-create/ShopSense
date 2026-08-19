"""
ShopSense Data Loader
=====================
Loads, cleans, and profiles the Amazon Reviews 2023 — Musical Instruments
dataset from HuggingFace.

Dataset:
    McAuley-Lab/Amazon-Reviews-2023 (Musical_Instruments category)
    https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Usage:
    python -m src.data_loader
"""

import os
import json
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HuggingFace URLs for the Musical Instruments category
REVIEWS_JSONL_URL = (
    "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
    "/resolve/main/raw/review_categories/Musical_Instruments.jsonl"
)

METADATA_PARQUET_URLS = [
    (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        "/resolve/main/raw_meta_Musical_Instruments/full-00000-of-00002.parquet"
    ),
    (
        "https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023"
        "/resolve/main/raw_meta_Musical_Instruments/full-00001-of-00002.parquet"
    ),
]

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SAMPLES_DIR = DATA_DIR / "samples"

# Review fields to keep for the recommendation system
REVIEW_KEEP_COLS = ["user_id", "parent_asin", "rating", "timestamp"]

# Metadata fields to keep
META_KEEP_COLS = [
    "parent_asin", "title", "main_category", "categories",
    "price", "store", "average_rating", "rating_number",
]

# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _download_file(url: str, dest: Path, desc: str = "") -> Path:
    """Download a file if it doesn't already exist locally."""
    if dest.exists() and dest.stat().st_size > 1024 * 1024:
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  ✓ Already cached: {dest.name} ({size_mb:.1f} MB)")
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    print(f"  ↓ Downloading {desc or dest.name}...")
    start = time.time()

    # Stream download with progress
    req = urllib.request.Request(url)
    with urllib.request.urlopen(req) as response:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        chunk_size = 8 * 1024 * 1024  # 8 MB chunks

        with open(dest, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                if total > 0:
                    pct = downloaded / total * 100
                    mb = downloaded / (1024 * 1024)
                    print(f"\r    {mb:.0f} MB / {total / 1024 / 1024:.0f} MB "
                          f"({pct:.1f}%)", end="", flush=True)

    elapsed = time.time() - start
    size_mb = dest.stat().st_size / (1024 * 1024)
    print(f"\n  ✓ Downloaded: {dest.name} ({size_mb:.1f} MB, {elapsed:.1f}s)")
    return dest


# ---------------------------------------------------------------------------
# Memory-Efficient Loading and Cleaning Functions
# ---------------------------------------------------------------------------


def load_and_clean_reviews_streamed(jsonl_path: Path, chunk_size: int = 500_000) -> pd.DataFrame:
    """
    Stream and parse reviews from JSONL, keeping only the 4 required fields
    and cleaning on-the-fly to minimize RAM usage.
    """
    print("\n📦 Loading and cleaning reviews (streaming mode)...")
    start_time = time.time()

    chunks: List[pd.DataFrame] = []
    current_batch = {
        "user_id": [],
        "parent_asin": [],
        "rating": [],
        "timestamp": [],
    }

    total_lines = 0
    missing_uid = 0
    missing_asin = 0
    missing_rating = 0
    invalid_rating = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            total_lines += 1
            obj = json.loads(line)

            uid = obj.get("user_id")
            asin = obj.get("parent_asin")
            rating = obj.get("rating")
            ts = obj.get("timestamp")

            if not uid:
                missing_uid += 1
                continue
            if not asin:
                missing_asin += 1
                continue
            if rating is None:
                missing_rating += 1
                continue

            try:
                rating = float(rating)
            except (ValueError, TypeError):
                invalid_rating += 1
                continue

            if not (1.0 <= rating <= 5.0):
                invalid_rating += 1
                continue

            current_batch["user_id"].append(uid)
            current_batch["parent_asin"].append(asin)
            current_batch["rating"].append(rating)
            current_batch["timestamp"].append(int(ts) if ts is not None else 0)

            if len(current_batch["user_id"]) >= chunk_size:
                chunk_df = pd.DataFrame(current_batch)
                chunks.append(chunk_df)
                current_batch = {"user_id": [], "parent_asin": [], "rating": [], "timestamp": []}
                print(f"    Parsed {total_lines:,} raw lines...", end="\r", flush=True)

    if current_batch["user_id"]:
        chunks.append(pd.DataFrame(current_batch))

    print(f"\n  ✓ Finished reading {total_lines:,} lines in {time.time() - start_time:.1f}s")
    print(f"  Missing user_id: {missing_uid:,}")
    print(f"  Missing parent_asin: {missing_asin:,}")
    print(f"  Missing rating: {missing_rating:,}")
    print(f"  Invalid rating (not 1–5): {invalid_rating:,}")

    df = pd.concat(chunks, ignore_index=True)
    n_before_dedup = len(df)

    # Remove exact duplicate interactions (same user + product + rating + timestamp)
    df = df.drop_duplicates(subset=["user_id", "parent_asin", "rating", "timestamp"])
    n_dupes = n_before_dedup - len(df)
    print(f"  Exact duplicates removed: {n_dupes:,}")

    # Convert timestamp (milliseconds) to datetime
    df["timestamp_dt"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
    df = df.reset_index(drop=True)

    print(f"  Final clean reviews: {len(df):,} interactions")
    return df


def load_metadata() -> pd.DataFrame:
    """
    Download (if needed) and load product metadata from Parquet files.
    """
    print("\n📦 Loading product metadata...")
    parquet_paths = []
    for i, url in enumerate(METADATA_PARQUET_URLS):
        dest = RAW_DIR / f"meta_musical_instruments_{i}.parquet"
        _download_file(url, dest, desc=f"metadata part {i + 1}")
        parquet_paths.append(dest)

    dfs = [pd.read_parquet(p, columns=META_KEEP_COLS) for p in parquet_paths]
    meta = pd.concat(dfs, ignore_index=True)
    print(f"  Raw metadata loaded: {len(meta):,} products")
    return meta


def clean_metadata(meta: pd.DataFrame) -> pd.DataFrame:
    """
    Basic metadata cleaning:
    - Deduplicate by parent_asin (keep first)
    - Clean price field
    - Flatten categories
    """
    print("\n🧹 Cleaning metadata...")
    n_start = len(meta)

    # Deduplicate by parent_asin
    meta = meta.drop_duplicates(subset=["parent_asin"], keep="first")
    print(f"  Deduplicated: {n_start:,} → {len(meta):,}")

    meta = meta.copy()
    # Clean price: parse float
    if "price" in meta.columns:
        def _parse_price(val):
            if val is None:
                return np.nan
            if isinstance(val, (int, float)):
                return float(val) if not np.isnan(val) else np.nan
            s = str(val).replace("$", "").replace(",", "").strip()
            try:
                return float(s)
            except (ValueError, TypeError):
                return np.nan

        meta["price"] = meta["price"].apply(_parse_price)
        valid_prices = meta["price"].notna().sum()
        print(f"  Products with valid price: {valid_prices:,} / {len(meta):,}")

    # Flatten categories list to pipe-separated string
    if "categories" in meta.columns:
        def _format_cats(val):
            if val is None:
                return ""
            if isinstance(val, (list, np.ndarray)):
                return " | ".join(str(item) for item in val if item is not None)
            return str(val)

        meta["categories_flat"] = meta["categories"].apply(_format_cats)

    return meta.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------


def profile_dataset(reviews: pd.DataFrame, meta: Optional[pd.DataFrame] = None):
    """Print a concise dataset profile."""
    print("\n" + "=" * 60)
    print("📊 DATASET PROFILE")
    print("=" * 60)

    n_interactions = len(reviews)
    n_users = reviews["user_id"].nunique()
    n_products = reviews["parent_asin"].nunique()

    print(f"\n  Interactions:     {n_interactions:,}")
    print(f"  Unique users:    {n_users:,}")
    print(f"  Unique products: {n_products:,}")
    print(f"  Sparsity:        {1.0 - (n_interactions / (n_users * n_products)):.6f}")
    print(f"  Avg ratings/user:{n_interactions / n_users:.2f}")
    print(f"  Avg ratings/item:{n_interactions / n_products:.2f}")

    # Rating distribution
    print("\n  Rating distribution:")
    rating_counts = reviews["rating"].value_counts().sort_index()
    for rating, count in rating_counts.items():
        pct = count / n_interactions * 100
        bar = "█" * int(pct / 2)
        print(f"    {rating:.0f} ★  {count:>8,}  ({pct:5.1f}%)  {bar}")

    # Date range
    if "timestamp_dt" in reviews.columns:
        valid_dates = reviews["timestamp_dt"].dropna()
        if len(valid_dates) > 0:
            print(f"\n  Date range:      {valid_dates.min().date()} → {valid_dates.max().date()}")

    # Missing values
    print("\n  Missing values (reviews):")
    for col in reviews.columns:
        n_miss = reviews[col].isna().sum()
        if n_miss > 0:
            print(f"    {col}: {n_miss:,}")
    if reviews.isna().sum().sum() == 0:
        print("    (none)")

    # Metadata summary
    if meta is not None:
        n_meta = len(meta)
        matched = reviews["parent_asin"].isin(meta["parent_asin"]).sum()
        print(f"\n  Metadata:        {n_meta:,} unique products loaded")
        print(f"  Reviews matched: {matched:,} / {n_interactions:,} "
              f"({matched / n_interactions * 100:.1f}%)")

        if "price" in meta.columns:
            valid_p = meta["price"].dropna()
            if len(valid_p) > 0:
                print(f"  Price range:     ${valid_p.min():.2f} – ${valid_p.max():.2f} "
                      f"(median ${valid_p.median():.2f})")

        if "title" in meta.columns:
            has_title = meta["title"].notna().sum()
            print(f"  Has title:       {has_title:,} / {n_meta:,}")

    print("\n" + "=" * 60)


# ---------------------------------------------------------------------------
# Save functions
# ---------------------------------------------------------------------------


def save_samples(
    reviews: pd.DataFrame,
    meta: pd.DataFrame,
    n_review_sample: int = 5000,
    n_meta_sample: int = 1000,
):
    """Save small sample files for development/testing."""
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    # Sample reviews
    n_rev = min(n_review_sample, len(reviews))
    review_sample = reviews.sample(n=n_rev, random_state=42)
    review_path = SAMPLES_DIR / "reviews_sample.csv"
    cols_to_save = [c for c in review_sample.columns if c != "timestamp_dt"]
    review_sample[cols_to_save].to_csv(review_path, index=False)
    print(f"  ✓ Saved review sample: {review_path.name} ({n_rev:,} rows)")

    # Sample metadata — prioritize products present in review sample
    sampled_asins = set(review_sample["parent_asin"].unique())
    meta_matched = meta[meta["parent_asin"].isin(sampled_asins)]
    meta_extra = meta[~meta["parent_asin"].isin(sampled_asins)]
    n_extra = max(0, n_meta_sample - len(meta_matched))
    if n_extra > 0 and len(meta_extra) > 0:
        meta_combined = pd.concat([
            meta_matched,
            meta_extra.sample(n=min(n_extra, len(meta_extra)), random_state=42)
        ], ignore_index=True)
    else:
        meta_combined = meta_matched.head(n_meta_sample)

    meta_path = SAMPLES_DIR / "metadata_sample.csv"
    cols_to_drop = [c for c in ["categories"] if c in meta_combined.columns]
    meta_combined.drop(columns=cols_to_drop, errors="ignore").to_csv(
        meta_path, index=False
    )
    print(f"  ✓ Saved metadata sample: {meta_path.name} ({len(meta_combined):,} rows)")


def save_processed(reviews: pd.DataFrame, meta: pd.DataFrame):
    """Save full cleaned datasets for downstream use."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    rev_path = PROCESSED_DIR / "reviews_clean.parquet"
    cols = [c for c in reviews.columns if c != "timestamp_dt"]
    reviews[cols].to_parquet(rev_path, index=False)
    size_mb = rev_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {rev_path.name} ({size_mb:.1f} MB)")

    meta_path = PROCESSED_DIR / "metadata_clean.parquet"
    meta_cols = [c for c in meta.columns if c != "categories"]
    meta[meta_cols].to_parquet(meta_path, index=False)
    size_mb = meta_path.stat().st_size / (1024 * 1024)
    print(f"  ✓ Saved: {meta_path.name} ({size_mb:.1f} MB)")


def report_disk_usage():
    """Report total disk space used by downloaded/processed data."""
    print("\n💾 Disk usage:")
    total = 0
    for dirpath in [RAW_DIR, PROCESSED_DIR, SAMPLES_DIR]:
        if dirpath.exists():
            dir_size = sum(f.stat().st_size for f in dirpath.rglob("*") if f.is_file())
            total += dir_size
            print(f"  {dirpath.relative_to(PROJECT_ROOT)}: "
                  f"{dir_size / (1024 * 1024):.1f} MB")
    print(f"  Total: {total / (1024 * 1024):.1f} MB")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_pipeline() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Execute the full data loading pipeline:
    1. Verify raw files exist or download
    2. Load & clean metadata
    3. Load & clean reviews (streamed)
    4. Profile
    5. Save processed + samples
    """
    print("🚀 ShopSense Data Pipeline")
    print(f"   Dataset: Amazon Reviews 2023 — Musical Instruments")
    print(f"   Source:  McAuley-Lab/Amazon-Reviews-2023 (HuggingFace)")
    print(f"   Project: {PROJECT_ROOT}")

    # --- Step 1: Load metadata ---
    print("\n" + "-" * 60)
    print("STEP 1: Load metadata")
    print("-" * 60)
    meta = load_metadata()
    meta = clean_metadata(meta)

    # --- Step 2: Check / Download reviews JSONL ---
    print("\n" + "-" * 60)
    print("STEP 2: Reviews dataset")
    print("-" * 60)
    reviews_path = RAW_DIR / "Musical_Instruments_reviews.jsonl"
    if not reviews_path.exists() or reviews_path.stat().st_size < 100 * 1024 * 1024:
        _download_file(REVIEWS_JSONL_URL, reviews_path, desc="reviews JSONL (~1.5 GB)")

    # --- Step 3: Stream, Parse & Clean reviews ---
    print("\n" + "-" * 60)
    print("STEP 3: Load, parse & clean reviews")
    print("-" * 60)
    reviews = load_and_clean_reviews_streamed(reviews_path)

    # --- Step 4: Profile ---
    print("\n" + "-" * 60)
    print("STEP 4: Dataset profile")
    print("-" * 60)
    profile_dataset(reviews, meta)

    # --- Step 5: Save ---
    print("\n" + "-" * 60)
    print("STEP 5: Save processed data & samples")
    print("-" * 60)
    save_processed(reviews, meta)
    save_samples(reviews, meta)

    # --- Disk usage ---
    report_disk_usage()

    print("\n✅ Data pipeline complete!")
    return reviews, meta


if __name__ == "__main__":
    run_pipeline()

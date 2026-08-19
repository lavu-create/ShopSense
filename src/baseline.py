"""
ShopSense Popularity Baseline Recommender
=========================================
Implements a robust non-personalized popularity baseline fitted strictly
on training data, with unseen item filtering and full ranking evaluation.

Usage:
    python -m src.baseline
"""

import math
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Set, Tuple, Any

import numpy as np
import pandas as pd

from src.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    catalog_coverage_at_k,
    evaluate_recommendations,
)

# ---------------------------------------------------------------------------
# Configuration & Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

TRAIN_PATH = PROCESSED_DIR / "train.parquet"
TEST_PATH = PROCESSED_DIR / "test.parquet"
METADATA_PATH = PROCESSED_DIR / "metadata_clean.parquet"
REPORT_OUTPUT_PATH = REPORTS_DIR / "baseline_report.md"


# ---------------------------------------------------------------------------
# Popularity Recommender Class
# ---------------------------------------------------------------------------

class PopularityRecommender:
    """
    Popularity-based recommendation baseline.
    Ranks products by historical interaction volume and Bayesian-smoothed rating.
    Excludes items already seen in training for each user.
    """

    def __init__(self, m_threshold: int = 5):
        """
        Parameters
        ----------
        m_threshold : int
            Minimum interaction threshold for Bayesian rating smoothing.
        """
        self.m_threshold = m_threshold
        self.is_fitted = False

        # Internal state
        self.user_seen_items: Dict[str, Set[str]] = {}
        self.item_stats_df: pd.DataFrame = pd.DataFrame()
        self.ranked_item_ids: List[str] = []
        self.catalog_items: Set[str] = set()
        self.global_mean_rating: float = 0.0

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        """
        Fits the popularity model on training interactions only.
        """
        print(f"📊 Fitting PopularityRecommender on {len(train_df):,} training interactions...")
        start_time = time.time()

        # 1. Record seen items per user from train
        user_groups = train_df.groupby("user_id")["parent_asin"].apply(set)
        self.user_seen_items = user_groups.to_dict()

        # 2. Compute item statistics
        self.global_mean_rating = float(train_df["rating"].mean())
        C = self.global_mean_rating
        m = self.m_threshold

        stats = train_df.groupby("parent_asin").agg(
            interaction_count=("rating", "count"),
            avg_rating=("rating", "mean"),
        ).reset_index()

        # 3. Compute Bayesian Weighted Rating (WR)
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        v = stats["interaction_count"]
        R = stats["avg_rating"]
        stats["weighted_rating"] = (v / (v + m)) * R + (m / (v + m)) * C

        # 4. Popularity Score: Interaction volume weighted by Bayesian rating
        # Ensures products with hundreds of ratings rank above products with only 5 five-star ratings
        stats["popularity_score"] = stats["interaction_count"] * stats["weighted_rating"]

        # Sort descending by popularity score, breaking ties with interaction_count and avg_rating
        stats = stats.sort_values(
            by=["popularity_score", "interaction_count", "avg_rating"],
            ascending=[False, False, False],
        ).reset_index(drop=True)

        self.item_stats_df = stats
        self.ranked_item_ids = stats["parent_asin"].tolist()
        self.catalog_items = set(self.ranked_item_ids)
        self.is_fitted = True

        elapsed = time.time() - start_time
        print(f"  ✓ Fitted in {elapsed:.2f}s across {len(self.catalog_items):,} unique products and {len(self.user_seen_items):,} users.")
        return self

    def recommend(
        self,
        user_id: Optional[str] = None,
        k: int = 10,
        exclude_seen: bool = True,
    ) -> List[str]:
        """
        Generate top-K recommendations for a given user.
        Excludes items already interacted with in training.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend().")

        seen = self.user_seen_items.get(user_id, set()) if (exclude_seen and user_id) else set()

        recommendations = []
        for item_id in self.ranked_item_ids:
            if item_id not in seen:
                recommendations.append(item_id)
                if len(recommendations) == k:
                    break

        return recommendations

    def recommend_all(
        self,
        user_ids: Sequence[str],
        k: int = 10,
        exclude_seen: bool = True,
    ) -> Dict[str, List[str]]:
        """
        Generates top-K recommendations for a list of users.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before calling recommend_all().")

        results = {}
        for uid in user_ids:
            results[uid] = self.recommend(user_id=uid, k=k, exclude_seen=exclude_seen)
        return results

    def get_top_n_products(self, n: int = 10) -> pd.DataFrame:
        """Returns the top N popular products with full stats."""
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted.")
        return self.item_stats_df.head(n).copy()


# ---------------------------------------------------------------------------
# Evaluation Pipeline
# ---------------------------------------------------------------------------

def run_popularity_evaluation(
    train_path: Path = TRAIN_PATH,
    test_path: Path = TEST_PATH,
    metadata_path: Path = METADATA_PATH,
    k: int = 10,
) -> Tuple[PopularityRecommender, Dict[str, Any]]:
    """
    Fits and evaluates the Popularity baseline on the chronological test split.
    """
    print("🚀 ShopSense Step 6: Popularity Baseline Recommender Evaluation")

    # Load datasets
    train_df = pd.read_parquet(train_path)
    test_df = pd.read_parquet(test_path)

    # 1. Fit Recommender
    recommender = PopularityRecommender(m_threshold=5)
    recommender.fit(train_df)

    # 2. Build Ground Truth Test Sets
    # Full actuals (all test interactions per user)
    all_actuals: Dict[str, List[str]] = (
        test_df.groupby("user_id")["parent_asin"].apply(list).to_dict()
    )

    # Warm actuals (only test interactions whose products exist in the training catalog)
    train_catalog = recommender.catalog_items
    warm_test_df = test_df[test_df["parent_asin"].isin(train_catalog)]
    warm_actuals: Dict[str, List[str]] = (
        warm_test_df.groupby("user_id")["parent_asin"].apply(list).to_dict()
    )

    eval_users = sorted(all_actuals.keys())
    print(f"\n⚡ Generating Top-{k} recommendations for {len(eval_users):,} evaluation users...")
    start_time = time.time()
    recommendations = recommender.recommend_all(eval_users, k=k, exclude_seen=True)
    elapsed = time.time() - start_time
    print(f"  ✓ Generated {len(recommendations):,} recommendation lists in {elapsed:.2f}s ({len(eval_users)/elapsed:,.0f} users/sec)")

    # 3. Compute Ranking Metrics
    print("\n📏 Computing ranking metrics (Precision@K, Recall@K, NDCG@K, Coverage@K)...")
    overall_metrics = evaluate_recommendations(all_actuals, recommendations, k=k)
    warm_metrics = evaluate_recommendations(warm_actuals, recommendations, k=k)

    # 4. Catalog Coverage & List Stats
    coverage = catalog_coverage_at_k(recommendations, train_catalog, k=k)
    unique_recs = len(set(item for recs in recommendations.values() for item in recs))
    rec_lengths = [len(r) for r in recommendations.values()]
    avg_rec_len = float(np.mean(rec_lengths)) if rec_lengths else 0.0

    results = {
        "k": k,
        "precision_at_k": overall_metrics[f"precision@{k}"],
        "recall_at_k": overall_metrics[f"recall@{k}"],
        "ndcg_at_k": overall_metrics[f"ndcg@{k}"],
        "warm_precision_at_k": warm_metrics[f"precision@{k}"],
        "warm_recall_at_k": warm_metrics[f"recall@{k}"],
        "warm_ndcg_at_k": warm_metrics[f"ndcg@{k}"],
        "coverage_at_k": coverage,
        "unique_products_recommended": unique_recs,
        "avg_recommendation_length": avg_rec_len,
        "catalog_size": len(train_catalog),
        "n_eval_users": len(eval_users),
    }

    # 5. Enrich Top 10 Popular Products with Metadata
    top_10 = recommender.get_top_n_products(10)
    if metadata_path.exists():
        meta_df = pd.read_parquet(metadata_path)
        top_10 = top_10.merge(
            meta_df[["parent_asin", "title", "main_category", "price", "store", "categories_flat"]],
            on="parent_asin",
            how="left",
        )

    # 6. Generate Markdown Report
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    _generate_markdown_report(top_10, results, REPORT_OUTPUT_PATH)

    # 7. Print Console Summary
    _print_console_summary(top_10, results)

    return recommender, results


# ---------------------------------------------------------------------------
# Reporting Helpers
# ---------------------------------------------------------------------------

def _print_console_summary(top_10: pd.DataFrame, results: Dict[str, Any]):
    k = results["k"]
    print("\n" + "=" * 70)
    print("🏆 SHOPSENSE POPULARITY BASELINE RESULTS")
    print("=" * 70)

    print(f"\n📈 Overall Evaluation (All {results['n_eval_users']:,} Test Users):")
    print(f"   Precision@{k}:              {results['precision_at_k']:.6f}  ({results['precision_at_k']*100:.3f}%)")
    print(f"   Recall@{k}:                 {results['recall_at_k']:.6f}  ({results['recall_at_k']*100:.3f}%)")
    print(f"   NDCG@{k}:                   {results['ndcg_at_k']:.6f}")

    print(f"\n🔥 Warm-Item Evaluation (Known Training Catalog Items Only):")
    print(f"   Warm Precision@{k}:         {results['warm_precision_at_k']:.6f}  ({results['warm_precision_at_k']*100:.3f}%)")
    print(f"   Warm Recall@{k}:            {results['warm_recall_at_k']:.6f}  ({results['warm_recall_at_k']*100:.3f}%)")
    print(f"   Warm NDCG@{k}:              {results['warm_ndcg_at_k']:.6f}")

    print(f"\n🌐 Catalog Coverage & Diversity:")
    print(f"   Catalog Coverage@{k}:       {results['coverage_at_k']:.2f}% ({results['unique_products_recommended']:,} / {results['catalog_size']:,} products)")
    print(f"   Unique Items Recommended:   {results['unique_products_recommended']:,}")
    print(f"   Avg Rec List Length:        {results['avg_recommendation_length']:.1f}")

    print("\n🔝 Top 5 Most Popular Musical Instruments (Train Set):")
    for idx, row in top_10.head(5).iterrows():
        title = row.get("title", row["parent_asin"])
        if len(str(title)) > 45:
            title = str(title)[:42] + "..."
        price_str = f"${row['price']:.2f}" if pd.notna(row.get("price")) else "N/A"
        print(f"   {idx+1}. [{row['parent_asin']}] {title:<45} | {row['interaction_count']:>4} reviews | {row['avg_rating']:.2f}★ | {price_str}")

    print("=" * 70)


def _generate_markdown_report(top_10: pd.DataFrame, results: Dict[str, Any], output_path: Path):
    k = results["k"]
    top_10_md_rows = []
    for idx, row in top_10.iterrows():
        title = str(row.get("title", row["parent_asin"])).replace("|", "-")
        store = str(row.get("store", "N/A"))
        price = f"${row['price']:.2f}" if pd.notna(row.get("price")) else "N/A"
        top_10_md_rows.append(
            f"| {idx+1} | `{row['parent_asin']}` | {title} | {store} | {price} | {row['interaction_count']:,} | {row['avg_rating']:.2f} ★ | {row['popularity_score']:.1f} |"
        )
    top_10_table = "\n".join(top_10_md_rows)

    report_content = f"""# 📈 ShopSense Baseline Evaluation Report: Popularity Recommender

> Non-personalized popularity baseline fitted strictly on `data/processed/train.parquet` (136,814 interactions).

---

## 1. Baseline Methodology
The Popularity Baseline ranks products across the training catalog using a **Bayesian Weighted Rating & Volume Scoring** formulation:

$$\\text{{WR}} = \\left(\\frac{{v}}{{v + m}}\\right) R + \\left(\\frac{{m}}{{v + m}}\\right) C$$

$$\\text{{Score}} = v \\times \\text{{WR}}$$

Where:
- $v$ = interaction count of the item in the training set
- $m = 5$ = smoothing threshold parameter
- $R$ = sample mean rating of the item
- $C = 4.48$ = global average rating across the entire training set

For any evaluation user $u$, `recommend(user_id=u, k=10)` generates the top-$k$ products from the sorted popularity list, **strictly filtering out any products already rated by $u$ in the training set**.

---

## 2. Top 10 Most Popular Products (Training Set)

| Rank | ASIN | Product Title | Brand / Store | Price | Interactions ($v$) | Avg Rating ($R$) | Score |
|:---:|:---:|:---|:---:|:---:|:---:|:---:|:---:|
{top_10_table}

---

## 3. Evaluation Results on Chronological Test Set (K = {k})

| Metric | Overall Test Set (All Items) | Warm-Item Test Set (Known Catalog Items) |
|:---|:---:|:---:|
| **Precision@{k}** | **{results['precision_at_k']:.6f}** ({results['precision_at_k']*100:.3f}%) | **{results['warm_precision_at_k']:.6f}** ({results['warm_precision_at_k']*100:.3f}%) |
| **Recall@{k}** | **{results['recall_at_k']:.6f}** ({results['recall_at_k']*100:.3f}%) | **{results['warm_recall_at_k']:.6f}** ({results['warm_recall_at_k']*100:.3f}%) |
| **NDCG@{k}** | **{results['ndcg_at_k']:.6f}** | **{results['warm_ndcg_at_k']:.6f}** |

---

## 4. Catalog Coverage & Recommendation Diversity

| Metric | Value |
|:---|:---:|
| **Catalog Coverage@{k}** | **{results['coverage_at_k']:.2f}%** |
| **Unique Products Recommended** | **{results['unique_products_recommended']:,} / {results['catalog_size']:,}** |
| **Average Recommendation List Length** | **{results['avg_recommendation_length']:.1f}** |
| **Total Evaluation Users** | **{results['n_eval_users']:,}** |

---

## 5. Key Analytical Takeaways
1. **Low Catalog Coverage is Expected:** As a global popularity baseline without personalization, the model recommends the same top items (shifted only by user seen-item exclusions). Only **{results['unique_products_recommended']}** distinct items appear in the top-10 across all 10,678 users, representing **{results['coverage_at_k']:.2f}%** catalog coverage.
2. **Benchmark for Collaborative Filtering:** The baseline establishes our non-personalized benchmark:
   - **Recall@10 = {results['recall_at_k']*100:.3f}%**
   - **NDCG@10 = {results['ndcg_at_k']:.4f}**
   - Personalized collaborative filtering models (User-Based, Item-Based, and Hybrid) in subsequent steps must demonstrate measurable improvements over this benchmark.
3. **Cold-Start / Unseen Item Impact:** Unseen test items represent 0.57% of test interactions, leading to a slight difference between overall and warm-item Recall/NDCG.
"""
    output_path.write_text(report_content, encoding="utf-8")
    print(f"\n📝 Saved baseline report to {output_path}")


if __name__ == "__main__":
    run_popularity_evaluation()

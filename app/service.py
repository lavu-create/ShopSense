"""
ShopSense Recommender Service Layer
===================================
Manages singleton recommender initialization, thread-safe inference routing,
and metadata/score enrichment for API responses.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd
import scipy.sparse as sp
from fastapi import HTTPException, status

from src.recommender import ShopSenseRecommender
from app.schemas import ItemRecommendation

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
TRAIN_PATH = DATA_DIR / "train.parquet"
METADATA_PATH = DATA_DIR / "metadata_clean.parquet"


class RecommenderService:
    """
    Singleton service wrapper around ShopSenseRecommender.
    Loads and fits the multi-tier recommender once at application startup.
    """

    def __init__(self):
        self.recommender: Optional[ShopSenseRecommender] = None
        self.metadata_map: Dict[str, Dict[str, Any]] = {}
        self.is_initialized: bool = False

    def initialize(
        self,
        train_path: Path = TRAIN_PATH,
        metadata_path: Path = METADATA_PATH,
        min_cf_interactions: int = 10,
    ) -> None:
        """
        Loads training interactions and clean metadata, then fits the unified system.
        """
        if self.is_initialized:
            return

        if not train_path.exists():
            raise FileNotFoundError(f"Training data not found at: {train_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found at: {metadata_path}")

        train_df = pd.read_parquet(train_path)
        meta_df = pd.read_parquet(metadata_path)

        rec = ShopSenseRecommender(min_cf_interactions=min_cf_interactions)
        rec.fit(train_df=train_df, metadata_df=meta_df)

        self.recommender = rec
        self.metadata_map = rec.content_model.item_metadata_map
        self.is_initialized = True

    def _enrich_item(
        self,
        asin: str,
        rank: int,
        score: Optional[float] = None,
    ) -> ItemRecommendation:
        """Enriches an ASIN with catalog metadata fields and rounded numeric score."""
        meta = self.metadata_map.get(asin, {})
        title = meta.get("title") or asin
        category = meta.get("main_category") or meta.get("categories_flat") or "Musical Instruments"
        
        raw_price = meta.get("price")
        price = float(raw_price) if (raw_price is not None and pd.notna(raw_price)) else None
        
        clean_score = float(round(score, 4)) if (score is not None and pd.notna(score)) else None

        return ItemRecommendation(
            rank=rank,
            parent_asin=asin,
            title=title,
            category=category,
            price=price,
            score=clean_score,
        )

    def _compute_scores_for_recommendations(
        self,
        user_id: str,
        strategy: str,
        asin_list: List[str],
    ) -> List[Optional[float]]:
        """
        Extracts genuine model prediction/similarity scores computed by the underlying engine.
        Does not invent scores; queries the fitted model representations directly.
        """
        if not asin_list or self.recommender is None:
            return [None] * len(asin_list)

        try:
            if strategy == "content_based_sparse_history":
                # Content-based user profile cosine similarity: Score(u, i) = u_profile . x_i^T
                hist = self.recommender.user_train_history.get(user_id, [])
                ratings = self.recommender.user_train_ratings.get(user_id, [])
                if not hist:
                    return [None] * len(asin_list)

                content_model = self.recommender.content_model
                valid_indices = [content_model.item_to_idx[item] for item in hist if item in content_model.item_to_idx]
                if not valid_indices:
                    return [None] * len(asin_list)

                weights = np.array(
                    [ratings[i] for i, item in enumerate(hist) if item in content_model.item_to_idx],
                    dtype=np.float32,
                )
                total_w = np.sum(weights)
                if total_w > 0:
                    weights = weights / total_w

                profile_vec = sp.csr_matrix(weights) @ content_model.X_tfidf[valid_indices]
                profile_norm = sp.linalg.norm(profile_vec)
                if profile_norm > 0:
                    profile_vec = profile_vec / profile_norm

                scores_arr = (content_model.X_tfidf @ profile_vec.T).toarray().ravel()
                return [
                    float(scores_arr[content_model.item_to_idx[asin]])
                    if asin in content_model.item_to_idx
                    else None
                    for asin in asin_list
                ]

            elif strategy == "hybrid_cf":
                # Weighted hybrid score: Score_hybrid(u, i) = w_pop*s_pop + w_user*s_user + w_item*s_item
                hybrid_model = self.recommender.hybrid_model
                if user_id not in hybrid_model.user_model.matrix.user_to_idx:
                    return [None] * len(asin_list)

                u_idx = hybrid_model.user_model.matrix.user_to_idx[user_id]
                u_norm = hybrid_model.user_model.matrix.R_norm[u_idx]
                sims_u = (u_norm @ hybrid_model.user_model.matrix.R_norm.T).toarray().ravel()
                sims_u[u_idx] = 0.0

                pos_u_mask = sims_u > 0.0
                user_scores_raw: Dict[str, float] = {}
                if np.any(pos_u_mask):
                    pos_idx = np.where(pos_u_mask)[0]
                    pos_s = sims_u[pos_idx]
                    if len(pos_idx) > hybrid_model.user_model.n_neighbors:
                        top_loc = np.argpartition(pos_s, -hybrid_model.user_model.n_neighbors)[-hybrid_model.user_model.n_neighbors:]
                        top_neighbors = pos_idx[top_loc]
                        top_s = pos_s[top_loc]
                    else:
                        top_neighbors = pos_idx
                        top_s = pos_s
                    sub_R = hybrid_model.user_model.matrix.R[top_neighbors]
                    u_cand_scores = (sub_R.T @ top_s).ravel()
                    c_idx = np.where(u_cand_scores > 0.0)[0]
                    user_scores_raw = {hybrid_model.user_model.matrix.idx_to_item[c]: float(u_cand_scores[c]) for c in c_idx}

                u_vec = hybrid_model.item_model.matrix.R[u_idx]
                item_scores_arr = (u_vec @ hybrid_model.item_model.S_sparse.T).toarray().ravel()
                pos_i_idx = np.where(item_scores_arr > 0.0)[0]
                item_scores_raw: Dict[str, float] = {
                    hybrid_model.item_model.matrix.idx_to_item[c]: float(item_scores_arr[c])
                    for c in pos_i_idx
                }

                user_norm = hybrid_model._normalize_scores(user_scores_raw)
                item_norm = hybrid_model._normalize_scores(item_scores_raw)

                scores_list = []
                for asin in asin_list:
                    s_pop = hybrid_model.pop_scores_norm.get(asin, 0.0)
                    s_user = user_norm.get(asin, 0.0)
                    s_item = item_norm.get(asin, 0.0)
                    score = hybrid_model.w_pop * s_pop + hybrid_model.w_user * s_user + hybrid_model.w_item * s_item
                    scores_list.append(float(score))
                return scores_list

            elif strategy == "cold_user_popularity_prior":
                # Normalized Bayesian popularity score in [0, 1]
                pop_scores = self.recommender.hybrid_model.pop_scores_norm
                return [
                    float(pop_scores.get(asin))
                    if asin in pop_scores
                    else None
                    for asin in asin_list
                ]

        except Exception:
            return [None] * len(asin_list)

        return [None] * len(asin_list)

    def recommend(
        self,
        user_id: str,
        k: int = 10,
    ) -> Tuple[str, List[ItemRecommendation]]:
        """
        Generates top-K recommendations for a user via multi-tier routing,
        enriched with product metadata and genuine model scores.
        """
        if not self.is_initialized or self.recommender is None:
            raise RuntimeError("RecommenderService is not initialized.")

        asin_list, strategy = self.recommender.recommend(user_id=user_id, k=k)
        scores_list = self._compute_scores_for_recommendations(user_id=user_id, strategy=strategy, asin_list=asin_list)

        recommendations = [
            self._enrich_item(
                asin=asin,
                rank=idx + 1,
                score=scores_list[idx] if idx < len(scores_list) else None,
            )
            for idx, asin in enumerate(asin_list)
        ]
        return strategy, recommendations

    def get_similar_items(
        self,
        parent_asin: str,
        k: int = 10,
    ) -> Tuple[str, List[ItemRecommendation]]:
        """
        Retrieves similar items for target ASIN.
        Raises 404 if item does not exist in catalog metadata.
        """
        if not self.is_initialized or self.recommender is None:
            raise RuntimeError("RecommenderService is not initialized.")

        # Verify whether item exists in CF or metadata catalog
        is_in_cf = (
            self.recommender.item_cf_model is not None and
            self.recommender.item_cf_model.is_fitted and
            self.recommender.item_cf_model.matrix is not None and
            parent_asin in self.recommender.item_cf_model.matrix.item_to_idx
        )
        is_in_meta = parent_asin in self.metadata_map

        if not is_in_cf and not is_in_meta:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Product '{parent_asin}' not found in catalog metadata.",
            )

        sim_tuples, strategy = self.recommender.get_similar_items(item_id=parent_asin, k=k)

        # Handle edge case where item has 0 similarity matches: fallback to popularity
        if len(sim_tuples) == 0:
            strategy = "popularity_fallback"
            pop_items = self.recommender.popularity_model.recommend(user_id=None, k=k)
            similar_items = [
                self._enrich_item(asin=asin, rank=idx + 1)
                for idx, asin in enumerate(pop_items)
            ]
        else:
            similar_items = [
                self._enrich_item(asin=asin, rank=idx + 1, score=score)
                for idx, (asin, score) in enumerate(sim_tuples[:k])
            ]

        return strategy, similar_items


# Global singleton instance
recommender_service = RecommenderService()


def get_service() -> RecommenderService:
    """Dependency injector for RecommenderService."""
    return recommender_service

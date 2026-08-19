"""
ShopSense API Pydantic Request & Response Schemas
=================================================
Defines strict data contracts and validation models for all API endpoints.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator, ConfigDict


class HealthResponse(BaseModel):
    """Health check endpoint response schema."""
    status: str = Field(default="ok", description="Service health status")
    service: str = Field(default="ShopSense Recommendation API", description="Service name")


class ItemRecommendation(BaseModel):
    """Enriched recommendation record."""
    model_config = ConfigDict(extra="ignore")

    rank: int = Field(..., ge=1, description="1-indexed recommendation position")
    parent_asin: str = Field(..., description="Unique product ASIN identifier")
    title: Optional[str] = Field(default=None, description="Product headline title")
    category: Optional[str] = Field(default=None, description="Primary product department/taxonomy")
    price: Optional[float] = Field(default=None, description="Product list price in USD")
    score: Optional[float] = Field(default=None, description="Model prediction or similarity score")


class RecommendRequest(BaseModel):
    """Request schema for POST /recommend."""
    user_id: str = Field(..., description="Unique user identifier", min_length=1)
    k: int = Field(default=10, ge=1, le=20, description="Number of recommendations to return (1-20)")

    @field_validator("user_id")
    @classmethod
    def validate_user_id(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("user_id must be a non-empty string")
        return v.strip()


class RecommendResponse(BaseModel):
    """Response schema for POST /recommend."""
    user_id: str = Field(..., description="Target user identifier")
    strategy: str = Field(
        ...,
        description="Routing strategy used (hybrid_cf, content_based_sparse_history, cold_user_popularity_prior)",
    )
    recommendations: List[ItemRecommendation] = Field(
        default_factory=list,
        description="Ranked list of top-K product recommendations",
    )


class SimilarItemsRequest(BaseModel):
    """Request schema for POST /similar-items."""
    parent_asin: str = Field(..., description="Target product parent_asin", min_length=1)
    k: int = Field(default=10, ge=1, le=20, description="Number of similar items to return (1-20)")

    @field_validator("parent_asin")
    @classmethod
    def validate_parent_asin(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("parent_asin must be a non-empty string")
        return v.strip()


class SimilarItemsResponse(BaseModel):
    """Response schema for POST /similar-items."""
    parent_asin: str = Field(..., description="Target query product ASIN")
    strategy: str = Field(
        ...,
        description="Routing strategy used (item_cf_similarity, cold_item_content_match, popularity_fallback)",
    )
    similar_items: List[ItemRecommendation] = Field(
        default_factory=list,
        description="Ranked list of similar products",
    )
    recommendations: Optional[List[ItemRecommendation]] = Field(
        default=None,
        description="Alias for similar_items list",
    )


class ErrorResponse(BaseModel):
    """Standard error response schema."""
    detail: str = Field(..., description="Human-readable error description")

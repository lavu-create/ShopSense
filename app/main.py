"""
ShopSense FastAPI Recommendation & Inference Service
====================================================
Production-grade REST API exposing multi-tier personalized recommendations
and similar-item similarity matching.

Launch:
    uvicorn app.main:app --reload --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from typing import List

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.schemas import (
    ErrorResponse,
    HealthResponse,
    RecommendRequest,
    RecommendResponse,
    SimilarItemsRequest,
    SimilarItemsResponse,
)
from app.service import RecommenderService, get_service

# Configure structured logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("shopsense_api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Starts the API without loading the heavy recommendation model.
    The recommender initializes when the first recommendation request arrives.
    """
    logger.info("Initializing ShopSense Recommendation Engine...")
    service = get_service()
    logger.info("ShopSense API started. Recommendation engine will initialize on first request.")
    yield
    logger.info("ShopSense Recommendation API shutting down.")


# Initialize FastAPI application
app = FastAPI(
    title="ShopSense Recommendation API",
    description="Multi-tier personalized recommendation engine and similar-product similarity matching API.",
    version="1.0.0",
    lifespan=lifespan,
    responses={
        422: {"model": ErrorResponse, "description": "Validation Error"},
        500: {"model": ErrorResponse, "description": "Internal Server Error"},
    },
)

# ---------------------------------------------------------------------------
# CORS Configuration
# ---------------------------------------------------------------------------
cors_origins_env = os.getenv("CORS_ORIGINS", "*")
allowed_origins: List[str] = [origin.strip() for origin in cors_origins_env.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins if allowed_origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Clean Error Handlers (Traceback Shield)
# ---------------------------------------------------------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Formats validation errors into clean user-friendly JSON without tracebacks."""
    error_details = []
    for err in exc.errors():
        field = " -> ".join(str(loc) for loc in err.get("loc", []))
        msg = err.get("msg", "Invalid parameter value")
        error_details.append(f"Field '{field}': {msg}")

    clean_msg = "; ".join(error_details)
    logger.warning(f"Validation failure on {request.method} {request.url.path}: {clean_msg}")

    return JSONResponse(
        status_code=422,
        content={"detail": clean_msg},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Passes through HTTP exceptions with uniform structure."""
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Shields internal server errors and logs details securely."""
    logger.error(f"Internal server error handling {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing your request."},
    )


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    tags=["System"],
)
async def health_check() -> HealthResponse:
    """Returns service operational status."""
    return HealthResponse(status="ok", service="ShopSense Recommendation API")


@app.post(
    "/recommend",
    response_model=RecommendResponse,
    summary="Generate personalized recommendations for a user",
    tags=["Recommendations"],
    responses={
        422: {"model": ErrorResponse, "description": "Validation error (invalid user_id or k)"},
    },
)
async def get_recommendations(
    request: RecommendRequest,
    service: RecommenderService = Depends(get_service),
) -> RecommendResponse:
    """
    Routes recommendation requests dynamically:
    - 10+ interactions -> Hybrid CF (User-CF + Item-CF + Bayesian Popularity)
    - 1-9 interactions -> Content-Based Sparse History Profile
    - 0 interactions / Unknown User -> Bayesian Popularity Prior
    """
    strategy, recommendations = service.recommend(user_id=request.user_id, k=request.k)
    return RecommendResponse(
        user_id=request.user_id,
        strategy=strategy,
        recommendations=recommendations,
    )


@app.post(
    "/similar-items",
    response_model=SimilarItemsResponse,
    summary="Retrieve similar items for a target product",
    tags=["Recommendations"],
    responses={
        404: {"model": ErrorResponse, "description": "Product not found in metadata or CF catalog"},
        422: {"model": ErrorResponse, "description": "Validation error (invalid parent_asin or k)"},
    },
)
async def get_similar_items(
    request: SimilarItemsRequest,
    service: RecommenderService = Depends(get_service),
) -> SimilarItemsResponse:
    """
    Finds top-K similar items for a given product:
    - Known warm item in CF -> Item-CF Cosine Similarity
    - Cold item with metadata -> TF-IDF Content-Based Metadata Similarity
    - Unknown item -> 404 error with clean description
    """
    strategy, items = service.get_similar_items(parent_asin=request.parent_asin, k=request.k)
    return SimilarItemsResponse(
        parent_asin=request.parent_asin,
        strategy=strategy,
        similar_items=items,
        recommendations=items,
    )

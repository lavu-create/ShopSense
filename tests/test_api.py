"""
Integration and Unit Tests for ShopSense FastAPI Recommendation Service
========================================================================
Validates all REST endpoints, multi-tier routing logic, metadata enrichment,
schema conformance, and error-handling defenses.
"""

from pathlib import Path
import pytest
import pandas as pd
from fastapi.testclient import TestClient

from app.main import app

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_PATH = PROJECT_ROOT / "data" / "processed" / "train.parquet"
METADATA_PATH = PROJECT_ROOT / "data" / "processed" / "metadata_clean.parquet"


@pytest.fixture(scope="module")
def client():
    """Module-scoped test client managing FastAPI lifespan."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def test_data():
    """Provides test identifiers from parquet datasets."""
    train_df = pd.read_parquet(TRAIN_PATH)
    meta_df = pd.read_parquet(METADATA_PATH)

    vc = train_df["user_id"].value_counts()
    warm_user = vc[vc >= 10].index[0]
    sparse_user = vc[vc == 9].index[0]
    
    # Warm item with non-zero item-item similarities
    warm_item = "0634003224"
    
    # Cold item present in metadata but unseen in training
    cold_items = meta_df[~meta_df["parent_asin"].isin(train_df["parent_asin"])]["parent_asin"]
    cold_item = cold_items.iloc[0]

    return {
        "warm_user": warm_user,
        "sparse_user": sparse_user,
        "warm_item": warm_item,
        "cold_item": cold_item,
    }


# ---------------------------------------------------------------------------
# 1. Health Endpoint Tests
# ---------------------------------------------------------------------------

def test_health_endpoint(client):
    """Verify GET /health returns 200 OK and expected structure."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "ShopSense Recommendation API"


# ---------------------------------------------------------------------------
# 2. Recommendation Routing Endpoint Tests
# ---------------------------------------------------------------------------

def test_recommend_warm_user(client, test_data):
    """Verify POST /recommend routes warm user (>= 10 interactions) to hybrid_cf."""
    payload = {"user_id": test_data["warm_user"], "k": 10}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == test_data["warm_user"]
    assert data["strategy"] == "hybrid_cf"
    assert len(data["recommendations"]) == 10

    # Validate individual recommendation fields
    for idx, rec in enumerate(data["recommendations"]):
        assert rec["rank"] == idx + 1
        assert isinstance(rec["parent_asin"], str) and len(rec["parent_asin"]) > 0
        assert isinstance(rec["title"], str)
        assert isinstance(rec["category"], str)
        assert rec["price"] is None or isinstance(rec["price"], (int, float))


def test_recommend_sparse_user(client, test_data):
    """Verify POST /recommend routes sparse user (1-9 interactions) to content_based_sparse_history."""
    payload = {"user_id": test_data["sparse_user"], "k": 10}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == test_data["sparse_user"]
    assert data["strategy"] == "content_based_sparse_history"
    assert len(data["recommendations"]) == 10


def test_recommend_unknown_user(client):
    """Verify POST /recommend routes completely unknown user (0 interactions) to cold_user_popularity_prior."""
    payload = {"user_id": "UNKNOWN_COLD_USER_XYZ_0000", "k": 10}
    response = client.post("/recommend", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["user_id"] == "UNKNOWN_COLD_USER_XYZ_0000"
    assert data["strategy"] == "cold_user_popularity_prior"
    assert len(data["recommendations"]) == 10
    # No duplicate recommendations
    asins = [r["parent_asin"] for r in data["recommendations"]]
    assert len(set(asins)) == len(asins)


# ---------------------------------------------------------------------------
# 3. Similar Items Endpoint Tests
# ---------------------------------------------------------------------------

def test_similar_items_warm_item(client, test_data):
    """Verify POST /similar-items routes warm item in CF to item_cf_similarity."""
    payload = {"parent_asin": test_data["warm_item"], "k": 5}
    response = client.post("/similar-items", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["parent_asin"] == test_data["warm_item"]
    assert data["strategy"] == "item_cf_similarity"
    assert len(data["similar_items"]) == 5

    for idx, item in enumerate(data["similar_items"]):
        assert item["rank"] == idx + 1
        assert isinstance(item["parent_asin"], str) and len(item["parent_asin"]) > 0
        assert isinstance(item["title"], str)
        assert item["score"] is not None
        assert isinstance(item["score"], float)


def test_similar_items_cold_item(client, test_data):
    """Verify POST /similar-items routes cold item with metadata to cold_item_content_match."""
    payload = {"parent_asin": test_data["cold_item"], "k": 5}
    response = client.post("/similar-items", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["parent_asin"] == test_data["cold_item"]
    assert data["strategy"] == "cold_item_content_match"
    assert len(data["similar_items"]) == 5


def test_similar_items_unknown_item_not_found(client):
    """Verify POST /similar-items returns clean 404 for product absent from metadata."""
    payload = {"parent_asin": "COMPLETELY_NON_EXISTENT_ASIN_999", "k": 5}
    response = client.post("/similar-items", json=payload)
    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "not found in catalog metadata" in data["detail"].lower()
    # Traceback shield verification: detail should be clean string, not traceback
    assert "Traceback" not in data["detail"]


# ---------------------------------------------------------------------------
# 4. Validation and Parameter Limit Tests
# ---------------------------------------------------------------------------

def test_recommend_invalid_k(client, test_data):
    """Verify POST /recommend enforces 1 <= k <= 20 with clean 422 responses."""
    # Test k < 1
    r_zero = client.post("/recommend", json={"user_id": test_data["warm_user"], "k": 0})
    assert r_zero.status_code == 422
    assert "Field 'body -> k'" in r_zero.json()["detail"]

    # Test k > 20
    r_large = client.post("/recommend", json={"user_id": test_data["warm_user"], "k": 21})
    assert r_large.status_code == 422
    assert "Field 'body -> k'" in r_large.json()["detail"]

    # Test non-integer k
    r_type = client.post("/recommend", json={"user_id": test_data["warm_user"], "k": "invalid"})
    assert r_type.status_code == 422


def test_recommend_missing_and_empty_fields(client):
    """Verify POST /recommend rejects missing or empty user_id."""
    # Empty string
    r_empty = client.post("/recommend", json={"user_id": "", "k": 10})
    assert r_empty.status_code == 422

    # Whitespace only
    r_blank = client.post("/recommend", json={"user_id": "   ", "k": 10})
    assert r_blank.status_code == 422

    # Missing user_id field
    r_missing = client.post("/recommend", json={"k": 10})
    assert r_missing.status_code == 422


def test_similar_items_missing_and_empty_fields(client):
    """Verify POST /similar-items rejects missing or empty parent_asin."""
    r_empty = client.post("/similar-items", json={"parent_asin": "", "k": 5})
    assert r_empty.status_code == 422

    r_blank = client.post("/similar-items", json={"parent_asin": "   ", "k": 5})
    assert r_blank.status_code == 422

    r_missing = client.post("/similar-items", json={"k": 5})
    assert r_missing.status_code == 422


def test_recommend_k_limit_and_length(client, test_data):
    """Verify recommendation count strictly matches requested K."""
    for k_val in [1, 3, 7, 15, 20]:
        response = client.post("/recommend", json={"user_id": test_data["warm_user"], "k": k_val})
        assert response.status_code == 200
        data = response.json()
        assert len(data["recommendations"]) == k_val
        ranks = [r["rank"] for r in data["recommendations"]]
        assert ranks == list(range(1, k_val + 1))


def test_cors_headers_present(client):
    """Verify CORS preflight response headers."""
    response = client.options(
        "/recommend",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "POST",
        },
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") in ["*", "http://localhost:3000"]

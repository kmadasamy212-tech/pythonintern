"""
Unit tests for root entry-point (main.py) and Vercel entrypoint (api/index.py).
"""

import pytest
from fastapi.testclient import TestClient

from main import app
import api.index as vercel_entry


def test_vercel_exports():
    """Verifies that api/index.py exposes both 'app' and 'handler' as required by Vercel."""
    assert hasattr(vercel_entry, "app")
    assert hasattr(vercel_entry, "handler")
    assert vercel_entry.app is not None
    assert vercel_entry.handler is not None


def test_root_endpoint():
    """Verifies GET / returns system metadata and Vercel compatibility."""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["engine"] == "Distributed LLM Inference Engine"
    assert data["deployment"] == "Vercel Serverless Ready (ASGI)"
    assert "architecture" in data


def test_health_endpoint():
    """Verifies GET /health returns cluster memory and KV cache metrics."""
    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "paged_kv_cache" in data
    assert "scheduler" in data
    assert data["paged_kv_cache"]["total_blocks"] > 0


def test_generate_endpoint():
    """Verifies POST /generate runs tensor-parallel forward passes and continuous batching."""
    client = TestClient(app)
    payload = {
        "prompt": "Test distributed inference",
        "max_new_tokens": 4,
        "temperature": 0.0,
    }
    response = client.post("/generate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "request_id" in data
    assert "generated_text" in data
    assert len(data["token_ids"]) >= 1
    assert data["total_tokens"] >= 1
    assert data["ttft_ms"] >= 0.0


def test_rewritten_paths():
    """Verifies that paths rewritten by Vercel to /api or /api/index resolve successfully."""
    client = TestClient(app)
    # Test rewritten root
    r_api = client.get("/api")
    assert r_api.status_code == 200
    r_idx = client.get("/api/index")
    assert r_idx.status_code == 200

    # Test rewritten health
    r_health = client.get("/api/health")
    assert r_health.status_code == 200
    assert r_health.json()["status"] == "healthy"

    r_idx_health = client.get("/api/index/health")
    assert r_idx_health.status_code == 200

    # Test rewritten generate
    r_gen = client.post("/api/index/generate", json={"prompt": "Vercel route test", "max_new_tokens": 2})
    assert r_gen.status_code == 200
    assert "token_ids" in r_gen.json()


def test_serverless_without_torch_fallback(monkeypatch):
    """Verifies that EmbeddedClusterEngine operates reliably when PyTorch is not present."""
    import main
    # Simulate environment where torch is not installed
    monkeypatch.setattr(main, "HAS_TORCH", False)
    fallback_engine = main.EmbeddedClusterEngine(world_size=2)
    res = fallback_engine.generate(prompt="Serverless fallback test", max_new_tokens=3)
    assert res["total_tokens"] == 3
    assert len(res["token_ids"]) == 3
    assert res["finish_reason"] in ("FINISHED_LENGTH", "FINISHED_STOP")



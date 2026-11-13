from fastapi.testclient import TestClient

from app.main import create_app
from app.obs import metrics_store


def test_health_counted_and_metrics_endpoint_present():
    metrics_store.reset()
    client = TestClient(create_app())
    client.get("/health")
    resp = client.get("/metrics")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"total_requests", "total_tokens", "total_cost_usd",
                                "avg_latency_ms", "refusals"}
    assert body["total_requests"] >= 1


def test_app_state_exposes_metrics_store():
    app = create_app()
    assert app.state.metrics_store is metrics_store

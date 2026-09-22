import pytest
from fastapi.testclient import TestClient

from agent_service.api import create_app
from agent_service.config import Settings


@pytest.fixture(scope="module")
def client():
    app = create_app(Settings(backend="rules"))
    with TestClient(app) as c:
        yield c


def test_health_and_readiness(client):
    assert client.get("/healthz").json()["status"] == "ok"
    ready = client.get("/readyz").json()
    assert ready == {"status": "ready", "backend": "rules"}


def test_triage_returns_structured_response_and_request_id(client):
    r = client.post("/v1/triage", json={"text": "You charged me twice this month, please refund EUR 84,50."},
                    headers={"x-request-id": "test-123"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["request_id"] == "test-123"
    assert r.headers["x-request-id"] == "test-123"
    assert body["outcome"] == "answered"
    assert body["classification"]["category"] == "billing"
    assert body["extraction"]["amount_eur"] == 84.5
    assert body["citations"]
    assert {t["node"] for t in body["trace"]} >= {"classify", "draft", "guard"}


def test_validation_rejects_short_text(client):
    assert client.post("/v1/triage", json={"text": "hi"}).status_code == 422


def test_metrics_endpoint_exposes_counters(client):
    client.post("/v1/triage", json={"text": "Do you cover e-bikes under the contents policy?"})
    text = client.get("/metrics").text
    assert "agent_requests_total" in text
    assert "agent_node_latency_seconds" in text
    assert "agent_triage_outcomes_total" in text

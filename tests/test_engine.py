import numpy as np
from fastapi.testclient import TestClient

from contextcart.app import app, engine
from contextcart.engine import Engine
from contextcart.retrieval import BM25, HashEncoder, fuse
from contextcart.state import MemoryState


def test_hash_is_stable_and_normalized():
    encoder = HashEncoder()
    assert np.allclose(encoder.encode("trail hiking"), encoder.encode("hiking trail"))
    assert np.isclose(np.linalg.norm(encoder.encode("trail")), 1)
    assert not encoder.encode("").any()


def test_bm25_and_fusion():
    assert BM25(["coffee beans", "hiking boots"]).scores("coffee")[0] > 0
    assert fuse({"a": 1, "b": 0}, {"b": 1}, 0)["a"] > 0
    assert fuse({}, {}) == {}


def test_filters_and_relevance():
    e = Engine()
    rows = e.search("coffee", "session-a", category="kitchen", max_price=40)["items"]
    assert rows and all(p["category"] == "kitchen" and p["price"] <= 40 and p["stock"] for p in rows)
    assert e.search("coffee", "session-a")["items"][0]["id"].startswith("kitchen")
    assert e.search("coffee", "session-a", category="missing")["items"] == []


def test_session_isolation_expiry_delete_and_personalization():
    now = [0]
    state = MemoryState(ttl=5, clock=lambda: now[0])
    e = Engine(state=state)
    before = e.search("travel", "session-a")
    e.click("session-a", "outdoors-1")
    after = e.search("travel", "session-a")
    assert after["personalized"] and before["items"] != after["items"]
    assert not e.search("travel", "session-b")["personalized"]
    now[0] = 6
    assert not e.search("travel", "session-a")["personalized"]
    e.click("session-a", "outdoors-1")
    state.delete("session-a")
    assert state.get("session-a") is None


def test_bounded_state_and_ema():
    s = MemoryState(capacity=1)
    s.update("one", [1.0, 0.0])
    s.update("one", [0.0, 1.0])
    assert np.allclose(s.get("one"), [0.7, 0.3])
    s.update("two", [1.0, 0.0])
    assert s.get("one") is None


class Broken:
    def query(self, *args):
        raise TimeoutError()

    def rerank(self, *args):
        raise TimeoutError()

    def get(self, *args):
        raise TimeoutError()


def test_failures_are_visible_and_fallback_works():
    e = Engine(index=Broken(), reranker=Broken(), state=Broken())
    result = e.search("coffee", "session-a")
    assert result["items"] and len(result["warnings"]) == 3


def test_api_flow_and_validation():
    engine.cache_clear()
    with TestClient(app) as client:
        assert client.get("/").status_code == 200
        assert client.get("/health").json()["status"] == "ok"
        assert (
            client.get("/api/recommendations", params={"q": "coffee", "session": "short"}).status_code == 422
        )
        assert (
            client.post("/api/events", json={"session": "session-a", "product_id": "bad"}).status_code == 404
        )
        assert (
            client.post("/api/events", json={"session": "session-a", "product_id": "kitchen-1"}).status_code
            == 202
        )
        params = {"q": "coffee", "session": "session-a"}
        assert client.get("/api/recommendations", params=params).json()["personalized"]
        assert client.delete("/api/session", params={"session": "session-a"}).status_code == 204
        assert not client.get("/api/recommendations", params=params).json()["personalized"]

import httpx
import pytest

from contextcart.cloud import CohereEncoder, CohereReranker, PineconeIndex


def wire(adapter, handler):
    adapter.client.close()
    adapter.client = httpx.Client(transport=httpx.MockTransport(handler), base_url="https://test.invalid")


def test_embedding_contract():
    e = CohereEncoder("test-placeholder")
    wire(e, lambda r: httpx.Response(200, json={"embeddings": {"float": [[3.0, 4.0]]}}))
    assert list(e.encode("coffee")) == [0.6, 0.8]


def test_pinecone_query_and_upsert():
    seen = []
    p = PineconeIndex("https://example.pinecone.io", "test-placeholder", "test")

    def handle(r):
        seen.append(r.url.path)
        return httpx.Response(200, json={"matches": [{"id": "one", "score": 0.9}]})

    wire(p, handle)
    assert p.query([1.0, 0.0], 5) == {"one": 0.9}
    p.upsert([{"id": "one"}], [[1.0, 0.0]])
    assert seen == ["/query", "/vectors/upsert"]


def test_rerank_and_invalid_response():
    c = CohereReranker("test-placeholder")
    wire(c, lambda r: httpx.Response(200, json={"results": [{"index": 0, "relevance_score": 0.8}]}))
    assert c.rerank("coffee", [{"text": "beans"}])[0][1] == 0.8
    with pytest.raises(ValueError):
        c.rerank("coffee", [{"text": "beans"}, {"text": "tea"}])


def test_external_failure_raises():
    p = PineconeIndex("https://example.pinecone.io", "test-placeholder", "test")
    wire(p, lambda r: httpx.Response(429))
    with pytest.raises(httpx.HTTPStatusError):
        p.query([1], 1)
    with pytest.raises(ValueError):
        PineconeIndex("http://localhost", "test-placeholder", "test")

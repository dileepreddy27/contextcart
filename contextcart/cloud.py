"""Explicit external adapters. Credentials stay in environment variables."""

import httpx
import numpy as np

from .retrieval import unit


class CohereEncoder:
    name = "cohere-embed-english-v3.0"
    dimension = 1024

    def __init__(self, key):
        self.client = httpx.Client(
            base_url="https://api.cohere.com", timeout=15, headers={"Authorization": "Bearer " + key}
        )

    def encode(self, text, input_type="search_query"):
        r = self.client.post(
            "/v2/embed",
            json={
                "model": "embed-english-v3.0",
                "texts": [text],
                "input_type": input_type,
                "embedding_types": ["float"],
            },
        )
        r.raise_for_status()
        return unit(np.asarray(r.json()["embeddings"]["float"][0]))


class PineconeIndex:
    def __init__(self, host, key, namespace):
        if not host.startswith("https://") or not host.split("/")[2].endswith(".pinecone.io"):
            raise ValueError("Pinecone host must be an HTTPS pinecone.io endpoint")
        self.client = httpx.Client(base_url=host, timeout=15, headers={"Api-Key": key})
        self.namespace = namespace

    def query(self, vector, limit):
        r = self.client.post(
            "/query",
            json={
                "vector": list(vector),
                "topK": limit,
                "namespace": self.namespace,
                "includeMetadata": False,
            },
        )
        r.raise_for_status()
        return {m["id"]: m["score"] for m in r.json()["matches"]}

    def upsert(self, products, vectors):
        for offset in range(0, len(products), 100):
            payload = [
                {"id": p["id"], "values": list(v)}
                for p, v in zip(products[offset : offset + 100], vectors[offset : offset + 100])
            ]
            r = self.client.post("/vectors/upsert", json={"namespace": self.namespace, "vectors": payload})
            r.raise_for_status()


class CohereReranker:
    def __init__(self, key):
        self.client = httpx.Client(
            base_url="https://api.cohere.com", timeout=15, headers={"Authorization": "Bearer " + key}
        )

    def rerank(self, query, products):
        r = self.client.post(
            "/v2/rerank",
            json={
                "model": "rerank-v3.5",
                "query": query,
                "documents": [p["text"] for p in products],
                "top_n": len(products),
            },
        )
        r.raise_for_status()
        rows = r.json()["results"]
        indices = [row["index"] for row in rows]
        if sorted(indices) != list(range(len(products))):
            raise ValueError("Incomplete rerank response")
        return [(products[row["index"]], row["relevance_score"]) for row in rows]

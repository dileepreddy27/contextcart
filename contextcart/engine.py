import json
import time
from pathlib import Path

import numpy as np

from .retrieval import BM25, HashEncoder, fuse, unit
from .state import MemoryState


def catalog():
    return json.loads(Path(__file__).with_name("data").joinpath("products.json").read_text())


class Engine:
    def __init__(self, products=None, encoder=None, state=None, index=None, reranker=None):
        self.products = products if products is not None else catalog()
        self.by_id = {p["id"]: p for p in self.products}
        self.encoder = encoder or HashEncoder()
        self.state = state or MemoryState()
        self.index, self.reranker = index, reranker
        self.bm25 = BM25([p["text"] for p in self.products])
        # Cloud document vectors must use the document input type.
        self.vectors = np.array(
            [
                self.encoder.encode(p["text"], input_type="search_document")
                if self.encoder.name.startswith("cohere")
                else self.encoder.encode(p["text"])
                for p in self.products
            ]
        )

    def click(self, session, product_id):
        if product_id not in self.by_id:
            raise KeyError(product_id)
        i = next(i for i, p in enumerate(self.products) if p["id"] == product_id)
        self.state.update(session, self.vectors[i].tolist())

    def search(self, query, session, limit=6, category=None, max_price=None, weight=0.5):
        start = time.perf_counter()
        warnings = []
        q = self.encoder.encode(query)
        try:
            history = self.state.get(session)
        except Exception:
            history = None
            warnings.append("Session store unavailable; using query only")
        context = unit(0.65 * q + 0.35 * unit(history)) if history is not None else q
        eligible = {
            p["id"]
            for p in self.products
            if p["stock"] > 0
            and (not category or p["category"] == category)
            and (max_price is None or p["price"] <= max_price)
        }
        sparse = {p["id"]: s for p, s in zip(self.products, self.bm25.scores(query)) if p["id"] in eligible}
        dense = {
            p["id"]: float(s) for p, s in zip(self.products, self.vectors @ context) if p["id"] in eligible
        }
        retrieval = "local-exact"
        if self.index:
            try:
                dense = {
                    k: v
                    for k, v in self.index.query(context, min(100, len(self.products))).items()
                    if k in eligible
                }
                retrieval = "pinecone"
            except Exception:
                warnings.append("Pinecone unavailable; using local exact vectors")
        scores = fuse(sparse, dense, weight)
        ids = sorted(scores, key=lambda k: (-scores[k], k))[:30]
        candidates = [self.by_id[k] for k in ids]
        ranked = [(p, scores[p["id"]]) for p in candidates]
        reranking = "rrf-baseline"
        if candidates and self.reranker:
            try:
                ranked = self.reranker.rerank(query, candidates)
                reranking = "cohere-rerank-v3.5"
            except Exception:
                warnings.append("Cohere rerank unavailable; keeping fused ranking")
        return {
            "items": [
                {
                    **p,
                    "score": round(float(s), 6),
                    "bm25": round(sparse.get(p["id"], 0), 4),
                    "vector_similarity": round(dense.get(p["id"], 0), 4),
                }
                for p, s in ranked[:limit]
            ],
            "mode": self.encoder.name,
            "retrieval": retrieval,
            "reranking": reranking,
            "personalized": history is not None,
            "candidate_count": len(candidates),
            "elapsed_ms": round((time.perf_counter() - start) * 1000, 3),
            "warnings": warnings,
        }

"""Small author-labeled synthetic relevance diagnostic, not an accuracy benchmark."""

import json
import math

from .engine import Engine

CASES = [
    ("waterproof hiking", ["outdoors-1", "outdoors-2", "outdoors-3"]),
    ("coffee", ["kitchen-1", "kitchen-2", "kitchen-5", "kitchen-6"]),
    ("wireless office", ["office-1", "office-6"]),
    ("yoga exercise", ["fitness-1", "fitness-3", "fitness-4"]),
    ("running hydration", ["fitness-2", "fitness-5"]),
    ("desk adjustable", ["office-2", "office-3", "office-4"]),
]


def evaluate():
    e = Engine()
    result = {}
    for label, weight in [("bm25", 0), ("hashed_vector", 1), ("hybrid", 0.5)]:
        recalls, ndcgs = [], []
        for query, relevant in CASES:
            ids = [p["id"] for p in e.search(query, "evaluation", limit=5, weight=weight)["items"]]
            recalls.append(len(set(ids) & set(relevant)) / len(relevant))
            dcg = sum(1 / math.log2(i + 2) for i, key in enumerate(ids) if key in relevant)
            ideal = sum(1 / math.log2(i + 2) for i in range(min(5, len(relevant))))
            ndcgs.append(dcg / ideal)
        result[label] = {
            "recall_at_5": round(sum(recalls) / len(recalls), 4),
            "ndcg_at_5": round(sum(ndcgs) / len(ndcgs), 4),
        }
    return {"catalog_size": len(e.products), "queries": len(CASES), "results": result}


if __name__ == "__main__":
    print(json.dumps(evaluate(), indent=2))

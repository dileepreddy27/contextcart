"""Exact local retrieval; hashed vectors are a lexical baseline, not a semantic model."""

import hashlib
import math
import re
from collections import Counter

import numpy as np


def tokens(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def unit(v):
    v = np.asarray(v, dtype=float)
    norm = np.linalg.norm(v)
    return v / norm if norm else v


class HashEncoder:
    dimension = 128
    name = "hashed-lexical-128"

    def encode(self, text):
        v = np.zeros(self.dimension)
        for word in tokens(text):
            h = int.from_bytes(hashlib.sha256(word.encode()).digest()[:8], "big")
            v[h % self.dimension] += 1 if (h >> 8) % 2 else -1
        return unit(v)


class BM25:
    def __init__(self, documents):
        self.docs = [Counter(tokens(d)) for d in documents]
        self.lengths = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lengths) / max(1, len(self.docs))
        self.df = Counter(w for d in self.docs for w in d)

    def scores(self, query):
        result = []
        for d, length in zip(self.docs, self.lengths):
            score = 0.0
            for w in set(tokens(query)):
                f = d[w]
                idf = math.log(1 + (len(self.docs) - self.df[w] + 0.5) / (self.df[w] + 0.5))
                if f:
                    score += idf * f * 2.5 / (f + 1.5 * (0.25 + 0.75 * length / self.avg))
            result.append(score)
        return result


def fuse(sparse, dense, weight=0.5):
    """Weighted reciprocal rank fusion; exclude nonpositive evidence."""
    result = {}
    for scores, factor in ((sparse, 1 - weight), (dense, weight)):
        ordered = sorted(scores, key=lambda k: (-scores[k], k))
        for rank, key in enumerate(ordered, 1):
            if scores[key] > 0 and factor > 0:
                result[key] = result.get(key, 0) + factor / (60 + rank)
    return result

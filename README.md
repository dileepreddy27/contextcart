# ContextCart

[![checks](https://github.com/dileepreddy27/contextcart/actions/workflows/ci.yml/badge.svg)](https://github.com/dileepreddy27/contextcart/actions/workflows/ci.yml)

**Hybrid product discovery with expiring session context and inspectable ranking.** A FastAPI recommendation lab combining BM25, vector retrieval, weighted reciprocal-rank fusion, and optional transformer reranking. Browse products to change a session vector, inspect retrieval scores, filter inventory, and clear the session from the same interface.

![Architecture](docs/architecture.svg)

### Why this exists

Keyword search captures exact product terms but misses context from a shopper's recent browsing. ContextCart demonstrates how to combine the query with a bounded session history, retrieve candidates through two independent channels, and expose why results changed. It is a portfolio demonstration with 24 original synthetic products, not a production storefront or validated CTR optimizer.

### Run the local demo

Requires Python 3.11+. No API keys, GPU, external database, or model download needed.

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -e ".[dev,redis]"
python -m uvicorn contextcart.app:app --host 127.0.0.1 --port 8133
```

Open http://127.0.0.1:8133 and search `travel`. Browse **Alpine Trail Pack**, then repeat the search to see session context influence vector scores. Try a category or price filter. **Forget session** deletes the server-side vector. API documentation is at `/docs`.

The UI shows the active encoder, retrieval backend, reranking mode, measured request computation time, candidate count in API responses, and degraded-mode warnings. Local mode uses a **128-dimensional signed hash of tokens**: a deterministic lexical vector baseline, not pretrained semantic embeddings. Its reranking mode is explicitly `rrf-baseline`; it does not simulate a transformer result.

### Architecture and implementation

- **BM25:** term frequency, document frequency, length normalization, and configurable fusion weight; no external search service required.
- **Context:** click vectors update an exponential moving average (70% previous / 30% new). Query vectors mix 65% query / 35% normalized session context. First click initializes context directly.
- **State:** bounded in-process memory by default (10,000 sessions), 30-minute TTL refreshed by clicks. Optional Redis stores pseudonymous hashed session keys and updates vectors atomically with Lua. Redis and embedding modes use separate namespaces.
- **Retrieval:** local exact cosine scores or Pinecone dense queries. BM25 remains local in both modes; this is application-level hybrid fusion, not Pinecone sparse retrieval. Weighted reciprocal-rank fusion uses constant 60 and excludes nonpositive evidence.
- **Ranking:** at most 30 fused candidates go to optional Cohere rerank-v3.5. Out-of-stock and price/category-ineligible items are excluded. Cloud query fetches at most 100 neighbors before filtering, which can reduce recall on larger catalogs.
- **Failure behavior:** Redis read failure yields a visible query-only fallback; Pinecone failure uses locally held vectors; rerank failure retains fused ranking. Embedding failures return 503. Failed clicks return 503 rather than falsely acknowledging persistence.
- **API:** `GET /api/recommendations`, `GET /api/products`, `POST /api/events`, `DELETE /api/session`, and `GET /health`. Input sizes and ranking parameters are bounded.

### Optional Redis

```sh
docker compose up -d
# Set REDIS_URL=redis://localhost:6379/0 in your shell before starting the app.
```

The Compose service binds only to localhost and disables persistence. In-memory state is per-process and disappears on restart; Redis is needed for shared state across workers. Session IDs are browser-generated, not authenticated identities. Never put real customer data into this demo.

### Optional cloud adapters

Use an existing **1024-dimensional cosine Pinecone vector index**, a dedicated namespace, and a Cohere account with access to `embed-english-v3.0` and `rerank-v3.5`. No resources are provisioned by this repository. Set these environment variables in your shell (PowerShell uses `$env:NAME="value"`):

```text
CONTEXTCART_MODE=cloud
COHERE_API_KEY=<your-key>
PINECONE_API_KEY=<your-key>
PINECONE_HOST=https://<your-index-host>.pinecone.io
PINECONE_NAMESPACE=contextcart-v1
REDIS_URL=redis://localhost:6379/0
```

Run `python -m contextcart.ingest` to explicitly upload the synthetic catalog, then start the server. Document embedding calls use `search_document`; queries use `search_query`. Cloud startup embeds the catalog again to support session clicks and local fallback. This is deliberately simple and incurs API cost; durable embedding caching/batching is future work. Queries and product text are sent to Cohere and vectors to Pinecone. Do not enable cloud mode without understanding those providers' terms and costs.

Adapter contracts follow [Cohere Embed v2](https://docs.cohere.com/v2/reference/embed), [Cohere Rerank](https://docs.cohere.com/v2/reference/rerank), and [Pinecone vector operations](https://sdk.pinecone.io/python/how-to/vectors/upsert-and-query.html). HTTP contract tests are mocked; live paid cloud calls have **not** been run.

### Tests and measured diagnostic

```sh
python -m ruff check .
python -m pytest -q
python -m contextcart.evaluate
```

Local verification: **11 passed, 1 skipped** (the real Redis integration requires `TEST_REDIS_URL`). CI starts a real Redis service and runs that integration test as well. Tests cover API validation and session reset, ranking eligibility, normalized deterministic vectors, session isolation/expiry/capacity, EMA updates, service fallback, and HTTP adapter contracts.

The included evaluation uses six author-labeled queries over the same 24 synthetic products. No training occurs; there is no learned model or held-out training split. Results from an actual local execution:

| Retrieval mode | Recall@5 | NDCG@5 |
| --- | ---: | ---: |
| BM25 | 1.0000 | 1.0000 |
| Hashed vector | 0.9583 | 0.9666 |
| Hybrid | 1.0000 | 1.0000 |

These easy lexical fixtures provide regression evidence only. Hybrid ties BM25 here; no improvement in semantic accuracy, conversion, or latency at scale is claimed. Personalization is tested behaviorally, not evaluated with real clickstream data. Cloud ranking quality is unmeasured.

### Tradeoffs and limitations

The local vector baseline has hash collisions and no semantic understanding. The complete catalog and document vectors remain in process. There is no result cache, durable event log, inventory synchronization, authentication, rate limiting, or production observability. Redis TTL is bounded per session but aggregate Redis memory policy must be configured by an operator. Cloud reranking uses the query alone, so it may weaken session-driven ordering. Health checks initialize the engine but are not dependency readiness probes. The demo is intended to bind to localhost, not be exposed publicly without access controls.

### Data, license, and evidence-backed resume bullets

All catalog names, descriptions, prices, stock, and relevance labels were authored as synthetic fixtures for this project. There are no real customers, scraped product records, or third-party images. Code and synthetic data are MIT licensed; dependency and service notices are in [THIRD_PARTY.md](THIRD_PARTY.md).

- Built a FastAPI recommendation lab combining BM25, vector retrieval, reciprocal-rank fusion, inventory filters, and inspectable per-result scores.
- Implemented expiring session personalization with bounded local state and an atomic Redis Lua update, plus explicit fallbacks for vector retrieval and reranking failures.
- Added Pinecone/Cohere HTTP adapters and reproducible retrieval diagnostics with clear separation between locally verified baselines and unrun cloud integrations.

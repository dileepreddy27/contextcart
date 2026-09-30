import os
from functools import lru_cache
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .engine import Engine

app = FastAPI(title="ContextCart", version="0.1.0")
Session = Annotated[str, Query(min_length=8, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")]


@lru_cache
def engine():
    kwargs = {}
    if os.getenv("CONTEXTCART_MODE", "local") == "cloud":
        from .cloud import CohereEncoder, CohereReranker, PineconeIndex

        key = os.environ["COHERE_API_KEY"]
        kwargs.update(
            encoder=CohereEncoder(key),
            reranker=CohereReranker(key),
            index=PineconeIndex(
                os.environ["PINECONE_HOST"],
                os.environ["PINECONE_API_KEY"],
                os.getenv("PINECONE_NAMESPACE", "contextcart-v1"),
            ),
        )
    if os.getenv("REDIS_URL"):
        import redis

        from .state import RedisState

        kwargs["state"] = RedisState(
            redis.Redis.from_url(os.environ["REDIS_URL"], socket_timeout=2, socket_connect_timeout=2),
            namespace=kwargs["encoder"].name if "encoder" in kwargs else "hashed-lexical-128",
        )
    return Engine(**kwargs)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).with_name("static") / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "mode": engine().encoder.name}


@app.get("/api/products")
def products():
    return engine().products


@app.get("/api/recommendations")
def recommend(
    session: Session,
    q: str = Query(min_length=1, max_length=300),
    limit: int = Query(6, ge=1, le=20),
    category: str | None = None,
    max_price: float | None = Query(None, ge=0),
    weight: float = Query(0.5, ge=0, le=1),
):
    try:
        return engine().search(q, session, limit, category, max_price, weight)
    except Exception as exc:
        raise HTTPException(503, "Recommendation service unavailable") from exc


class Click(BaseModel):
    session: str = Field(min_length=8, max_length=80, pattern=r"^[a-zA-Z0-9_-]+$")
    product_id: str = Field(min_length=1, max_length=80)


@app.post("/api/events", status_code=202)
def event(click: Click):
    try:
        engine().click(click.session, click.product_id)
    except KeyError as exc:
        raise HTTPException(404, "Unknown product") from exc
    except Exception as exc:
        raise HTTPException(503, "Session store unavailable") from exc
    return {"accepted": True}


@app.delete("/api/session", status_code=204)
def forget(session: Session):
    try:
        engine().state.delete(session)
    except Exception as exc:
        raise HTTPException(503, "Session store unavailable") from exc

"""Explicit opt-in upload to an existing, user-provisioned Pinecone index."""

from .app import engine

if __name__ == "__main__":
    service = engine()
    if not service.index:
        raise SystemExit("Set CONTEXTCART_MODE=cloud and cloud credentials first")
    service.index.upsert(service.products, service.vectors)
    print(f"Uploaded {len(service.products)} synthetic products")

import os

import pytest

from contextcart.state import RedisState


def test_real_redis_atomic_update_and_delete():
    url = os.getenv("TEST_REDIS_URL")
    if not url:
        pytest.skip("Set TEST_REDIS_URL for real Redis integration")
    import uuid

    import redis

    client = redis.Redis.from_url(url)
    state = RedisState(client, ttl=10)
    session = str(uuid.uuid4())
    try:
        state.update(session, [1.0, 0.0])
        state.update(session, [0.0, 1.0])
        assert state.get(session) == [0.7, 0.3]
        assert 0 < client.ttl(state.key(session)) <= 10
        state.delete(session)
        assert state.get(session) is None
    finally:
        state.delete(session)

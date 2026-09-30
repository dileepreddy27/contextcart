import hashlib
import threading
import time


class MemoryState:
    """Bounded, process-local session vectors with TTL. No durable identity."""

    def __init__(self, ttl=1800, capacity=10000, clock=time.monotonic):
        self.ttl, self.capacity, self.clock = ttl, capacity, clock
        self.data = {}
        self.lock = threading.Lock()

    def get(self, session):
        with self.lock:
            value = self.data.get(session)
            if value and value[1] > self.clock():
                return list(value[0])
            self.data.pop(session, None)
            return None

    def update(self, session, vector):
        with self.lock:
            old = self.data.get(session)
            if old and old[1] > self.clock():
                vector = [0.7 * a + 0.3 * b for a, b in zip(old[0], vector)]
            if session not in self.data and len(self.data) >= self.capacity:
                self.data.pop(min(self.data, key=lambda k: self.data[k][1]))
            self.data[session] = (list(vector), self.clock() + self.ttl)

    def delete(self, session):
        with self.lock:
            self.data.pop(session, None)


class RedisState:
    # Atomic EMA update avoids lost concurrent clicks; keys are pseudonymous hashes.
    SCRIPT = """
local old = redis.call('GET', KEYS[1])
local v = cjson.decode(ARGV[1])
if old then
 local prev = cjson.decode(old)
 for i=1,#v do v[i] = 0.7 * prev[i] + 0.3 * v[i] end
end
redis.call('SET', KEYS[1], cjson.encode(v), 'EX', ARGV[2])
return 1
"""

    def __init__(self, client, ttl=1800, namespace="hashed-lexical-128"):
        self.client, self.ttl, self.namespace = client, ttl, namespace

    def key(self, session):
        return "contextcart:" + self.namespace + ":" + hashlib.sha256(session.encode()).hexdigest()

    def get(self, session):
        import json

        value = self.client.get(self.key(session))
        return json.loads(value) if value else None

    def update(self, session, vector):
        import json

        self.client.eval(self.SCRIPT, 1, self.key(session), json.dumps(list(vector)), self.ttl)

    def delete(self, session):
        self.client.delete(self.key(session))

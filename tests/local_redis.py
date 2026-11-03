"""A tiny in-memory stand-in for redis.asyncio.Redis (only what RAGService uses),
so cache tests run offline without a Redis server."""
import asyncio
import fnmatch

from redis import exceptions as redis_exceptions


class FakeLock:
    def __init__(self, lock: asyncio.Lock):
        self._lock = lock

    async def acquire(self) -> bool:
        await self._lock.acquire()
        return True

    async def release(self) -> None:
        self._lock.release()


class FakeRedis:
    def __init__(self):
        self.store: dict[str, str] = {}
        self.ttls: dict[str, int | None] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, key):
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value
        self.ttls[key] = ex
        return True

    async def incr(self, key):
        value = int(self.store.get(key, 0)) + 1
        self.store[key] = str(value)
        return value

    def lock(self, name, timeout=None, blocking_timeout=None):
        return FakeLock(self._locks.setdefault(name, asyncio.Lock()))

    async def scan_iter(self, match="*", count=None):
        for key in list(self.store):
            if fnmatch.fnmatch(key, match):
                yield key

    async def unlink(self, *keys):
        for key in keys:
            self.store.pop(key, None)
        return len(keys)


class BrokenRedis(FakeRedis):
    """Every call fails, like an unreachable server."""

    async def get(self, key):
        raise redis_exceptions.ConnectionError("down")

    async def set(self, key, value, ex=None):
        raise redis_exceptions.ConnectionError("down")

    async def incr(self, key):
        raise redis_exceptions.ConnectionError("down")

    async def scan_iter(self, match="*", count=None):
        raise redis_exceptions.ConnectionError("down")
        yield  # pragma: no cover  (makes this an async generator)

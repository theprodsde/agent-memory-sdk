"""Tests for the Redis memory store backend.

Uses fakeredis so no real Redis server is required in CI.
Skip the whole module if fakeredis is not installed.
"""
from __future__ import annotations

import pytest

try:
    import fakeredis

    FAKEREDIS_AVAILABLE = True
except ImportError:
    FAKEREDIS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not FAKEREDIS_AVAILABLE, reason="fakeredis not installed"
)


def make_store():
    from agent_memory.redis_store import RedisMemoryStore

    fake_client = fakeredis.FakeRedis(decode_responses=True)
    return RedisMemoryStore(redis_client=fake_client)


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_store_and_get():
    store = make_store()
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="hello", response="world")
    stored = store.store(entry)
    fetched = store.get(stored.id)
    assert fetched is not None
    assert fetched.query == "hello"
    assert fetched.response == "world"


def test_count_increments():
    store = make_store()
    from agent_memory.models import MemoryEntry

    assert store.count == 0
    store.store(MemoryEntry(query="a", response="b"))
    assert store.count == 1
    store.store(MemoryEntry(query="c", response="d"))
    assert store.count == 2


def test_update():
    store = make_store()
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="original", response="old response")
    store.store(entry)
    entry.response = "updated response"
    store.update(entry)
    fetched = store.get(entry.id)
    assert fetched is not None
    assert fetched.response == "updated response"


def test_delete():
    store = make_store()
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="to delete", response="bye")
    store.store(entry)
    assert store.count == 1
    assert store.delete(entry.id) is True
    assert store.get(entry.id) is None
    assert store.count == 0
    assert store.delete(entry.id) is False  # idempotent


def test_get_missing():
    store = make_store()
    assert store.get("nonexistent-id") is None


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


def test_list_scope_filter():
    store = make_store()
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(MemoryEntry(query="user q", response="r", scope=MemoryScope.USER))
    store.store(MemoryEntry(query="global q", response="r", scope=MemoryScope.GLOBAL))

    user_entries = store.list_all(scopes=[MemoryScope.USER])
    assert len(user_entries) == 1
    assert user_entries[0].scope == MemoryScope.USER

    global_entries = store.list_all(scopes=[MemoryScope.GLOBAL])
    assert len(global_entries) == 1


def test_list_archived_excluded_by_default():
    store = make_store()
    from agent_memory.models import MemoryEntry

    e = MemoryEntry(query="archived", response="r", archived=True)
    store.store(e)
    assert store.list_all() == []
    assert len(store.list_all(include_archived=True)) == 1


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def test_keyword_search_finds_match():
    store = make_store()
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="Python async programming", response="use asyncio"))
    store.store(MemoryEntry(query="How to bake bread", response="flour and yeast"))

    results = store.keyword_search("async programming", top_k=5)
    assert len(results) > 0
    best_entry, best_score = results[0]
    assert "async" in best_entry.query.lower() or "async" in best_entry.response.lower()
    assert best_score > 0


def test_keyword_search_empty_store():
    store = make_store()
    assert store.keyword_search("anything") == []


# ---------------------------------------------------------------------------
# TTL / expiry
# ---------------------------------------------------------------------------


def test_expired_excluded_by_default():
    store = make_store()
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="expired", response="old", expires_at=past)
    e.refresh_state()
    store.store(e)
    assert store.list_all() == []
    assert len(store.list_all(include_expired=True)) == 1


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats():
    store = make_store()
    from agent_memory.models import MemoryEntry, MemoryType

    store.store(MemoryEntry(query="q1", response="r1", type=MemoryType.FACT))
    store.store(MemoryEntry(query="q2", response="r2", type=MemoryType.CONVERSATION))
    s = store.stats()
    assert s["total"] == 2
    assert "fact" in s["by_type"]
    assert "conversation" in s["by_type"]


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_marks_expired():
    store = make_store()
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired()
    assert result["expired"] >= 1


def test_cleanup_delete():
    store = make_store()
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired(delete=True)
    assert result["deleted"] >= 1
    assert store.count == 0

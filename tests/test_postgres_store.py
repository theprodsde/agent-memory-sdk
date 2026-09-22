"""Tests for the PostgreSQL memory store backend.

Requires a running Postgres instance.  The connection string is read from
``AGENT_MEMORY_POSTGRES_DSN`` (default: ``postgresql://localhost/agent_memory_test``).
The entire module is skipped when psycopg2 is not installed or the server is
unreachable.
"""
from __future__ import annotations

import os

import pytest

POSTGRES_DSN = os.environ.get(
    "AGENT_MEMORY_POSTGRES_DSN",
    "postgresql://localhost/agent_memory_test",
)

try:
    import psycopg2

    conn = psycopg2.connect(POSTGRES_DSN)
    conn.close()
    POSTGRES_AVAILABLE = True
except Exception:
    POSTGRES_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not POSTGRES_AVAILABLE, reason="postgres not available"
)

TABLE = "test_memories"


@pytest.fixture()
def store():
    from agent_memory.postgres_store import PostgresMemoryStore

    s = PostgresMemoryStore(dsn=POSTGRES_DSN, table_name=TABLE, enable_embeddings=False)
    yield s
    # Teardown: drop the test table
    conn = psycopg2.connect(POSTGRES_DSN)
    with conn.cursor() as cur:
        cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    conn.commit()
    conn.close()


# ---------------------------------------------------------------------------
# Basic CRUD
# ---------------------------------------------------------------------------


def test_store_and_get(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="hello postgres", response="it works")
    stored = store.store(entry)
    fetched = store.get(stored.id)
    assert fetched is not None
    assert fetched.query == "hello postgres"


def test_count(store):
    from agent_memory.models import MemoryEntry

    assert store.count == 0
    store.store(MemoryEntry(query="a", response="b"))
    assert store.count == 1


def test_update(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="original", response="old")
    store.store(entry)
    entry.response = "new response"
    store.update(entry)
    assert store.get(entry.id).response == "new response"


def test_delete(store):
    from agent_memory.models import MemoryEntry

    entry = MemoryEntry(query="delete me", response="gone")
    store.store(entry)
    assert store.delete(entry.id) is True
    assert store.get(entry.id) is None
    assert store.delete(entry.id) is False


# ---------------------------------------------------------------------------
# Keyword search
# ---------------------------------------------------------------------------


def test_keyword_search(store):
    from agent_memory.models import MemoryEntry

    store.store(MemoryEntry(query="machine learning algorithms", response="gradient descent"))
    store.store(MemoryEntry(query="cooking recipes pasta", response="boil water"))

    results = store.keyword_search("machine learning", top_k=5)
    assert len(results) > 0
    assert "machine" in results[0][0].query.lower() or "learning" in results[0][0].query.lower()


def test_keyword_search_empty(store):
    assert store.keyword_search("nothing here") == []


# ---------------------------------------------------------------------------
# Scope filtering
# ---------------------------------------------------------------------------


def test_scope_filter(store):
    from agent_memory.models import MemoryEntry, MemoryScope

    store.store(MemoryEntry(query="user", response="u", scope=MemoryScope.USER))
    store.store(MemoryEntry(query="global", response="g", scope=MemoryScope.GLOBAL))

    user_entries = store.list_all(scopes=[MemoryScope.USER])
    assert len(user_entries) == 1
    assert user_entries[0].scope == MemoryScope.USER


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


def test_stats(store):
    from agent_memory.models import MemoryEntry, MemoryType

    store.store(MemoryEntry(query="q1", response="r1", type=MemoryType.FACT))
    store.store(MemoryEntry(query="q2", response="r2", type=MemoryType.CODE))
    s = store.stats()
    assert s["total"] == 2
    assert "fact" in s["by_type"]
    assert "code" in s["by_type"]


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------


def test_cleanup_expired(store):
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired()
    assert result["expired"] >= 1


def test_cleanup_delete(store):
    from datetime import datetime, timezone

    from agent_memory.models import MemoryEntry

    past = datetime(2000, 1, 1, tzinfo=timezone.utc)
    e = MemoryEntry(query="old", response="r", expires_at=past)
    e.refresh_state()
    store.store(e)
    result = store.cleanup_expired(delete=True)
    assert result["deleted"] >= 1
    assert store.count == 0

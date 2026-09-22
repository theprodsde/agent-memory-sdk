from __future__ import annotations

import json
from datetime import datetime
from typing import Any

from agent_memory.models import MemoryEntry, MemoryScope, MemoryState, MemoryType
from agent_memory.store import MemoryStore, bm25_scores


class RedisMemoryStore(MemoryStore):
    """Redis-backed persistent memory store.

    Requires: pip install agent-memory-sdk[redis]

    Storage layout (all keys prefixed with ``key_prefix``):
      - ``{prefix}:entry:{id}``  — JSON-encoded entry
      - ``{prefix}:ids``          — sorted set of all IDs scored by created_at timestamp
      - ``{prefix}:scope:{s}``   — set of IDs for each scope value
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6379,
        db: int = 0,
        password: str | None = None,
        url: str | None = None,
        key_prefix: str = "agent_memory",
        redis_client: Any | None = None,
    ) -> None:
        if redis_client is not None:
            self._client = redis_client
        else:
            try:
                import redis
            except ImportError:
                raise ImportError(
                    "Redis backend requires redis-py. "
                    "Install with: pip install agent-memory-sdk[redis]"
                ) from None
            if url:
                self._client = redis.from_url(url, decode_responses=True)
            else:
                self._client = redis.Redis(
                    host=host, port=port, db=db, password=password, decode_responses=True
                )
        self._prefix = key_prefix
        self._client.ping()

    def _key(self, memory_id: str) -> str:
        return f"{self._prefix}:entry:{memory_id}"

    def _ids_key(self) -> str:
        return f"{self._prefix}:ids"

    def _scope_key(self, scope: str) -> str:
        return f"{self._prefix}:scope:{scope}"

    @property
    def count(self) -> int:
        return int(self._client.zcard(self._ids_key()))

    def store(self, entry: MemoryEntry) -> MemoryEntry:
        entry.refresh_state()
        data = self._entry_to_dict(entry)
        pipeline = self._client.pipeline()
        pipeline.set(self._key(entry.id), json.dumps(data))
        pipeline.zadd(self._ids_key(), {entry.id: entry.created_at.timestamp()})
        pipeline.sadd(self._scope_key(entry.scope.value), entry.id)
        pipeline.execute()
        return entry

    def get(self, memory_id: str) -> MemoryEntry | None:
        raw = self._client.get(self._key(memory_id))
        if not raw:
            return None
        return self._dict_to_entry(json.loads(raw))

    def update(self, entry: MemoryEntry) -> MemoryEntry:
        existing = self.get(entry.id)
        if existing and existing.scope != entry.scope:
            # Remove from old scope set if scope changed
            self._client.srem(self._scope_key(existing.scope.value), entry.id)
        return self.store(entry)

    def delete(self, memory_id: str) -> bool:
        existing = self.get(memory_id)
        if not existing:
            return False
        pipeline = self._client.pipeline()
        pipeline.delete(self._key(memory_id))
        pipeline.zrem(self._ids_key(), memory_id)
        pipeline.srem(self._scope_key(existing.scope.value), memory_id)
        pipeline.execute()
        return True

    def list_all(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
        memory_type: MemoryType | None = None,
    ) -> list[MemoryEntry]:
        if scopes:
            raw_ids: set[str] = set()
            for scope in scopes:
                members = self._client.smembers(self._scope_key(scope.value))
                raw_ids.update(members)
        else:
            raw_ids = set(self._client.zrange(self._ids_key(), 0, -1))

        entries: list[MemoryEntry] = []
        for mid in raw_ids:
            entry = self.get(mid)
            if entry is None:
                continue
            if not include_archived and entry.archived:
                continue
            if not include_expired and entry.is_expired:
                continue
            if memory_type and entry.type != memory_type:
                continue
            entries.append(entry)

        entries.sort(key=lambda e: e.updated_at, reverse=True)
        return entries[offset : offset + limit]

    def search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        # Redis has no native vector index here; falls back to keyword search.
        # Swap in a real embedder + Redis VSS for true semantic search.
        return self.keyword_search(
            query,
            top_k=top_k,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        *,
        scopes: list[MemoryScope] | None = None,
        include_archived: bool = False,
        include_expired: bool = False,
    ) -> list[tuple[MemoryEntry, float]]:
        entries = self.list_all(
            limit=10_000,
            scopes=scopes,
            include_archived=include_archived,
            include_expired=include_expired,
        )
        if not entries:
            return []
        documents = [f"{e.query}\n{e.content}\n{' '.join(e.tags)}" for e in entries]
        return bm25_scores(query, entries, documents, top_k)

    def stats(self) -> dict[str, Any]:
        entries = self.list_all(limit=1_000_000, include_archived=True, include_expired=True)
        by_state: dict[str, int] = {}
        by_type: dict[str, int] = {}
        total_access = 0
        for entry in entries:
            entry.refresh_state()
            by_state[entry.state.value] = by_state.get(entry.state.value, 0) + 1
            by_type[entry.type.value] = by_type.get(entry.type.value, 0) + 1
            total_access += entry.access_count
        return {
            "total": len(entries),
            "by_state": by_state,
            "by_type": by_type,
            "total_access_count": total_access,
        }

    @staticmethod
    def _entry_to_dict(entry: MemoryEntry) -> dict[str, Any]:
        return {
            "id": entry.id,
            "query": entry.query,
            "response": entry.response,
            "content": entry.content,
            "type": entry.type.value,
            "scope": entry.scope.value,
            "metadata": entry.metadata,
            "tags": entry.tags,
            "confidence": entry.confidence,
            "requires_verification": entry.requires_verification,
            "archived": entry.archived,
            "state": entry.state.value,
            "access_count": entry.access_count,
            "created_at": entry.created_at.isoformat(),
            "updated_at": entry.updated_at.isoformat(),
            "last_accessed_at": (
                entry.last_accessed_at.isoformat() if entry.last_accessed_at else None
            ),
            "expires_at": entry.expires_at.isoformat() if entry.expires_at else None,
        }

    @staticmethod
    def _dict_to_entry(data: dict[str, Any]) -> MemoryEntry:
        last_accessed = data.get("last_accessed_at")
        expires_at = data.get("expires_at")
        return MemoryEntry(
            id=data["id"],
            query=data["query"],
            response=data["response"],
            content=data.get("content", data["response"]),
            type=MemoryType(data.get("type", "conversation")),
            scope=MemoryScope(data.get("scope", "user")),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
            confidence=float(data.get("confidence", 1.0)),
            requires_verification=bool(data.get("requires_verification", False)),
            archived=bool(data.get("archived", False)),
            state=MemoryState(data.get("state", "active")),
            access_count=int(data.get("access_count", 0)),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            last_accessed_at=datetime.fromisoformat(last_accessed) if last_accessed else None,
            expires_at=datetime.fromisoformat(expires_at) if expires_at else None,
        )

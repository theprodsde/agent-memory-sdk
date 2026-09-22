from __future__ import annotations

import asyncio
import builtins
from pathlib import Path

from agent_memory.decision import DecisionEngine
from agent_memory.models import MemoryDecision, MemoryEntry, MemoryScope, MemoryType
from agent_memory.policy import DecisionPolicy, DefaultPolicy
from agent_memory.retriever import MemoryRetriever
from agent_memory.sqlite_store import SqliteMemoryStore
from agent_memory.store import ChromaDBStore, MemoryStore
from agent_memory.ttl import parse_ttl


class Memory:
    """High-level SDK for persistent agent memory with a decision layer."""

    def __init__(
        self,
        persist_dir: str | Path = ".agent_memory",
        collection_name: str = "agent_memories",
        policy: DecisionPolicy | None = None,
        replay_threshold: float = 0.85,
        restore_threshold: float = 0.70,
        verify_threshold: float = 0.80,
        backend: str = "sqlite",  # "chromadb" | "sqlite" | "redis" | "postgres"
        embedder: object | None = None,
        enable_embeddings: bool | str = "auto",
        store: MemoryStore | None = None,
        **backend_kwargs: object,
    ) -> None:
        if store is not None:
            # Accept a pre-built store directly (useful for testing / custom backends)
            self.store: MemoryStore = store
        elif backend == "sqlite":
            self.store = SqliteMemoryStore(
                persist_dir=persist_dir,
                collection_name=collection_name,
                embedder=embedder,  # type: ignore[arg-type]
                enable_embeddings=enable_embeddings,
            )
        elif backend == "chromadb":
            self.store = ChromaDBStore(persist_dir=persist_dir, collection_name=collection_name)
        elif backend == "redis":
            from agent_memory.redis_store import RedisMemoryStore

            self.store = RedisMemoryStore(**backend_kwargs)  # type: ignore[arg-type]
        elif backend == "postgres":
            from agent_memory.postgres_store import PostgresMemoryStore

            self.store = PostgresMemoryStore(**backend_kwargs)  # type: ignore[arg-type]
        else:
            raise ValueError(
                f"Unknown backend: {backend!r}. "
                "Choices: 'sqlite', 'chromadb', 'redis', 'postgres'"
            )
        self._policy = policy or DefaultPolicy()
        self.retriever = MemoryRetriever(self.store, policy=self._policy)
        self.decision_engine = DecisionEngine(
            self.retriever,
            policy=self._policy,
            replay_threshold=replay_threshold,
            restore_threshold=restore_threshold,
            verify_threshold=verify_threshold,
        )

    def remember(
        self,
        query: str,
        response: str,
        *,
        content: str | None = None,
        type: MemoryType | str = MemoryType.CONVERSATION,
        scope: MemoryScope | str = MemoryScope.USER,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        requires_verification: bool = False,
        ttl: str | int | float | None = None,
    ) -> MemoryEntry:
        memory_type = MemoryType(type) if isinstance(type, str) else type
        memory_scope = MemoryScope(scope) if isinstance(scope, str) else scope
        entry = MemoryEntry(
            query=query,
            response=response,
            content=content or response,
            type=memory_type,
            scope=memory_scope,
            metadata=metadata or {},
            tags=tags or [],
            confidence=confidence,
            requires_verification=requires_verification,
            expires_at=parse_ttl(ttl),
        )
        entry.refresh_state()
        return self.store.store(entry)

    async def aremember(
        self,
        query: str,
        response: str,
        *,
        content: str | None = None,
        type: MemoryType | str = MemoryType.CONVERSATION,
        scope: MemoryScope | str = MemoryScope.USER,
        metadata: dict | None = None,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        requires_verification: bool = False,
        ttl: str | int | float | None = None,
    ) -> MemoryEntry:
        """Async version of remember()."""
        return await asyncio.to_thread(
            self.remember,
            query,
            response,
            content=content,
            type=type,
            scope=scope,
            metadata=metadata,
            tags=tags,
            confidence=confidence,
            requires_verification=requires_verification,
            ttl=ttl,
        )

    def resolve(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        scope: list[MemoryScope | str] | None = None,
        enable_verify: bool = True,
    ) -> MemoryDecision:
        scopes = None
        if scope:
            scopes = [MemoryScope(s) if isinstance(s, str) else s for s in scope]
        return self.decision_engine.decide(
            query,
            mode=mode,
            top_k=top_k,
            scopes=scopes,
            enable_verify=enable_verify,
        )

    async def aresolve(
        self,
        query: str,
        *,
        mode: str = "auto",
        top_k: int = 3,
        scope: list[MemoryScope | str] | None = None,
        enable_verify: bool = True,
    ) -> MemoryDecision:
        """Async version of resolve()."""
        return await asyncio.to_thread(
            self.resolve,
            query,
            mode=mode,
            top_k=top_k,
            scope=scope,
            enable_verify=enable_verify,
        )

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scope: list[MemoryScope | str] | None = None,
        include_archived: bool = False,
        type: MemoryType | str | None = None,
    ) -> list[MemoryEntry]:
        scopes = [MemoryScope(s) if isinstance(s, str) else s for s in scope] if scope else None
        memory_type = MemoryType(type) if isinstance(type, str) else type
        return self.store.list_all(
            limit=limit,
            offset=offset,
            scopes=scopes,
            include_archived=include_archived,
            memory_type=memory_type,
        )

    async def alist(
        self,
        limit: int = 100,
        offset: int = 0,
        *,
        scope: builtins.list[MemoryScope | str] | None = None,
        include_archived: bool = False,
        type: MemoryType | str | None = None,
    ) -> builtins.list[MemoryEntry]:
        """Async version of list()."""
        return await asyncio.to_thread(
            self.list,
            limit=limit,
            offset=offset,
            scope=scope,
            include_archived=include_archived,
            type=type,
        )

    def get(self, memory_id: str) -> MemoryEntry | None:
        return self.store.get(memory_id)

    async def aget(self, memory_id: str) -> MemoryEntry | None:
        """Async version of get()."""
        return await asyncio.to_thread(self.get, memory_id)

    def forget(self, memory_id: str) -> bool:
        return self.store.delete(memory_id)

    async def aforget(self, memory_id: str) -> bool:
        """Async version of forget()."""
        return await asyncio.to_thread(self.forget, memory_id)

    def archive(self, memory_id: str) -> MemoryEntry | None:
        entry = self.store.get(memory_id)
        if not entry:
            return None
        entry.archived = True
        entry.refresh_state()
        return self.store.update(entry)

    async def aarchive(self, memory_id: str) -> MemoryEntry | None:
        """Async version of archive()."""
        return await asyncio.to_thread(self.archive, memory_id)

    def cleanup(self, *, delete: bool = False) -> dict[str, int]:
        """
        Mark expired memories as expired, optionally deleting them.

        Returns counts: {"expired": N, "deleted": M}
        """
        return self.store.cleanup_expired(delete=delete)

    async def acleanup(self, *, delete: bool = False) -> dict[str, int]:
        """Async version of cleanup()."""
        return await asyncio.to_thread(self.cleanup, delete=delete)

    def stats(self) -> dict:
        """Return aggregate memory and usage statistics."""
        return self.store.stats()

    async def astats(self) -> dict:
        """Async version of stats()."""
        return await asyncio.to_thread(self.stats)

    def consolidate(self, similarity_threshold: float = 0.95) -> builtins.list[MemoryEntry]:
        """
        Merge near-duplicate active memories into summary entries.
        Returns newly created summary memories.
        """
        entries = self.store.list_all(limit=10_000)
        created: list[MemoryEntry] = []
        seen: set[str] = set()

        by_id = {entry.id: entry for entry in entries}
        for entry in entries:
            if entry.id in seen or entry.archived or entry.type == MemoryType.SUMMARY:
                continue

            # One search per entry (not per pair): near-duplicates of this
            # entry's query are its top search hits by construction.
            duplicates = [entry]
            hits = self.store.search(entry.query, top_k=10)
            for hit, score in hits:
                other = by_id.get(hit.id)
                if other is None or other.id == entry.id or other.id in seen:
                    continue
                if other.archived or other.type != entry.type or other.scope != entry.scope:
                    continue
                if score >= similarity_threshold:
                    duplicates.append(other)

            if len(duplicates) < 2:
                continue

            for dup in duplicates:
                seen.add(dup.id)
                dup.archived = True
                self.store.update(dup)

            merged_query = duplicates[0].query
            merged_content = "\n".join(f"- {d.response}" for d in duplicates)
            summary = self.remember(
                query=merged_query,
                response=merged_content,
                content=merged_content,
                type=MemoryType.SUMMARY,
                scope=duplicates[0].scope,
                tags=list({tag for d in duplicates for tag in d.tags}),
                metadata={"consolidated_from": [d.id for d in duplicates]},
            )
            created.append(summary)

        return created

    async def aconsolidate(self, similarity_threshold: float = 0.95) -> builtins.list[MemoryEntry]:
        """Async version of consolidate()."""
        return await asyncio.to_thread(self.consolidate, similarity_threshold=similarity_threshold)

    def format_restore_context(self, decision: MemoryDecision) -> str:
        if not decision.context:
            return ""

        lines = ["## Retrieved Memory Context", ""]
        for result in decision.context:
            entry = result.entry
            lines.append(
                f"### Memory {result.rank} (score: {result.final_score:.2f}, type: {entry.type.value})"
            )
            lines.append(f"**Prior query:** {entry.query}")
            lines.append(f"**Prior response:** {entry.response}")
            if entry.tags:
                lines.append(f"**Tags:** {', '.join(entry.tags)}")
            lines.append("")
        return "\n".join(lines)

    def format_verify_context(self, decision: MemoryDecision) -> str:
        if not decision.memory:
            return ""
        entry = decision.memory
        return (
            "## Memory Pending Verification\n\n"
            f"**Query:** {entry.query}\n"
            f"**Stored response:** {entry.response}\n"
            f"**Type:** {entry.type.value}\n"
            f"**Last updated:** {entry.updated_at.isoformat()}\n\n"
            "Validate this memory with available tools before reusing or regenerating."
        )

    # Backward compatibility
    def list_memories(self, limit: int = 100, offset: int = 0) -> builtins.list[MemoryEntry]:
        return self.list(limit=limit, offset=offset)


MemoryManager = Memory

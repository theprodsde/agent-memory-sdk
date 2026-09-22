"""Paged memory — hierarchical context management inspired by MemGPT/Letta.

Three tiers that mirror how human working memory works:

  1. **In-context buffer** — fixed-size deque of the most recent entries.
     Always returned in full; no search needed.  When full, oldest entries
     are paged out to the recall tier automatically.

  2. **Recall storage** — the standard Memory store.  Searched semantically
     and returned as additional context alongside the in-context buffer.

  3. **Archival storage** — archived entries in the same store.  Searched
     only when explicitly requested; keeps long-term facts out of the
     hot path without deleting them.

Usage::

    from agent_memory import Memory
    from agent_memory.paged_memory import PagedMemory

    base_memory = Memory(persist_dir=".agent_memory")
    paged = PagedMemory(base_memory, context_size=20)

    # Add new turns — old entries page out to recall automatically
    paged.add_turn("What is Python?", "A programming language.")
    paged.add_turn("Favourite framework?", "FastAPI.")

    # Get combined context for any query
    ctx = paged.get_context("Tell me about Python")
    print(ctx.format_for_llm())

    # Explicitly search archival (old, archived memories)
    old_entries = paged.search_archive("Python version history")
"""
from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agent_memory.models import MemoryEntry, MemoryScope, MemoryType

if TYPE_CHECKING:
    from agent_memory.manager import Memory


# ---------------------------------------------------------------------------
# PagedContext — the combined result of a get_context() call
# ---------------------------------------------------------------------------


@dataclass
class PagedContext:
    """Combined context from all three memory tiers."""

    in_context: list[MemoryEntry] = field(default_factory=list)
    """Entries from the active in-context buffer (always included)."""

    recalled: list[MemoryEntry] = field(default_factory=list)
    """Entries retrieved from recall storage via semantic search."""

    archived: list[MemoryEntry] = field(default_factory=list)
    """Entries retrieved from archival storage (only when explicitly searched)."""

    @property
    def all_entries(self) -> list[MemoryEntry]:
        """Deduplicated union of all tiers, in-context first."""
        seen: set[str] = set()
        out: list[MemoryEntry] = []
        for e in self.in_context + self.recalled + self.archived:
            if e.id not in seen:
                seen.add(e.id)
                out.append(e)
        return out

    def format_for_llm(self) -> str:
        """Render context as Markdown blocks suitable for an LLM system prompt."""
        lines: list[str] = []

        if self.in_context:
            lines.append("## Recent Context (in-context buffer)\n")
            for e in self.in_context:
                lines.append(f"**Q:** {e.query}")
                lines.append(f"**A:** {e.response}\n")

        if self.recalled:
            lines.append("## Recalled Context (semantic search)\n")
            for e in self.recalled:
                lines.append(f"**Q:** {e.query}")
                lines.append(f"**A:** {e.response}  *(type: {e.type.value})*\n")

        if self.archived:
            lines.append("## Archival Context\n")
            for e in self.archived:
                lines.append(f"**Q:** {e.query}")
                lines.append(f"**A:** {e.response}\n")

        return "\n".join(lines).strip()

    def __len__(self) -> int:
        return len(self.all_entries)


# ---------------------------------------------------------------------------
# PagedMemory
# ---------------------------------------------------------------------------


class PagedMemory:
    """Hierarchical memory wrapper that manages three context tiers.

    Parameters
    ----------
    memory:
        An initialised :class:`~agent_memory.manager.Memory` instance that
        acts as the recall + archival storage backend.
    context_size:
        Maximum number of entries in the in-context buffer before older
        entries are paged out to recall storage.
    recall_top_k:
        Number of entries fetched from recall storage in ``get_context()``.
    scope:
        Default scope used when persisting paged-out entries.
    """

    def __init__(
        self,
        memory: Memory,
        *,
        context_size: int = 20,
        recall_top_k: int = 5,
        scope: MemoryScope = MemoryScope.USER,
    ) -> None:
        self._memory = memory
        self.context_size = context_size
        self.recall_top_k = recall_top_k
        self.scope = scope
        # The in-context buffer — ephemeral, lives only in this process
        self._buffer: deque[MemoryEntry] = deque(maxlen=context_size)

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def add_turn(
        self,
        query: str,
        response: str,
        *,
        type: MemoryType | str = MemoryType.CONVERSATION,
        tags: list[str] | None = None,
        confidence: float = 1.0,
        requires_verification: bool = False,
    ) -> MemoryEntry:
        """Add a conversation turn to the in-context buffer.

        When the buffer reaches *context_size*, the oldest entry is
        automatically paged out to recall storage.
        """
        memory_type = MemoryType(type) if isinstance(type, str) else type
        entry = MemoryEntry(
            query=query,
            response=response,
            type=memory_type,
            scope=self.scope,
            tags=tags or [],
            confidence=confidence,
            requires_verification=requires_verification,
        )

        # If the buffer is full, the deque will drop the oldest automatically
        # (maxlen is set). Before that, persist it to recall.
        if len(self._buffer) == self.context_size:
            self._page_out_oldest()

        self._buffer.append(entry)
        return entry

    async def aadd_turn(self, query: str, response: str, **kwargs: Any) -> MemoryEntry:
        """Async version of :meth:`add_turn`."""
        return await asyncio.to_thread(self.add_turn, query, response, **kwargs)

    def add_entry(self, entry: MemoryEntry) -> None:
        """Add a pre-built :class:`~agent_memory.models.MemoryEntry` directly."""
        if len(self._buffer) == self.context_size:
            self._page_out_oldest()
        self._buffer.append(entry)

    # ------------------------------------------------------------------
    # Reading
    # ------------------------------------------------------------------

    def get_context(
        self,
        query: str,
        *,
        recall_top_k: int | None = None,
        include_archived: bool = False,
    ) -> PagedContext:
        """Return combined context from all tiers for *query*.

        * **In-context buffer** — returned in full (most recent first).
        * **Recall** — top-K entries from the memory store matching *query*.
        * **Archived** — only included when *include_archived* is True.
        """
        top_k = recall_top_k if recall_top_k is not None else self.recall_top_k

        # In-context buffer (newest first)
        in_ctx = list(reversed(self._buffer))

        # IDs already in the buffer (avoid duplicates in recall)
        buffer_ids = {e.id for e in self._buffer}

        # Recall from the store
        raw_recall = self._memory.store.keyword_search(query, top_k=top_k * 2)
        recalled = [
            e for e, _ in raw_recall
            if e.id not in buffer_ids and not e.archived
        ][:top_k]

        # Archival search
        archived: list[MemoryEntry] = []
        if include_archived:
            raw_arch = self._memory.store.keyword_search(
                query, top_k=top_k, include_archived=True
            )
            archived = [
                e for e, _ in raw_arch
                if e.archived and e.id not in buffer_ids
            ][:top_k]

        return PagedContext(
            in_context=in_ctx,
            recalled=recalled,
            archived=archived,
        )

    async def aget_context(self, query: str, **kwargs: Any) -> PagedContext:
        """Async version of :meth:`get_context`."""
        return await asyncio.to_thread(self.get_context, query, **kwargs)

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search only recall storage (does not include in-context buffer)."""
        buffer_ids = {e.id for e in self._buffer}
        raw = self._memory.store.keyword_search(query, top_k=top_k * 2)
        return [e for e, _ in raw if e.id not in buffer_ids and not e.archived][:top_k]

    def search_archive(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Search archival storage (archived entries only)."""
        raw = self._memory.store.keyword_search(
            query, top_k=top_k * 2, include_archived=True
        )
        return [e for e, _ in raw if e.archived][:top_k]

    # ------------------------------------------------------------------
    # Buffer management
    # ------------------------------------------------------------------

    def flush_to_recall(self) -> int:
        """Persist the entire in-context buffer to recall storage.

        Returns the number of entries persisted.
        """
        count = 0
        while self._buffer:
            entry = self._buffer.popleft()
            self._memory.store.store(entry)
            count += 1
        return count

    def clear_context(self) -> None:
        """Clear the in-context buffer without persisting (session end)."""
        self._buffer.clear()

    @property
    def context_entries(self) -> list[MemoryEntry]:
        """Current in-context buffer contents (newest first)."""
        return list(reversed(self._buffer))

    @property
    def context_count(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _page_out_oldest(self) -> None:
        """Persist the oldest buffer entry to the recall store."""
        if self._buffer:
            oldest = self._buffer[0]  # leftmost = oldest
            self._memory.store.store(oldest)

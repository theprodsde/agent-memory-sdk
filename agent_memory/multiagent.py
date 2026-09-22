from __future__ import annotations

import asyncio
import builtins
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from agent_memory.models import MemoryDecision, MemoryEntry, MemoryScope

if TYPE_CHECKING:
    from agent_memory.manager import Memory


class IsolationMode(str, Enum):
    """Controls how memories are shared between agents on the same store.

    * ``SHARED``     — every agent can read and search all memories.
    * ``NAMESPACED`` — agents see their own memories plus any marked
                       ``global=True`` (or memories with no ``agent_id``).
    * ``ISOLATED``   — agents see only the memories they created.
    """

    SHARED = "shared"
    NAMESPACED = "namespaced"
    ISOLATED = "isolated"


@dataclass
class AgentInfo:
    agent_id: str
    name: str | None = None
    scope: MemoryScope = MemoryScope.USER


class MultiAgentMemory:
    """Agent-scoped view over a shared :class:`~agent_memory.manager.Memory` store.

    Each agent has an ``agent_id`` that is stamped into
    ``entry.metadata["agent_id"]`` when the agent calls :meth:`remember`.
    The :attr:`isolation` mode controls which stored entries the agent can
    read back via :meth:`list`.

    Usage::

        shared_mem = Memory(persist_dir=".agent_memory")

        agent_a = MultiAgentMemory(shared_mem, agent_id="agent-a")
        agent_b = MultiAgentMemory(shared_mem, agent_id="agent-b")

        agent_a.remember("What is 2+2?", "4")
        agent_a.broadcast("company name", "Acme Inc.", type="fact")

        # agent_b sees only its own + global memories (NAMESPACED default)
        entries = agent_b.list()
    """

    def __init__(
        self,
        memory: Memory,
        agent_id: str,
        *,
        isolation: IsolationMode = IsolationMode.NAMESPACED,
    ) -> None:
        self._memory: Memory = memory
        self.agent_id = agent_id
        self.isolation = isolation

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        agent_id: str,
        *,
        isolation: IsolationMode = IsolationMode.NAMESPACED,
        **memory_kwargs: Any,
    ) -> MultiAgentMemory:
        """Create a new :class:`~agent_memory.manager.Memory` and wrap it."""
        from agent_memory.manager import Memory

        return cls(Memory(**memory_kwargs), agent_id=agent_id, isolation=isolation)

    # ------------------------------------------------------------------
    # Write operations
    # ------------------------------------------------------------------

    def remember(
        self,
        query: str,
        response: str,
        *,
        global_memory: bool = False,
        **kwargs: Any,
    ) -> MemoryEntry:
        """Store a memory tagged with this agent's ID.

        Set ``global_memory=True`` to make the entry visible to all agents
        (regardless of isolation mode).
        """
        metadata: dict[str, Any] = dict(kwargs.pop("metadata", None) or {})
        metadata["agent_id"] = self.agent_id
        if global_memory:
            metadata["global"] = True
        return self._memory.remember(query, response, metadata=metadata, **kwargs)

    async def aremember(
        self,
        query: str,
        response: str,
        *,
        global_memory: bool = False,
        **kwargs: Any,
    ) -> MemoryEntry:
        return await asyncio.to_thread(
            self.remember, query, response, global_memory=global_memory, **kwargs
        )

    def broadcast(
        self,
        query: str,
        response: str,
        **kwargs: Any,
    ) -> MemoryEntry:
        """Store a memory that is visible to ALL agents (shortcut for ``global_memory=True``)."""
        return self.remember(query, response, global_memory=True, **kwargs)

    # ------------------------------------------------------------------
    # Read operations
    # ------------------------------------------------------------------

    def resolve(self, query: str, **kwargs: Any) -> MemoryDecision:
        return self._memory.resolve(query, **kwargs)

    async def aresolve(self, query: str, **kwargs: Any) -> MemoryDecision:
        return await asyncio.to_thread(self.resolve, query, **kwargs)

    def list(
        self,
        limit: int = 100,
        offset: int = 0,
        **kwargs: Any,
    ) -> builtins.list[MemoryEntry]:
        """List memories visible to this agent under the current isolation mode."""
        all_entries = self._memory.list(limit=10_000, **kwargs)
        visible = self._filter_by_isolation(all_entries)
        return visible[offset : offset + limit]

    def get(self, memory_id: str) -> MemoryEntry | None:
        entry = self._memory.get(memory_id)
        if entry is None:
            return None
        visible: builtins.list[MemoryEntry] = self._filter_by_isolation([entry])
        return visible[0] if visible else None

    def forget(self, memory_id: str) -> bool:
        """Delete a memory if owned by this agent."""
        entry = self._memory.get(memory_id)
        if not entry:
            return False
        if (
            self.isolation != IsolationMode.SHARED
            and entry.metadata.get("agent_id") != self.agent_id
        ):
            return False
        return self._memory.forget(memory_id)

    # ------------------------------------------------------------------
    # Coordination helpers
    # ------------------------------------------------------------------

    def transfer_memory(
        self, memory_id: str, target_agent_id: str
    ) -> MemoryEntry | None:
        """Transfer ownership of a memory to *target_agent_id*.

        Only the owning agent can transfer.  Returns the updated entry or
        ``None`` if the memory does not exist or is not owned by this agent.
        """
        entry = self._memory.get(memory_id)
        if not entry:
            return None
        if entry.metadata.get("agent_id") != self.agent_id:
            return None
        entry.metadata["agent_id"] = target_agent_id
        entry.metadata["transferred_from"] = self.agent_id
        return self._memory.store.update(entry)

    def agents_with_memories(self) -> builtins.list[str]:
        """Return sorted list of agent IDs that have stored memories."""
        entries = self._memory.list(limit=10_000)
        return sorted(
            {
                str(e.metadata.get("agent_id"))
                for e in entries
                if e.metadata.get("agent_id")
            }
        )

    def stats(self) -> dict[str, Any]:
        """Store-level stats (backend-dependent aggregates)."""
        return self._memory.stats()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_by_isolation(
        self, entries: builtins.list[MemoryEntry]
    ) -> builtins.list[MemoryEntry]:
        if self.isolation == IsolationMode.SHARED:
            return entries

        result: builtins.list[MemoryEntry] = []
        for entry in entries:
            agent_id = entry.metadata.get("agent_id")
            is_global = bool(entry.metadata.get("global", False))

            if self.isolation == IsolationMode.ISOLATED:
                if agent_id == self.agent_id:
                    result.append(entry)
            else:  # NAMESPACED
                if agent_id == self.agent_id or is_global or agent_id is None:
                    result.append(entry)
        return result

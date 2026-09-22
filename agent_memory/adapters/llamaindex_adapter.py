"""LlamaIndex memory adapter for agent-memory-sdk.

Requires: ``pip install agent-memory-sdk[llamaindex]``

Usage::

    from agent_memory import Memory
    from agent_memory.adapters.llamaindex_adapter import AgentMemoryLlamaIndex

    memory = Memory(persist_dir=".agent_memory")
    li_memory = AgentMemoryLlamaIndex(memory=memory)

    # Attach to a LlamaIndex chat engine or agent:
    agent = OpenAIAgent.from_tools(tools, memory=li_memory)
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_memory.manager import Memory

try:
    from llama_index.core.llms import ChatMessage, MessageRole
    from llama_index.core.memory import BaseMemory as LlamaBaseMemory

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        from llama_index.llms import ChatMessage, MessageRole  # noqa: F401
        from llama_index.memory import BaseMemory as LlamaBaseMemory  # noqa: F401

        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        LLAMAINDEX_AVAILABLE = False
        LlamaBaseMemory = object  # type: ignore[misc,assignment]
        ChatMessage = object  # type: ignore[misc,assignment]
        MessageRole = object  # type: ignore[misc,assignment]


class AgentMemoryLlamaIndex(LlamaBaseMemory):  # type: ignore[misc]
    """LlamaIndex ``BaseMemory`` backed by agent-memory-sdk.

    Integrates with any LlamaIndex chat engine or agent that accepts a
    ``memory=`` argument, providing semantic retrieval over past interactions
    rather than simple sliding-window buffering.

    Parameters
    ----------
    memory:
        An initialised :class:`~agent_memory.manager.Memory` instance.
    token_limit:
        Approximate token budget for returned context (default 3000).
        Context is trimmed by cutting the oldest turns first.
    top_k:
        How many memories to retrieve per query.
    memory_type:
        Memory type applied to stored messages (default ``"conversation"``).
    scope:
        Memory scope applied to stored messages (default ``"user"``).
    """

    def __init__(
        self,
        memory: Memory,
        *,
        token_limit: int = 3000,
        top_k: int = 5,
        memory_type: str = "conversation",
        scope: str = "user",
    ) -> None:
        if not LLAMAINDEX_AVAILABLE:
            raise ImportError(
                "LlamaIndex adapter requires llama-index-core. "
                "Install with: pip install agent-memory-sdk[llamaindex]"
            )
        self._mem = memory
        self._token_limit = token_limit
        self._top_k = top_k
        self._memory_type = memory_type
        self._scope = scope
        # In-session buffer for messages not yet persisted
        self._buffer: list[Any] = []

    # ------------------------------------------------------------------
    # BaseMemory interface (llama-index-core ≥ 0.10)
    # ------------------------------------------------------------------

    @classmethod
    def from_defaults(
        cls,
        persist_dir: str = ".agent_memory",
        **kwargs: Any,
    ) -> AgentMemoryLlamaIndex:
        """Create a LlamaIndex memory backed by a fresh Memory at *persist_dir*."""
        from agent_memory.manager import Memory

        return cls(Memory(persist_dir=persist_dir), **kwargs)

    def get(
        self,
        input: str | None = None,
        initial_token_count: int = 0,
        **kwargs: Any,
    ) -> list[Any]:
        """Return relevant messages for *input* as a list of ChatMessage objects."""
        query = input or ""
        if not query:
            return list(self._buffer)

        decision = self._mem.resolve(query, top_k=self._top_k)
        messages: list[Any] = []

        if decision.memory:
            messages.append(
                ChatMessage(role=MessageRole.USER, content=decision.memory.query)
            )
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=decision.memory.response)
            )
        for ctx in decision.context:
            messages.append(
                ChatMessage(role=MessageRole.USER, content=ctx.entry.query)
            )
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=ctx.entry.response)
            )

        # Append current in-session buffer
        messages.extend(self._buffer)
        return self._trim_to_token_limit(messages, initial_token_count)

    def get_all(self) -> list[Any]:
        """Return all stored messages as ChatMessage objects."""
        entries = self._mem.list(limit=1_000)
        messages: list[Any] = []
        for entry in entries:
            messages.append(ChatMessage(role=MessageRole.USER, content=entry.query))
            messages.append(
                ChatMessage(role=MessageRole.ASSISTANT, content=entry.response)
            )
        return messages

    def put(self, message: Any) -> None:
        """Buffer a new message; persist when a USER+ASSISTANT pair is ready."""
        self._buffer.append(message)
        # Flush complete turn pairs
        self._flush_buffer()

    def set(self, messages: list[Any]) -> None:
        """Replace the in-session buffer."""
        self._buffer = list(messages)

    def reset(self) -> None:
        """Clear the in-session buffer (does not touch persistent memory)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _flush_buffer(self) -> None:
        """Persist complete USER→ASSISTANT pairs from the buffer."""
        i = 0
        flushed_up_to = 0
        while i < len(self._buffer) - 1:
            msg_a = self._buffer[i]
            msg_b = self._buffer[i + 1]
            role_a = getattr(msg_a, "role", None)
            role_b = getattr(msg_b, "role", None)
            is_pair = (
                str(role_a) in ("user", MessageRole.USER)
                and str(role_b) in ("assistant", MessageRole.ASSISTANT)
            )
            if is_pair:
                self._mem.remember(
                    query=str(msg_a.content),
                    response=str(msg_b.content),
                    type=self._memory_type,
                    scope=self._scope,
                )
                flushed_up_to = i + 2
            i += 1
        if flushed_up_to:
            self._buffer = self._buffer[flushed_up_to:]

    def _trim_to_token_limit(
        self, messages: list[Any], already_used: int = 0
    ) -> list[Any]:
        """Approximate token-budget trimming (4 chars ≈ 1 token)."""
        budget = self._token_limit - already_used
        result: list[Any] = []
        used = 0
        for msg in reversed(messages):
            tokens = len(str(getattr(msg, "content", ""))) // 4
            if used + tokens > budget:
                break
            result.insert(0, msg)
            used += tokens
        return result

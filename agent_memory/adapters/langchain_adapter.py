"""LangChain memory adapter for agent-memory-sdk.

Requires: ``pip install agent-memory-sdk[langchain]``

Usage::

    from agent_memory import Memory
    from agent_memory.adapters.langchain_adapter import AgentMemoryLangChain

    memory = Memory(persist_dir=".agent_memory")
    lc_memory = AgentMemoryLangChain(memory=memory)

    # Use with any LangChain chain that accepts a `memory=` argument:
    chain = ConversationChain(llm=llm, memory=lc_memory)

    # Or standalone:
    lc_memory.save_context({"input": "Hi"}, {"output": "Hello!"})
    vars = lc_memory.load_memory_variables({"input": "What did I say first?"})
    print(vars["history"])
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from agent_memory.manager import Memory

try:
    from langchain_core.memory import BaseMemory
    from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.memory import ConversationBufferMemory as _CBM  # noqa: F401
        from langchain.schema import AIMessage, BaseMessage, HumanMessage  # noqa: F401
        from langchain.schema.memory import BaseMemory  # noqa: F401

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False
        BaseMemory = object  # type: ignore[misc,assignment]
        BaseMessage = object  # type: ignore[misc,assignment]
        HumanMessage = object  # type: ignore[misc,assignment]
        AIMessage = object  # type: ignore[misc,assignment]


class AgentMemoryLangChain(BaseMemory):  # type: ignore[misc]
    """LangChain ``BaseMemory`` backed by agent-memory-sdk.

    Stores each conversation turn via :meth:`~agent_memory.manager.Memory.remember`
    and retrieves relevant context via :meth:`~agent_memory.manager.Memory.resolve`.

    Parameters
    ----------
    memory:
        An initialised :class:`~agent_memory.manager.Memory` instance.
    memory_key:
        The key under which context is returned in ``load_memory_variables``.
        Defaults to ``"history"``.
    input_key:
        The key in ``inputs`` dict that holds the human turn.  Defaults to
        ``"input"``.
    output_key:
        The key in ``outputs`` dict that holds the AI turn.  Defaults to
        ``"output"``.
    return_messages:
        When ``True``, return a list of :class:`BaseMessage` objects instead of
        a plain string.  Matches the interface expected by chat models.
    top_k:
        How many memories to retrieve per query.
    memory_type:
        Memory type applied to stored turns (default ``"conversation"``).
    scope:
        Memory scope applied to stored turns (default ``"user"``).
    """

    # Pydantic / langchain requires class-level field declarations.
    # We bypass that by using __init__ directly and storing state as
    # regular Python attributes, which is safe for non-Pydantic subclasses.

    def __init__(
        self,
        memory: Memory,
        *,
        memory_key: str = "history",
        input_key: str = "input",
        output_key: str = "output",
        return_messages: bool = False,
        top_k: int = 3,
        memory_type: str = "conversation",
        scope: str = "user",
    ) -> None:
        if not LANGCHAIN_AVAILABLE:
            raise ImportError(
                "LangChain adapter requires langchain-core. "
                "Install with: pip install agent-memory-sdk[langchain]"
            )
        super().__init__()
        self._mem = memory
        self.memory_key = memory_key
        self.input_key = input_key
        self.output_key = output_key
        self.return_messages = return_messages
        self.top_k = top_k
        self.memory_type = memory_type
        self.scope = scope

    # ------------------------------------------------------------------
    # BaseMemory interface
    # ------------------------------------------------------------------

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        """Return relevant context for the current query."""
        query: str = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: [] if self.return_messages else ""}

        decision = self._mem.resolve(query, top_k=self.top_k)

        if self.return_messages:
            messages: list[Any] = []
            if decision.memory:
                messages.append(HumanMessage(content=decision.memory.query))
                messages.append(AIMessage(content=decision.memory.response))
            for ctx in decision.context:
                messages.append(HumanMessage(content=ctx.entry.query))
                messages.append(AIMessage(content=ctx.entry.response))
            return {self.memory_key: messages}

        # Plain-text format
        lines: list[str] = []
        if decision.memory:
            lines.append(f"Human: {decision.memory.query}")
            lines.append(f"AI: {decision.memory.response}")
        for ctx in decision.context:
            lines.append(f"Human: {ctx.entry.query}")
            lines.append(f"AI: {ctx.entry.response}")
        return {self.memory_key: "\n".join(lines)}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, Any]) -> None:
        """Store a conversation turn."""
        query = inputs.get(self.input_key, "")
        response = outputs.get(self.output_key, "")
        if not query:
            return
        self._mem.remember(
            query=query,
            response=response,
            type=self.memory_type,
            scope=self.scope,
        )

    def clear(self) -> None:
        """Archive all memories stored through this adapter (non-destructive)."""
        entries = self._mem.list(limit=10_000)
        for entry in entries:
            self._mem.archive(entry.id)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_persist_dir(
        cls,
        persist_dir: str = ".agent_memory",
        **kwargs: Any,
    ) -> AgentMemoryLangChain:
        """Create a LangChain memory backed by a fresh Memory at *persist_dir*."""
        from agent_memory.manager import Memory

        return cls(Memory(persist_dir=persist_dir), **kwargs)

"""LlamaIndex integration — agent-memory-sdk as the memory backend.

Install:
    pip install agent-memory-sdk[llamaindex]
    pip install llama-index-llms-openai   # or another LLM provider

The adapter satisfies LlamaIndex's BaseMemory interface so it plugs into
any chat engine or OpenAI agent that accepts `memory=`.

agent-memory-sdk handles persistence, semantic retrieval, and the
replay/restore/verify decision; LlamaIndex provides the agent orchestration.
"""
from __future__ import annotations

from agent_memory import Memory
from agent_memory.adapters.llamaindex_adapter import AgentMemoryLlamaIndex

# ── 1. Create the shared memory store ────────────────────────────────────────
memory = Memory(persist_dir=".agent_memory", collection_name="llamaindex_demo")

# ── 2. Wrap it in the LlamaIndex adapter ─────────────────────────────────────
li_memory = AgentMemoryLlamaIndex(
    memory=memory,
    top_k=5,              # memories retrieved per query
    token_limit=3000,     # approximate token budget for returned context
    memory_type="conversation",
    scope="user",
)

# ── 3. Seed some context then demonstrate retrieval ──────────────────────────
memory.remember("API rate limits", "Free: 100 req/min. Pro: 1000 req/min.", type="fact")
memory.remember("How to authenticate", "Use Authorization: Bearer <token>", type="fact")

# get() returns ChatMessage objects for the given query
try:
    from llama_index.core.llms import ChatMessage, MessageRole

    msgs = li_memory.get(input="What are the API limits?")
    print(f"Retrieved {len(msgs)} messages for 'API limits' query:")
    for m in msgs:
        role = str(m.role).split(".")[-1].lower()
        print(f"  [{role}] {str(m.content)[:80]}")
    print()

    # put() buffers a turn; flushes USER+ASSISTANT pairs to persistent memory
    li_memory.put(ChatMessage(role=MessageRole.USER,      content="What is 2FA?"))
    li_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Two-factor authentication adds a second verification step."))

except ImportError:
    print("llama-index-core not installed — message demo skipped.")

# ── 4. Use with a LlamaIndex agent ────────────────────────────────────────────
try:
    from llama_index.agent.openai import OpenAIAgent
    from llama_index.core.tools import FunctionTool
    from llama_index.llms.openai import OpenAI

    def search_docs(query: str) -> str:
        """Search the documentation."""
        return f"Documentation result for: {query}"

    tools = [FunctionTool.from_defaults(fn=search_docs)]
    llm = OpenAI(model="gpt-4o-mini")

    # Pass the adapter directly — the agent uses it for context management
    agent = OpenAIAgent.from_tools(tools, llm=llm, memory=li_memory, verbose=True)
    response = agent.chat("How do I authenticate API requests?")
    print(f"Agent response: {response}")

except ImportError:
    print("llama-index-llms-openai not installed — agent demo skipped.")

print(f"\nTotal memories stored: {memory.store.count}")

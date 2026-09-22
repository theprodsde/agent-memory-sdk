"""LangChain integration — agent-memory-sdk as the memory backend.

Install:
    pip install agent-memory-sdk[langchain]
    pip install langchain-openai   # or any other LLM provider

The adapter satisfies LangChain's BaseMemory interface, so it drops into
any chain or agent that accepts a `memory=` argument.

agent-memory-sdk is the source of truth for what gets stored and retrieved;
LangChain is purely the LLM orchestration layer on top.
"""
from __future__ import annotations

from agent_memory import Memory
from agent_memory.adapters.langchain_adapter import AgentMemoryLangChain

# ── 1. Create the shared memory store ────────────────────────────────────────
memory = Memory(persist_dir=".agent_memory", collection_name="langchain_demo")

# ── 2. Wrap it in the LangChain adapter ──────────────────────────────────────
lc_memory = AgentMemoryLangChain(
    memory=memory,
    memory_key="history",       # key injected into chain inputs
    input_key="input",          # key that holds the human turn
    output_key="output",        # key that holds the AI turn
    return_messages=False,      # True → list[BaseMessage], False → plain string
    top_k=3,                    # how many past memories to retrieve
    memory_type="conversation", # MemoryType stored for each turn
    scope="user",               # MemoryScope for isolation
)

# ── 3. Use standalone (no LLM required for this demo) ────────────────────────

# Store a conversation turn
lc_memory.save_context(
    inputs={"input": "How do I reset my password?"},
    outputs={"output": "Go to Settings → Security → Reset Password."},
)

lc_memory.save_context(
    inputs={"input": "What payment methods do you accept?"},
    outputs={"output": "Visa, Mastercard, and PayPal."},
)

# Retrieve context relevant to a new query
vars_ = lc_memory.load_memory_variables({"input": "I forgot my password"})
print("Retrieved context:")
print(vars_["history"])
print()

# ── 4. Use with an actual LangChain ConversationChain ────────────────────────
try:
    from langchain.chains import ConversationChain
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = ConversationChain(llm=llm, memory=lc_memory, verbose=True)

    # First turn — memory decides whether to surface past context
    resp = chain.predict(input="How do I reset my password?")
    print(f"Chain response: {resp}")

except ImportError:
    print("langchain-openai not installed — chain demo skipped.")

# ── 5. Inspect what was stored ────────────────────────────────────────────────
print(f"\nTotal memories stored: {memory.store.count}")
for e in memory.list(limit=5):
    print(f"  [{e.type.value}] {e.query[:60]}  (confidence: {e.confidence:.0%})")

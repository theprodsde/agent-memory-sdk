"""Multi-agent memory — multiple agents sharing one store with isolation.

Three isolation modes:
  SHARED     — all agents see all memories
  NAMESPACED — each agent sees its own + global memories (default)
  ISOLATED   — each agent sees only its own memories

agent-memory-sdk is the central shared store. Each agent gets a scoped
view of it through MultiAgentMemory — no separate databases required.
"""
from __future__ import annotations

from agent_memory import Memory, MultiAgentMemory
from agent_memory.multiagent import IsolationMode

# ── One shared store, multiple agents ────────────────────────────────────────
shared_store = Memory(persist_dir=".agent_memory", collection_name="multi_agent_demo")

support_agent  = MultiAgentMemory(shared_store, agent_id="support",   isolation=IsolationMode.NAMESPACED)
research_agent = MultiAgentMemory(shared_store, agent_id="research",  isolation=IsolationMode.NAMESPACED)
admin_agent    = MultiAgentMemory(shared_store, agent_id="admin",     isolation=IsolationMode.SHARED)

# ── Each agent stores its own memories ───────────────────────────────────────
support_agent.remember(
    "How do I reset my password?",
    "Go to Settings → Security → Reset Password.",
    type="conversation", tags=["auth"],
)
support_agent.remember(
    "How do I cancel my subscription?",
    "Account → Billing → Cancel Plan.",
    type="conversation", tags=["billing"],
)

research_agent.remember(
    "Latest ML model accuracy benchmark",
    "GPT-4o achieves 88.7% on MMLU as of 2024.",
    type="fact", tags=["ml", "benchmark"],
    requires_verification=True,
)

# ── Broadcast: visible to every agent regardless of isolation ────────────────
admin_agent.broadcast(
    "Company name",
    "Acme Corp. — always use this name in responses.",
    type="fact", tags=["brand"],
)

admin_agent.broadcast(
    "Support escalation policy",
    "Escalate P1 issues to on-call within 15 minutes.",
    type="workflow", tags=["policy", "support"],
)

# ── NAMESPACED: each agent only sees its own + global memories ────────────────
print("Support agent sees:")
for e in support_agent.list():
    src = e.metadata.get("agent_id", "global")
    print(f"  [{src}] {e.query[:60]}")

print("\nResearch agent sees:")
for e in research_agent.list():
    src = e.metadata.get("agent_id", "global")
    print(f"  [{src}] {e.query[:60]}")

# ── SHARED: admin sees everything ─────────────────────────────────────────────
print(f"\nAdmin (SHARED) sees {len(admin_agent.list())} memories total")

# ── Resolve from each agent's perspective ────────────────────────────────────
print("\nSupport resolves 'reset password':")
d = support_agent.resolve("reset password")
print(f"  → {d.action.value}  conf={d.confidence:.2f}")

# NOTE: resolve() searches the full store by default (isolation applies to list()).
# To scope a resolve to only a specific agent's memories plus globals, combine
# resolve() with metadata filtering in post-processing, or use list() + keyword_search.
print("\nResearch resolves 'reset password':")
d = research_agent.resolve("reset password")
print(f"  → {d.action.value}  conf={d.confidence:.2f}  (resolve searches full store)")
print("  Use list() + keyword_search() for isolation-aware search.")

print("\nBoth resolve 'company name' (global broadcast):")
for name, agent in [("support", support_agent), ("research", research_agent)]:
    d = agent.resolve("What is the company name?")
    print(f"  {name} → {d.action.value}  conf={d.confidence:.2f}")

# ── Transfer memory ownership ─────────────────────────────────────────────────
entries = support_agent.list()
if entries:
    target = entries[0]
    result = support_agent.transfer_memory(target.id, "research")
    if result:
        print(f"\nTransferred '{target.query[:40]}' from support → research")

# ── Discover active agents ────────────────────────────────────────────────────
print(f"\nAgents with memories: {admin_agent.agents_with_memories()}")

"""PostgreSQL backend — agent-memory-sdk with Postgres as the memory store.

Install:
    pip install agent-memory-sdk[postgres]

Requires a running Postgres instance:
    docker compose -f docker-compose.dev.yml up -d postgres
    # default DSN: postgresql://agent_memory:agent_memory@localhost/agent_memory

agent-memory-sdk creates and manages its own table; it does not interfere
with any existing schema in the database.
"""
from __future__ import annotations

import os

from agent_memory import Memory, MemoryAction, MemoryScope, MemoryType

DSN = os.environ.get(
    "AGENT_MEMORY_POSTGRES_DSN",
    "postgresql://agent_memory:agent_memory@localhost/agent_memory",
)

# ── Connect ───────────────────────────────────────────────────────────────────
memory = Memory(
    backend="postgres",
    dsn=DSN,
    table_name="agent_memories",   # table created automatically if absent
)

print(f"Connected to Postgres — existing memories: {memory.store.count}")

# ── Store memories across multiple scopes ─────────────────────────────────────
memory.remember(
    "How do I reset my password?",
    "Go to Settings → Security → Reset Password.",
    type=MemoryType.CONVERSATION,
    scope=MemoryScope.GLOBAL,
    tags=["auth"],
)

memory.remember(
    "Project deadline is Friday",
    "The Q4 release must ship by 2024-12-20.",
    type=MemoryType.FACT,
    scope=MemoryScope.PROJECT,
    tags=["deadline", "q4"],
    requires_verification=True,
)

memory.remember(
    "User prefers dark mode",
    "Apply dark theme by default for this user.",
    type=MemoryType.PREFERENCE,
    scope=MemoryScope.USER,
    tags=["ui", "preference"],
)

memory.remember(
    "Onboarding workflow",
    "1. Create account  2. Verify email  3. Invite team  4. Set up SSO.",
    type=MemoryType.WORKFLOW,
    scope=MemoryScope.TEAM,
    tags=["onboarding"],
)

print(f"Total after seeding: {memory.store.count}")

# ── Scope-filtered resolve ─────────────────────────────────────────────────────
# Only search user-scoped or global memories for this user's session
user_decision = memory.resolve(
    "How do I reset my password?",
    scope=[MemoryScope.USER, MemoryScope.GLOBAL],
)
print(f"\nUser-scoped resolve: {user_decision.action.value}  "
      f"conf={user_decision.confidence:.2f}")
if user_decision.action == MemoryAction.REPLAY and user_decision.memory:
    print(f"  → {user_decision.memory.response}")

# Search across all scopes
global_decision = memory.resolve("What is the project deadline?")
print(f"\nGlobal resolve: {global_decision.action.value}  "
      f"conf={global_decision.confidence:.2f}")

# ── List by scope ─────────────────────────────────────────────────────────────
print("\nMemories by scope:")
for scope in [MemoryScope.USER, MemoryScope.PROJECT, MemoryScope.GLOBAL]:
    entries = memory.list(scope=[scope], limit=10)
    print(f"  {scope.value}: {len(entries)} entries")

# ── SQL-powered stats ─────────────────────────────────────────────────────────
stats = memory.stats()
print(f"\nStats: {stats}")

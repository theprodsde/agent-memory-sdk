"""Redis backend — agent-memory-sdk with Redis as the memory store.

Install:
    pip install agent-memory-sdk[redis]

Requires a running Redis server:
    docker compose -f docker-compose.dev.yml up -d redis
    # or: docker run -p 6379:6379 redis:7-alpine

agent-memory-sdk handles all the storage, BM25 keyword search, TTL, and
decision logic. Redis is just the persistence layer.
"""
from __future__ import annotations

from agent_memory import Memory, MemoryAction, MemoryType

# ── Connect to Redis ──────────────────────────────────────────────────────────
# Option A: URL string (recommended)
memory = Memory(
    backend="redis",
    url="redis://localhost:6379/0",
    key_prefix="myapp_memory",   # namespace all keys under this prefix
)

# Option B: host/port/db
# memory = Memory(backend="redis", host="localhost", port=6379, db=0)

# Option C: Redis Cloud / authenticated
# memory = Memory(backend="redis", url="redis://:password@host:6380/0")

# ── Store memories ────────────────────────────────────────────────────────────
memory.remember(
    "How do I reset my password?",
    "Go to Settings → Security → Reset Password. Link expires in 30 minutes.",
    type=MemoryType.CONVERSATION,
    tags=["auth", "password"],
    confidence=1.0,
)

memory.remember(
    "What is the API rate limit?",
    "Free: 100 req/min. Pro: 1000 req/min. Enterprise: unlimited.",
    type=MemoryType.FACT,
    tags=["api", "limits"],
    confidence=1.0,
    requires_verification=True,  # will return VERIFY, not REPLAY
)

memory.remember(
    "How do I invite team members?",
    "Settings → Team → Invite. Invitations expire after 7 days.",
    type=MemoryType.WORKFLOW,
    tags=["team", "onboarding"],
)

print(f"Stored {memory.store.count} memories in Redis")

# ── Resolve ────────────────────────────────────────────────────────────────────
test_queries = [
    ("How do I reset my password?",           "→ exact match, expect REPLAY"),
    ("I forgot my password",                   "→ paraphrase, expect REPLAY or RESTORE"),
    ("What are the rate limits for the API?",  "→ similar, expect VERIFY (flagged fact)"),
    ("How do I add a new user to my account?", "→ close paraphrase of invite workflow"),
    ("What is the capital of France?",         "→ out-of-domain, expect NONE"),
]

print()
for query, note in test_queries:
    d = memory.resolve(query)
    badge = {"replay": "✅ REPLAY", "restore": "📋 RESTORE",
             "verify": "⚠️  VERIFY", "none": "❌ NONE"}.get(d.action.value, d.action.value)
    print(f"{badge}  conf={d.confidence:.2f}  {note}")
    print(f"  Q: {query}")
    if d.action == MemoryAction.REPLAY and d.memory:
        print(f"  A: {d.memory.response}")
    print()

# ── Stats ─────────────────────────────────────────────────────────────────────
stats = memory.stats()
print(f"Stats: total={stats['total']}  by_type={stats['by_type']}")

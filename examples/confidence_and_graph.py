"""Confidence learning + memory graph — advanced usage of agent-memory-sdk.

These features build on top of the core Memory store:
  - ConfidenceLearner: update confidence based on feedback events
  - MemoryGraph: discover relationships between memories

agent-memory-sdk is the single package — no separate dependencies.
"""
from __future__ import annotations

from agent_memory import (
    ConfidenceEvent,
    ConfidenceLearner,
    Memory,
    MemoryGraph,
    MemoryType,
)

# ── Shared setup ──────────────────────────────────────────────────────────────
memory = Memory(persist_dir=".agent_memory", collection_name="advanced_demo")
learner = ConfidenceLearner()

# ── Seed some memories ────────────────────────────────────────────────────────
facts = [
    ("Python async programming", "Use asyncio. Define coroutines with async def.", ["python", "async"]),
    ("asyncio event loop",       "Loop runs coroutines. Use asyncio.run() to start.", ["python", "async"]),
    ("Python type hints",        "Use : TypeName for annotations. Run mypy to check.", ["python", "types"]),
    ("Python list comprehensions","[x for x in iterable if condition]",              ["python", "syntax"]),
    ("REST API design",          "Use nouns for resources. HTTP verbs for actions.",  ["api", "rest"]),
    ("API authentication",       "Use Bearer tokens in Authorization header.",        ["api", "auth"]),
    ("SQL indexing",             "Index columns used in WHERE and JOIN clauses.",     ["sql", "performance"]),
    ("Database transactions",    "Use ACID transactions for data integrity.",         ["sql", "database"]),
]

for query, response, tags in facts:
    memory.remember(query, response, type=MemoryType.FACT, tags=tags)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIDENCE LEARNING
# ─────────────────────────────────────────────────────────────────────────────
print("=== Confidence Learning ===\n")

entries = memory.list(limit=100)

# Simulate a user confirming the first memory
first = entries[0]
print(f"Before: '{first.query[:40]}'  confidence={first.confidence:.0%}")

update = learner.record_event(first, ConfidenceEvent.USER_CONFIRMED, persist_timestamp=True)
memory.store.update(first)
print(f"After USER_CONFIRMED:  {update.old_confidence:.0%} → {update.new_confidence:.0%}  (Δ{update.delta:+.2f})")

# Simulate a wrong answer being caught
second = entries[1]
update = learner.record_event(second, ConfidenceEvent.VERIFIED_INCORRECT, persist_timestamp=True)
memory.store.update(second)
print(f"After VERIFIED_INCORRECT: {update.old_confidence:.0%} → {update.new_confidence:.0%}  (Δ{update.delta:+.2f})")

# Batch update: mark several as accessed
accessed = entries[2:5]
events   = [ConfidenceEvent.ACCESSED] * len(accessed)
updates  = learner.batch_update(accessed, events)
for e, u in zip(accessed, updates):
    memory.store.update(e)
print(f"Batch ACCESSED on {len(updates)} memories: avg Δ{sum(u.delta for u in updates)/len(updates):+.3f}")

# Run temporal decay across all entries
print("\nRunning temporal decay (half-life = 90 days)…")
decayed_count = 0
for e in memory.list(limit=10_000):
    u = learner.decay(e)
    if abs(u.delta) > 0.001:
        memory.store.update(e)
        decayed_count += 1
print(f"Decayed {decayed_count} memories")

# ─────────────────────────────────────────────────────────────────────────────
# MEMORY GRAPH
# ─────────────────────────────────────────────────────────────────────────────
print("\n=== Memory Graph ===\n")

graph = MemoryGraph.build(
    memory.store,
    similarity_threshold=0.3,    # lower = more edges
    tag_weight_threshold=0.25,   # min tag overlap fraction for an edge
    max_edges_per_node=5,
)

print(f"Graph: {len(graph.nodes)} nodes, {len(graph.edges)} edges")

# Neighbours of the first Python entry
python_id = next(mid for mid, e in graph.nodes.items() if "asyncio" in e.query.lower())
neighbours = graph.neighbors(python_id, min_weight=0.2)
print(f"\nNeighbours of '{graph.nodes[python_id].query}':")
for entry, weight in neighbours:
    print(f"  {weight:.3f}  {entry.query}")

# Find path between two memories
sql_id = next((mid for mid, e in graph.nodes.items() if "sql" in e.query.lower()), None)
if sql_id:
    path = graph.path(python_id, sql_id)
    if path:
        print(f"\nPath Python→SQL ({len(path)} hops):")
        print("  " + " → ".join(e.query[:30] for e in path))
    else:
        print("\nNo direct path between Python and SQL clusters")

# Clusters
clusters = graph.clusters(min_weight=0.25)
print(f"\nClusters at threshold 0.25: {len(clusters)}")
for i, cluster in enumerate(clusters, 1):
    print(f"  Cluster {i}: {', '.join(e.tags[0] if e.tags else '?' for e in cluster[:3])}"
          f"  ({len(cluster)} memories)")

# PageRank importance
scores = graph.importance_scores()
top5 = sorted(scores.items(), key=lambda x: -x[1])[:5]
print("\nTop 5 most important memories (PageRank):")
for nid, score in top5:
    print(f"  {score:.4f}  {graph.nodes[nid].query[:55]}")

# Export for external tools (e.g. d3.js, Gephi)
d = graph.to_dict()
print(f"\nGraph exported: {len(d['nodes'])} nodes, {len(d['edges'])} edges")

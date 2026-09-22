from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any

from agent_memory.models import MemoryEntry
from agent_memory.store import MemoryStore


@dataclass
class MemoryEdge:
    """A directed, weighted relationship between two memories."""

    source_id: str
    target_id: str
    weight: float
    relation: str = "similar"  # "similar" | "tagged" | "custom"


@dataclass
class MemoryGraph:
    """Graph of memory relationships built from a MemoryStore.

    Nodes are MemoryEntry objects; edges are built from semantic similarity
    (via store.search) and shared-tag overlap.  All edges are undirected in
    practice — both directions are stored in the adjacency list.

    Usage::

        graph = MemoryGraph.build(memory.store)
        neighbors = graph.neighbors(entry_id, min_weight=0.5)
        path = graph.path(a_id, b_id)
        clusters = graph.clusters()
        scores = graph.importance_scores()
    """

    nodes: dict[str, MemoryEntry] = field(default_factory=dict)
    edges: list[MemoryEdge] = field(default_factory=list)
    _adj: dict[str, list[tuple[str, float]]] = field(
        default_factory=dict, repr=False
    )

    @classmethod
    def build(
        cls,
        store: MemoryStore,
        *,
        similarity_threshold: float = 0.4,
        max_edges_per_node: int = 10,
        tag_weight_threshold: float = 0.3,
    ) -> MemoryGraph:
        """Build a graph from all active memories in *store*.

        For each memory, ``store.search()`` is called to find similar entries;
        tag overlap is computed in a second pass.  Edges below the respective
        thresholds are discarded.
        """
        graph = cls(_adj={})
        entries = store.list_all(limit=10_000)
        for entry in entries:
            graph.nodes[entry.id] = entry
            graph._adj[entry.id] = []

        seen_pairs: set[frozenset] = set()

        for entry in entries:
            hits = store.search(entry.query, top_k=max_edges_per_node + 1)
            for hit_entry, score in hits:
                if hit_entry.id == entry.id or score < similarity_threshold:
                    continue
                pair: frozenset = frozenset({entry.id, hit_entry.id})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge = MemoryEdge(
                    source_id=entry.id,
                    target_id=hit_entry.id,
                    weight=score,
                    relation="similar",
                )
                graph.edges.append(edge)
                graph._adj[entry.id].append((hit_entry.id, score))
                graph._adj.setdefault(hit_entry.id, []).append((entry.id, score))

        # Tag-overlap pass
        entry_list = list(entries)
        for i, a in enumerate(entry_list):
            for b in entry_list[i + 1 :]:
                if not a.tags or not b.tags:
                    continue
                shared = set(a.tags) & set(b.tags)
                if not shared:
                    continue
                tag_weight = len(shared) / max(len(a.tags), len(b.tags))
                if tag_weight < tag_weight_threshold:
                    continue
                pair = frozenset({a.id, b.id})
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                edge = MemoryEdge(
                    source_id=a.id,
                    target_id=b.id,
                    weight=tag_weight,
                    relation="tagged",
                )
                graph.edges.append(edge)
                graph._adj[a.id].append((b.id, tag_weight))
                graph._adj.setdefault(b.id, []).append((a.id, tag_weight))

        return graph

    def neighbors(
        self, memory_id: str, *, min_weight: float = 0.0
    ) -> list[tuple[MemoryEntry, float]]:
        """Return adjacent memories sorted by edge weight descending."""
        result: list[tuple[MemoryEntry, float]] = []
        for neighbor_id, weight in self._adj.get(memory_id, []):
            if weight < min_weight:
                continue
            entry = self.nodes.get(neighbor_id)
            if entry:
                result.append((entry, weight))
        result.sort(key=lambda p: p[1], reverse=True)
        return result

    def path(self, source_id: str, target_id: str) -> list[MemoryEntry]:
        """BFS shortest path between two memories. Returns ``[]`` if unreachable."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return []
        visited: set[str] = set()
        queue: deque[list[str]] = deque([[source_id]])
        while queue:
            cur_path = queue.popleft()
            node = cur_path[-1]
            if node == target_id:
                return [self.nodes[mid] for mid in cur_path]
            if node in visited:
                continue
            visited.add(node)
            for neighbor_id, _ in self._adj.get(node, []):
                if neighbor_id not in visited:
                    queue.append(cur_path + [neighbor_id])
        return []

    def clusters(self, *, min_weight: float = 0.4) -> list[list[MemoryEntry]]:
        """Return connected components where all edges have weight ≥ *min_weight*."""
        visited: set[str] = set()
        components: list[list[MemoryEntry]] = []

        for node_id in self.nodes:
            if node_id in visited:
                continue
            component: list[str] = []
            stack = [node_id]
            while stack:
                nid = stack.pop()
                if nid in visited:
                    continue
                visited.add(nid)
                component.append(nid)
                for neighbor_id, w in self._adj.get(nid, []):
                    if neighbor_id not in visited and w >= min_weight:
                        stack.append(neighbor_id)
            components.append(
                [self.nodes[mid] for mid in component if mid in self.nodes]
            )

        return sorted(components, key=len, reverse=True)

    def importance_scores(
        self, *, damping: float = 0.85, iterations: int = 20
    ) -> dict[str, float]:
        """PageRank-style importance scores for all nodes.

        Returns a mapping of ``memory_id → score`` (scores sum to 1.0).
        """
        n = len(self.nodes)
        if n == 0:
            return {}

        node_ids = list(self.nodes.keys())
        scores: dict[str, float] = {nid: 1.0 / n for nid in node_ids}

        for _ in range(iterations):
            new_scores: dict[str, float] = {}
            for nid in node_ids:
                in_score = 0.0
                for neighbor_id, weight in self._adj.get(nid, []):
                    out_weights = sum(w for _, w in self._adj.get(neighbor_id, []))
                    if out_weights > 0:
                        in_score += scores.get(neighbor_id, 0.0) * weight / out_weights
                new_scores[nid] = (1 - damping) / n + damping * in_score
            scores = new_scores

        return scores

    def to_dict(self) -> dict[str, Any]:
        """Serialise the graph as a plain dict (useful for JSON export / dashboards)."""
        return {
            "nodes": [
                {
                    "id": mid,
                    "query": entry.query[:80],
                    "type": entry.type.value,
                    "scope": entry.scope.value,
                    "tags": entry.tags,
                }
                for mid, entry in self.nodes.items()
            ],
            "edges": [
                {
                    "source": e.source_id,
                    "target": e.target_id,
                    "weight": round(e.weight, 4),
                    "relation": e.relation,
                }
                for e in self.edges
            ],
        }

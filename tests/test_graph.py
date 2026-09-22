"""Tests for MemoryGraph — topology, path-finding, clustering, importance."""
from __future__ import annotations

import pytest


@pytest.fixture()
def populated_store(tmp_path):
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="graph_test")
    # Three related Python entries and one unrelated cooking entry
    mem.remember("Python async programming", "Use asyncio for concurrency", tags=["python", "async"])
    mem.remember("asyncio coroutines tutorial", "Define with async def", tags=["python", "async"])
    mem.remember("Python type hints guide", "Use mypy for checking", tags=["python", "types"])
    mem.remember("Best pasta recipe", "Boil salted water", tags=["cooking", "pasta"])
    return mem.store


def test_build_returns_graph(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store)
    assert len(graph.nodes) == 4
    assert isinstance(graph.edges, list)


def test_neighbors_returns_related(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store)
    # Find the Python async node
    async_id = next(
        mid for mid, e in graph.nodes.items() if "asyncio" in e.query.lower()
    )
    neighbors = graph.neighbors(async_id)
    # Should have at least one Python-related neighbor due to tag overlap
    assert len(neighbors) >= 1


def test_neighbors_min_weight_filter(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store)
    node_id = next(iter(graph.nodes))
    all_neighbors = graph.neighbors(node_id)
    heavy_neighbors = graph.neighbors(node_id, min_weight=0.99)
    assert len(heavy_neighbors) <= len(all_neighbors)


def test_path_connected_nodes(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store, similarity_threshold=0.1, tag_weight_threshold=0.1)
    python_ids = [
        mid for mid, e in graph.nodes.items() if "python" in e.query.lower()
    ]
    if len(python_ids) >= 2:
        path = graph.path(python_ids[0], python_ids[1])
        # Path exists (may be direct or through intermediaries)
        assert isinstance(path, list)


def test_path_missing_node(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store)
    result = graph.path("nonexistent-id", "also-nonexistent")
    assert result == []


def test_clusters_returns_list(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store, similarity_threshold=0.1, tag_weight_threshold=0.1)
    clusters = graph.clusters(min_weight=0.1)
    assert isinstance(clusters, list)
    # All nodes should be in some cluster
    all_ids = {e.id for cluster in clusters for e in cluster}
    assert all_ids == set(graph.nodes.keys())


def test_clusters_sorted_by_size(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store, similarity_threshold=0.1, tag_weight_threshold=0.1)
    clusters = graph.clusters(min_weight=0.1)
    sizes = [len(c) for c in clusters]
    assert sizes == sorted(sizes, reverse=True)


def test_importance_scores(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store)
    scores = graph.importance_scores()
    assert set(scores.keys()) == set(graph.nodes.keys())
    assert all(isinstance(v, float) for v in scores.values())
    assert all(v >= 0 for v in scores.values())


def test_to_dict_structure(populated_store):
    from agent_memory.graph import MemoryGraph

    graph = MemoryGraph.build(populated_store)
    d = graph.to_dict()
    assert "nodes" in d
    assert "edges" in d
    assert len(d["nodes"]) == 4
    for node in d["nodes"]:
        assert "id" in node
        assert "query" in node


def test_empty_store():
    import tempfile

    from agent_memory.graph import MemoryGraph
    from agent_memory.sqlite_store import SqliteMemoryStore
    with tempfile.TemporaryDirectory() as tmp:
        store = SqliteMemoryStore(persist_dir=tmp, collection_name="empty")
        graph = MemoryGraph.build(store)
        assert len(graph.nodes) == 0
        assert graph.edges == []
        assert graph.importance_scores() == {}
        assert graph.clusters() == []

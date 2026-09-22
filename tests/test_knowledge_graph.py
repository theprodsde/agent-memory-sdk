"""Tests for KnowledgeGraph — typed entities, relations, and queries."""
from __future__ import annotations

import pytest

from agent_memory.knowledge_graph import (
    EntityType,
    KnowledgeGraph,
    RelationType,
)
from agent_memory.models import MemoryType

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def mem(tmp_path):
    from agent_memory.manager import Memory
    m = Memory(persist_dir=tmp_path, collection_name="kg_test")
    m.remember(
        "Where does Alice work?",
        "Alice works at Acme Corp as a senior engineer.",
        type=MemoryType.FACT, tags=["person", "work"],
    )
    m.remember(
        "Where is Acme Corp located?",
        "Acme Corp is located in London.",
        type=MemoryType.FACT, tags=["organization", "location"],
    )
    m.remember(
        "What technology does Bob use?",
        "Bob uses Python and FastAPI for all his projects.",
        type=MemoryType.FACT, tags=["person", "technology"],
    )
    m.remember(
        "What is the API rate limit?",
        "The API rate limit is 1000 requests per minute.",
        type=MemoryType.FACT, tags=["api", "limits"],
    )
    return m


@pytest.fixture()
def kg(mem):
    return KnowledgeGraph.build(mem.store)


# ---------------------------------------------------------------------------
# Build
# ---------------------------------------------------------------------------


def test_build_produces_entities(kg):
    assert len(kg.entities) > 0


def test_build_produces_relations(kg):
    assert len(kg.relations) > 0


# ---------------------------------------------------------------------------
# add_entity / find_entity
# ---------------------------------------------------------------------------


def test_add_entity_basic(kg):
    e = kg.add_entity("Charlie", EntityType.PERSON, {"role": "devops"})
    assert e.name == "Charlie"
    assert e.type == EntityType.PERSON
    assert e.attributes["role"] == "devops"


def test_find_entity_case_insensitive(kg):
    kg.add_entity("Django Framework", EntityType.PRODUCT)
    found = kg.find_entity("django framework")
    assert found is not None
    assert found.name == "Django Framework"


def test_add_entity_merges_duplicates(kg):
    e1 = kg.add_entity("TestOrg", EntityType.ORGANIZATION)
    e2 = kg.add_entity("TestOrg", EntityType.ORGANIZATION, {"country": "UK"})
    assert e1.id == e2.id
    assert e1.attributes.get("country") == "UK"


def test_find_entity_returns_none_for_missing(kg):
    assert kg.find_entity("NonExistentEntity12345") is None


# ---------------------------------------------------------------------------
# add_relation / relations_for
# ---------------------------------------------------------------------------


def test_add_and_query_relation(kg):
    alice = kg.add_entity("Alice2", EntityType.PERSON)
    org   = kg.add_entity("WidgetCo", EntityType.ORGANIZATION)
    kg.add_relation(alice.id, RelationType.WORKS_AT, org.id, target_is_entity=True)

    rels = kg.relations_for(alice.id)
    assert any(r.relation == RelationType.WORKS_AT for r in rels)


def test_relation_filter_by_type(kg):
    person = kg.add_entity("Dave", EntityType.PERSON)
    org    = kg.add_entity("MegaCorp", EntityType.ORGANIZATION)
    loc    = kg.add_entity("Berlin", EntityType.LOCATION)
    kg.add_relation(person.id, RelationType.WORKS_AT,   org.id, target_is_entity=True)
    kg.add_relation(person.id, RelationType.LOCATED_IN, loc.id, target_is_entity=True)

    works_at = kg.relations_for(person.id, relation_type=RelationType.WORKS_AT)
    assert all(r.relation == RelationType.WORKS_AT for r in works_at)


# ---------------------------------------------------------------------------
# neighbors
# ---------------------------------------------------------------------------


def test_neighbors_returns_connected_entities(kg):
    a = kg.add_entity("EntityA", EntityType.CONCEPT)
    b = kg.add_entity("EntityB", EntityType.CONCEPT)
    kg.add_relation(a.id, RelationType.RELATED_TO, b.id, target_is_entity=True)
    nbrs = kg.neighbors(a.id)
    nbr_ids = {e.id for e, _ in nbrs}
    assert b.id in nbr_ids


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


def test_search_by_name(kg):
    kg.add_entity("FastAPI Framework", EntityType.PRODUCT)
    results = kg.search("fastapi")
    assert any("FastAPI" in e.name for e in results)


def test_search_by_attribute(kg):
    e = kg.add_entity("Eve", EntityType.PERSON, {"team": "platform"})
    results = kg.search("platform")
    assert any(r.id == e.id for r in results)


def test_search_no_match(kg):
    results = kg.search("xyzxyzxyznonexistent")
    assert results == []


# ---------------------------------------------------------------------------
# find_by_type
# ---------------------------------------------------------------------------


def test_find_by_type(kg):
    kg.add_entity("NewOrg", EntityType.ORGANIZATION)
    orgs = kg.find_by_type(EntityType.ORGANIZATION)
    assert any(e.name == "NewOrg" for e in orgs)


# ---------------------------------------------------------------------------
# path
# ---------------------------------------------------------------------------


def test_path_direct(kg):
    a = kg.add_entity("PathA", EntityType.CONCEPT)
    b = kg.add_entity("PathB", EntityType.CONCEPT)
    kg.add_relation(a.id, RelationType.RELATED_TO, b.id, target_is_entity=True)
    path = kg.path("PathA", "PathB")
    assert len(path) >= 2
    assert path[0].id == a.id
    assert path[-1].id == b.id


def test_path_indirect(kg):
    a = kg.add_entity("HopA", EntityType.CONCEPT)
    b = kg.add_entity("HopB", EntityType.CONCEPT)
    c = kg.add_entity("HopC", EntityType.CONCEPT)
    kg.add_relation(a.id, RelationType.RELATED_TO, b.id, target_is_entity=True)
    kg.add_relation(b.id, RelationType.RELATED_TO, c.id, target_is_entity=True)
    path = kg.path("HopA", "HopC")
    assert len(path) >= 2


def test_path_no_connection(kg):
    kg.add_entity("Island1", EntityType.CONCEPT)
    kg.add_entity("Island2", EntityType.CONCEPT)
    path = kg.path("Island1", "Island2")
    assert path == []


def test_path_same_entity(kg):
    kg.add_entity("SameEntity", EntityType.CONCEPT)
    path = kg.path("SameEntity", "SameEntity")
    assert len(path) == 1


# ---------------------------------------------------------------------------
# merge_entities
# ---------------------------------------------------------------------------


def test_merge_entities(kg):
    a = kg.add_entity("AcmeCorp", EntityType.ORGANIZATION)
    b = kg.add_entity("Acme Corporation", EntityType.ORGANIZATION)
    merged = kg.merge_entities(a.id, b.id)
    assert merged.id == a.id
    assert b.id not in kg.entities
    assert "Acme Corporation" in merged.aliases
    # Looking up old name should now resolve to the merged entity
    found = kg.find_entity("Acme Corporation")
    assert found is not None and found.id == a.id


# ---------------------------------------------------------------------------
# to_dict export
# ---------------------------------------------------------------------------


def test_to_dict_structure(kg):
    d = kg.to_dict()
    assert "entities" in d
    assert "relations" in d
    for node in d["entities"]:
        assert "id" in node
        assert "name" in node
        assert "type" in node


# ---------------------------------------------------------------------------
# memory.knowledge_graph() factory
# ---------------------------------------------------------------------------


def test_memory_knowledge_graph_factory(tmp_path):
    from agent_memory.manager import Memory

    m = Memory(persist_dir=tmp_path, collection_name="kg_factory")
    m.remember("Who founded Python?", "Guido van Rossum created Python.", type="fact")
    kg = m.knowledge_graph()
    assert isinstance(kg, KnowledgeGraph)
    assert len(kg.entities) > 0

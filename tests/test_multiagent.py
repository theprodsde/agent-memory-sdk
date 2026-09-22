"""Tests for MultiAgentMemory — isolation modes, transfer, broadcast."""
from __future__ import annotations

import pytest

from agent_memory.multiagent import IsolationMode, MultiAgentMemory


@pytest.fixture()
def shared_memory(tmp_path):
    from agent_memory.manager import Memory

    return Memory(persist_dir=tmp_path, collection_name="multi_agent_test")


def make_agent(memory, agent_id: str, isolation: IsolationMode = IsolationMode.NAMESPACED):
    return MultiAgentMemory(memory, agent_id=agent_id, isolation=isolation)


# ---------------------------------------------------------------------------
# NAMESPACED mode (default)
# ---------------------------------------------------------------------------


def test_agent_sees_own_memories(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    b = make_agent(shared_memory, "agent-b")

    a.remember("Python tricks", "Use list comprehensions")
    b.remember("Database indexing", "Use B-tree indexes")

    a_entries = a.list()
    b_entries = b.list()

    a_ids = {e.id for e in a_entries}
    b_ids = {e.id for e in b_entries}

    assert len(a_ids & b_ids) == 0  # no overlap in NAMESPACED mode


def test_global_memory_visible_to_all(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    b = make_agent(shared_memory, "agent-b")

    a.broadcast("Company name", "Acme Corp")

    # Both agents should see the global memory
    assert any(e.query == "Company name" for e in a.list())
    assert any(e.query == "Company name" for e in b.list())


def test_global_flag_in_metadata(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    entry = a.broadcast("global fact", "42")
    assert entry.metadata.get("global") is True
    assert entry.metadata.get("agent_id") == "agent-a"


# ---------------------------------------------------------------------------
# ISOLATED mode
# ---------------------------------------------------------------------------


def test_isolated_cannot_see_other_agents(shared_memory):
    a = make_agent(shared_memory, "agent-a", IsolationMode.ISOLATED)
    b = make_agent(shared_memory, "agent-b", IsolationMode.ISOLATED)

    a.remember("secret knowledge", "only for agent-a")
    b.remember("other knowledge", "only for agent-b")

    a_entries = a.list()
    b_entries = b.list()

    assert all(e.metadata.get("agent_id") == "agent-a" for e in a_entries)
    assert all(e.metadata.get("agent_id") == "agent-b" for e in b_entries)


def test_isolated_global_still_hidden(shared_memory):
    # In ISOLATED mode, even global memories from other agents are hidden
    a = make_agent(shared_memory, "agent-a", IsolationMode.ISOLATED)
    b = make_agent(shared_memory, "agent-b", IsolationMode.ISOLATED)

    a.broadcast("I am global", "but b is isolated")
    b_entries = b.list()
    assert not any(e.query == "I am global" for e in b_entries)


# ---------------------------------------------------------------------------
# SHARED mode
# ---------------------------------------------------------------------------


def test_shared_sees_everything(shared_memory):
    a = make_agent(shared_memory, "agent-a", IsolationMode.SHARED)
    b = make_agent(shared_memory, "agent-b", IsolationMode.SHARED)

    a.remember("a's memory", "from a")
    b.remember("b's memory", "from b")

    a_entries = a.list()
    assert len(a_entries) == 2  # sees both


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def test_transfer_memory(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    make_agent(shared_memory, "agent-b")

    entry = a.remember("transferable knowledge", "transfer me")
    result = a.transfer_memory(entry.id, "agent-b")

    assert result is not None
    assert result.metadata["agent_id"] == "agent-b"
    assert result.metadata["transferred_from"] == "agent-a"


def test_transfer_fails_for_non_owner(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    b = make_agent(shared_memory, "agent-b")

    entry = a.remember("owned by a", "can't steal")
    result = b.transfer_memory(entry.id, "agent-b")
    assert result is None


def test_transfer_nonexistent_memory(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    result = a.transfer_memory("nonexistent-id", "agent-b")
    assert result is None


# ---------------------------------------------------------------------------
# agents_with_memories
# ---------------------------------------------------------------------------


def test_agents_with_memories(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    b = make_agent(shared_memory, "agent-b")

    a.remember("alpha", "a")
    b.remember("beta", "b")

    agents = a.agents_with_memories()
    assert "agent-a" in agents
    assert "agent-b" in agents


def test_agents_with_memories_empty(shared_memory):
    a = make_agent(shared_memory, "agent-a")
    assert a.agents_with_memories() == []


# ---------------------------------------------------------------------------
# Metadata tagging
# ---------------------------------------------------------------------------


def test_agent_id_stamped_in_metadata(shared_memory):
    a = make_agent(shared_memory, "my-agent")
    entry = a.remember("tagged query", "tagged response")
    assert entry.metadata["agent_id"] == "my-agent"


# ---------------------------------------------------------------------------
# Async interface
# ---------------------------------------------------------------------------


async def test_async_remember(shared_memory):
    a = make_agent(shared_memory, "async-agent")
    entry = await a.aremember("async query", "async response")
    assert entry.query == "async query"
    assert entry.metadata["agent_id"] == "async-agent"


async def test_async_resolve(shared_memory):
    a = make_agent(shared_memory, "async-agent")
    await a.aremember("async fact", "stored asynchronously")
    decision = await a.aresolve("async fact")
    assert decision is not None

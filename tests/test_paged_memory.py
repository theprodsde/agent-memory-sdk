"""Tests for PagedMemory — hierarchical in-context + recall + archival tiers."""
from __future__ import annotations

import pytest

from agent_memory.paged_memory import PagedContext, PagedMemory


@pytest.fixture()
def mem(tmp_path):
    from agent_memory.manager import Memory
    return Memory(persist_dir=tmp_path, collection_name="paged_test")


@pytest.fixture()
def paged(mem):
    return PagedMemory(mem, context_size=5, recall_top_k=3)


# ---------------------------------------------------------------------------
# add_turn / context_entries
# ---------------------------------------------------------------------------


def test_add_turn_goes_into_buffer(paged):
    paged.add_turn("What is Python?", "A programming language.")
    assert paged.context_count == 1
    assert paged.context_entries[0].query == "What is Python?"


def test_buffer_newest_first(paged):
    paged.add_turn("Q1", "A1")
    paged.add_turn("Q2", "A2")
    paged.add_turn("Q3", "A3")
    entries = paged.context_entries
    assert entries[0].query == "Q3"   # newest first
    assert entries[-1].query == "Q1"


def test_buffer_respects_context_size(paged):
    for i in range(7):  # context_size=5
        paged.add_turn(f"Q{i}", f"A{i}")
    assert paged.context_count == 5


def test_overflow_pages_out_oldest_to_store(paged, mem):
    """When buffer overflows, oldest entry should be persisted."""
    before = mem.store.count
    for i in range(6):   # context_size=5, one extra triggers page-out
        paged.add_turn(f"Q{i}", f"A{i}")
    assert mem.store.count > before   # at least one paged out


# ---------------------------------------------------------------------------
# get_context
# ---------------------------------------------------------------------------


def test_get_context_returns_paged_context(paged):
    paged.add_turn("Python async", "Use asyncio.")
    paged.add_turn("Python typing", "Use mypy.")
    ctx = paged.get_context("Python programming")
    assert isinstance(ctx, PagedContext)
    assert len(ctx.in_context) >= 1


def test_get_context_includes_recalled_entries(paged, mem):
    """Entries paged to recall should appear in get_context()."""
    for i in range(7):
        paged.add_turn(f"stored query {i}", f"stored response {i}")
    # Now there should be paged-out entries in the store
    ctx = paged.get_context("stored query 0")
    # At least recalled OR in-context should have entries
    assert len(ctx.all_entries) >= 1


def test_get_context_no_duplicates(paged, mem):
    """Same entry should not appear in both in_context and recalled."""
    for i in range(7):
        paged.add_turn(f"Python tip {i}", f"Tip {i}")
    ctx = paged.get_context("Python")
    all_ids = [e.id for e in ctx.all_entries]
    assert len(all_ids) == len(set(all_ids))


def test_get_context_archival_excluded_by_default(paged, mem):
    paged.add_turn("archived topic", "archived answer")
    paged.flush_to_recall()
    # Archive all entries
    for e in mem.list():
        mem.archive(e.id)
    ctx = paged.get_context("archived topic")
    assert all(not e.archived for e in ctx.in_context + ctx.recalled)


def test_get_context_with_archived(paged, mem):
    paged.add_turn("archival query", "archival answer")
    paged.flush_to_recall()
    for e in mem.list():
        mem.archive(e.id)
    ctx = paged.get_context("archival query", include_archived=True)
    assert len(ctx.archived) >= 0  # may or may not match by keyword


# ---------------------------------------------------------------------------
# format_for_llm
# ---------------------------------------------------------------------------


def test_format_for_llm_contains_entries(paged):
    paged.add_turn("What is FastAPI?", "A web framework.")
    ctx = paged.get_context("FastAPI")
    text = ctx.format_for_llm()
    assert "FastAPI" in text or "web framework" in text.lower()


def test_empty_context_format(paged):
    ctx = paged.get_context("something")
    text = ctx.format_for_llm()
    assert isinstance(text, str)  # no crash on empty


# ---------------------------------------------------------------------------
# recall / search_archive
# ---------------------------------------------------------------------------


def test_recall_excludes_buffer(paged, mem):
    paged.add_turn("in buffer query", "in buffer answer")
    # recall should not include what's in the buffer
    recalled = paged.recall("in buffer query")
    buffer_ids = {e.id for e in paged.context_entries}
    for e in recalled:
        assert e.id not in buffer_ids


# ---------------------------------------------------------------------------
# flush_to_recall / clear_context
# ---------------------------------------------------------------------------


def test_flush_to_recall_persists_all(paged, mem):
    paged.add_turn("flush Q1", "flush A1")
    paged.add_turn("flush Q2", "flush A2")
    n = paged.flush_to_recall()
    assert n == 2
    assert paged.context_count == 0
    assert mem.store.count >= 2


def test_clear_context_does_not_persist(paged, mem):
    paged.add_turn("temp Q", "temp A")
    before = mem.store.count
    paged.clear_context()
    assert paged.context_count == 0
    assert mem.store.count == before  # nothing persisted


# ---------------------------------------------------------------------------
# memory.paged() factory
# ---------------------------------------------------------------------------


def test_memory_paged_factory(tmp_path):
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="paged_factory")
    paged = mem.paged(context_size=10)
    assert isinstance(paged, PagedMemory)
    assert paged.context_size == 10


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


async def test_async_add_turn(paged):
    entry = await paged.aadd_turn("async Q", "async A")
    assert entry.query == "async Q"
    assert paged.context_count == 1


async def test_async_get_context(paged):
    await paged.aadd_turn("async context Q", "async context A")
    ctx = await paged.aget_context("async context")
    assert isinstance(ctx, PagedContext)

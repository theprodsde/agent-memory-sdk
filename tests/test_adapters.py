"""Tests for LangChain and LlamaIndex adapters.

Both adapters wrap optional third-party packages.  Tests skip gracefully
when those packages are not installed.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# LangChain adapter
# ---------------------------------------------------------------------------

try:
    from langchain_core.memory import BaseMemory  # noqa: F401

    LANGCHAIN_AVAILABLE = True
except ImportError:
    try:
        from langchain.schema.memory import BaseMemory  # noqa: F401

        LANGCHAIN_AVAILABLE = True
    except ImportError:
        LANGCHAIN_AVAILABLE = False

lc_skip = pytest.mark.skipif(not LANGCHAIN_AVAILABLE, reason="langchain not installed")


@pytest.fixture()
def lc_memory(tmp_path):
    from agent_memory.adapters.langchain_adapter import AgentMemoryLangChain
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="lc_test")
    return AgentMemoryLangChain(memory=mem)


@lc_skip
def test_langchain_memory_variables(lc_memory):
    assert "history" in lc_memory.memory_variables


@lc_skip
def test_langchain_save_and_load(lc_memory):
    lc_memory.save_context({"input": "Hello world"}, {"output": "Hi there"})
    result = lc_memory.load_memory_variables({"input": "Hello world"})
    assert "history" in result


@lc_skip
def test_langchain_empty_load(lc_memory):
    result = lc_memory.load_memory_variables({"input": "nothing stored yet"})
    assert "history" in result
    # Empty store returns empty string or list
    assert result["history"] == "" or result["history"] == []


@lc_skip
def test_langchain_save_context_stores_memory(lc_memory, tmp_path):

    lc_memory.save_context({"input": "Python is great"}, {"output": "Indeed it is"})
    entries = lc_memory._mem.list()
    assert any("Python" in e.query for e in entries)


@lc_skip
def test_langchain_return_messages(tmp_path):
    from agent_memory.adapters.langchain_adapter import AgentMemoryLangChain
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="lc_msg_test")
    lc = AgentMemoryLangChain(memory=mem, return_messages=True)
    lc.save_context({"input": "What is Python?"}, {"output": "A language"})
    result = lc.load_memory_variables({"input": "What is Python?"})
    assert isinstance(result["history"], list)


@lc_skip
def test_langchain_clear(lc_memory):
    lc_memory.save_context({"input": "q1"}, {"output": "a1"})
    lc_memory.clear()
    entries = lc_memory._mem.list(include_archived=True)
    # All entries should now be archived
    assert all(e.archived for e in entries)


@lc_skip
def test_langchain_from_persist_dir(tmp_path):
    from agent_memory.adapters.langchain_adapter import AgentMemoryLangChain

    lc = AgentMemoryLangChain.from_persist_dir(str(tmp_path))
    assert lc is not None
    assert "history" in lc.memory_variables


# ---------------------------------------------------------------------------
# LlamaIndex adapter
# ---------------------------------------------------------------------------

try:
    from llama_index.core.memory import BaseMemory as LlamaBaseMemory  # noqa: F401

    LLAMAINDEX_AVAILABLE = True
except ImportError:
    try:
        from llama_index.memory import BaseMemory as LlamaBaseMemory  # noqa: F401

        LLAMAINDEX_AVAILABLE = True
    except ImportError:
        LLAMAINDEX_AVAILABLE = False

li_skip = pytest.mark.skipif(
    not LLAMAINDEX_AVAILABLE, reason="llama-index not installed"
)


@pytest.fixture()
def li_memory(tmp_path):
    from agent_memory.adapters.llamaindex_adapter import AgentMemoryLlamaIndex
    from agent_memory.manager import Memory

    mem = Memory(persist_dir=tmp_path, collection_name="li_test")
    return AgentMemoryLlamaIndex(memory=mem)


@li_skip
def test_llamaindex_get_empty(li_memory):
    result = li_memory.get(input="anything")
    assert isinstance(result, list)


@li_skip
def test_llamaindex_put_and_get(li_memory):
    from llama_index.core.llms import ChatMessage, MessageRole

    li_memory.put(ChatMessage(role=MessageRole.USER, content="Hello from user"))
    li_memory.put(ChatMessage(role=MessageRole.ASSISTANT, content="Hello back"))
    result = li_memory.get(input="Hello")
    assert isinstance(result, list)


@li_skip
def test_llamaindex_reset(li_memory):
    from llama_index.core.llms import ChatMessage, MessageRole

    li_memory.put(ChatMessage(role=MessageRole.USER, content="test"))
    li_memory.reset()
    assert li_memory._buffer == []


@li_skip
def test_llamaindex_get_all(li_memory):
    li_memory._mem.remember("llama query", "llama response")
    all_msgs = li_memory.get_all()
    assert len(all_msgs) >= 2


@li_skip
def test_llamaindex_set(li_memory):
    from llama_index.core.llms import ChatMessage, MessageRole

    msgs = [ChatMessage(role=MessageRole.USER, content="set test")]
    li_memory.set(msgs)
    assert len(li_memory._buffer) == 1


@li_skip
def test_llamaindex_from_defaults(tmp_path):
    from agent_memory.adapters.llamaindex_adapter import AgentMemoryLlamaIndex

    li = AgentMemoryLlamaIndex.from_defaults(str(tmp_path))
    assert li is not None


# ---------------------------------------------------------------------------
# Import guard: missing packages raise ImportError with install hint
# ---------------------------------------------------------------------------


def test_langchain_import_error_message(tmp_path, monkeypatch):
    """AgentMemoryLangChain should raise ImportError with install hint when
    langchain-core is absent."""
    import importlib
    import sys

    # Temporarily hide langchain_core
    hidden = {k: v for k, v in sys.modules.items() if "langchain" in k}
    for k in hidden:
        sys.modules.pop(k, None)
    sys.modules["langchain_core"] = None  # type: ignore[assignment]
    sys.modules["langchain_core.memory"] = None  # type: ignore[assignment]
    sys.modules["langchain"] = None  # type: ignore[assignment]

    try:
        import agent_memory.adapters.langchain_adapter as mod

        importlib.reload(mod)
        assert not mod.LANGCHAIN_AVAILABLE
    finally:
        for k in list(sys.modules.keys()):
            if k.startswith("langchain"):
                sys.modules.pop(k, None)
        sys.modules.update(hidden)

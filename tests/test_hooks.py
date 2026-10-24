"""Tests for the pre-inference memory gate (``agent-memory hook``).

Covers the three things that can break a hook in a way the user feels:
the JSON contract the host parses, the fail-open guarantee, and the gate's
threshold calibration (exact repeats gate, paraphrases and unrelated queries
must not).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory import Memory
from agent_memory.hooks import (
    HOOK_EVENT,
    MAX_CONTEXT_CHARS,
    HookConfig,
    build_output,
    decision_relevance,
    run_hook,
    should_skip,
)
from agent_memory.hooks import main as hook_main

# Mirrors the calibration fixture: a small project knowledge base of the kind a
# coding session accumulates.
KNOWLEDGE_BASE = [
    ("What test runner does this repo use?",
     "pytest via `make test`; lint and types via `make check`.", "fact"),
    ("Where does the core code live?",
     "agent_memory/ holds the package, tests/ holds the tests.", "fact"),
    ("What are the code conventions here?",
     "from __future__ import annotations at the top, full type hints.", "preference"),
    ("Should I commit changes when done?",
     "No — leave changes in the working tree for the maintainer.", "preference"),
    ("How do I run the dashboard?",
     "Install the [dashboard] extra, then run agent-memory-dashboard.", "fact"),
    ("What backends are supported?",
     "SQLite, ChromaDB, Redis, PostgreSQL, Qdrant.", "fact"),
    ("How do I reset my password?",
     "Settings → Security → Reset Password.", "conversation"),
    ("What is the API rate limit?",
     "1000 requests/min per API key.", "fact"),
]

PARAPHRASES = [
    "how do I run the tests here",
    "where is the main package code",
    "what style rules does this project follow",
    "do you want me to commit",
    "how do I start the dashboard",
    "which databases can I use as a store",
    "i forgot my password",
    "how many requests can I send per minute",
]

UNRELATED = [
    "what is the weather in Tokyo",
    "write a python CSV parser for me",
    "deploy the ios app to testflight",
    "refactor the billing module please",
    "what is my current git branch name",
    "explain the RRF fusion maths to me",
    "who maintains this project these days",
]


@pytest.fixture(scope="module")
def store(tmp_path_factory: pytest.TempPathFactory) -> Memory:
    path = tmp_path_factory.mktemp("hook_store")
    memory = Memory(persist_dir=str(path), collection_name="hooks")
    for query, response, type_ in KNOWLEDGE_BASE:
        memory.remember(query, response, type=type_, tags=["project"])
    return memory


def _payload(prompt: str) -> dict:
    return {
        "session_id": "test-session",
        "transcript_path": "/tmp/transcript.jsonl",
        "cwd": "/tmp/project",
        "permission_mode": "default",
        "hook_event_name": HOOK_EVENT,
        "prompt": prompt,
    }


# ----------------------------------------------------------------------
# Prefilter
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt",
    [
        "",
        "   ",
        "ok",
        "yes please",
        "/review the diff",
        "!git status",
        "What test runner does this repo use? no-memory",
    ],
)
def test_prefilter_skips_unresolvable_prompts(prompt: str) -> None:
    assert should_skip(prompt, HookConfig()) is not None


def test_prefilter_skips_long_pasted_prompts() -> None:
    config = HookConfig()
    assert should_skip("x" * (config.max_prompt_chars + 1), config) is not None


def test_prefilter_allows_a_real_question() -> None:
    assert should_skip("What test runner does this repo use?", HookConfig()) is None


def test_prefilter_does_not_construct_a_store() -> None:
    """A skipped prompt must not pay for the store or the embedder.

    Guards the latency budget: the hook runs before every prompt, so the skip
    path has to stay at roughly a bare interpreter start. Passing a store that
    explodes on use proves nothing touched it.
    """

    class _Exploding:
        def resolve(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("prefilter should have skipped before resolving")

    for prompt in ("", "ok", "/review", "!ls", "x" * 5000):
        assert run_hook(_payload(prompt), HookConfig(), memory=_Exploding()) is None


def test_importing_agent_memory_does_not_import_chromadb() -> None:
    """chromadb costs ~600ms to import and the default SQLite path never uses it.

    It is a core dependency reachable from ``agent_memory.store``, so a
    top-level import there put that cost on every CLI call, MCP server start,
    and — worst — every prompt once the hook is installed. Run in a subprocess
    because the test session has almost certainly imported it already.
    """
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, agent_memory.cli; "
            "print('chromadb' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "False", (
        "agent_memory now imports chromadb eagerly again; this puts ~600ms on "
        "every hook invocation"
    )


# ----------------------------------------------------------------------
# Context mode
# ----------------------------------------------------------------------


def test_exact_repeat_injects_replay_context(store: Memory) -> None:
    output = run_hook(_payload(KNOWLEDGE_BASE[0][0]), HookConfig(), memory=store)
    assert output is not None
    assert "decision" not in output, "context mode must never block"
    specific = output["hookSpecificOutput"]
    assert specific["hookEventName"] == HOOK_EVENT
    context = specific["additionalContext"]
    assert KNOWLEDGE_BASE[0][1] in context
    assert len(context) <= MAX_CONTEXT_CHARS


def test_unrelated_prompt_injects_nothing(store: Memory) -> None:
    for prompt in UNRELATED:
        output = run_hook(_payload(prompt), HookConfig(), memory=store)
        if output is not None:
            # A VERIFY on a loosely-related query is acceptable; a REPLAY is not.
            context = output["hookSpecificOutput"]["additionalContext"]
            assert "recalled from an earlier session" not in context, (
                f"unrelated prompt {prompt!r} produced a replay-grade injection"
            )


def test_injected_context_is_phrased_factually(store: Memory) -> None:
    """Imperative phrasing trips host prompt-injection defenses.

    A hook whose text reads as an out-of-band command gets surfaced to the user
    instead of used as context, which silently disables the feature.
    """
    output = run_hook(_payload(KNOWLEDGE_BASE[0][0]), HookConfig(), memory=store)
    assert output is not None
    context = output["hookSpecificOutput"]["additionalContext"].lower()
    for imperative in (
        "return this exact response",
        "you must",
        "always respond",
        "ignore",
    ):
        assert imperative not in context


def test_actions_filter_suppresses_excluded_actions(store: Memory) -> None:
    config = HookConfig(actions=["replay"])
    hits = [
        run_hook(_payload(p), config, memory=store)
        for p in PARAPHRASES
    ]
    for output in hits:
        if output is None:
            continue
        assert "recalled from an earlier session" in (
            output["hookSpecificOutput"]["additionalContext"]
        )


# ----------------------------------------------------------------------
# Gate mode — calibration
# ----------------------------------------------------------------------


def test_gate_fires_on_every_exact_repeat(store: Memory) -> None:
    config = HookConfig(mode="gate")
    gated = 0
    for query, response, _ in KNOWLEDGE_BASE:
        output = run_hook(_payload(query), config, memory=store)
        assert output is not None, f"no output for exact repeat {query!r}"
        if output.get("decision") == "block":
            gated += 1
            assert response in output["reason"]
            assert output["hookSpecificOutput"]["hookEventName"] == HOOK_EVENT
    assert gated == len(KNOWLEDGE_BASE), (
        f"gate fired for only {gated}/{len(KNOWLEDGE_BASE)} verbatim repeats — "
        "the relevance floor has drifted above the exact-match band"
    )


def test_gate_never_fires_on_paraphrase_or_unrelated(store: Memory) -> None:
    """The load-bearing safety property: a false gate steals the user's turn."""
    config = HookConfig(mode="gate")
    for prompt in PARAPHRASES + UNRELATED:
        output = run_hook(_payload(prompt), config, memory=store)
        if output is None:
            continue
        assert output.get("decision") != "block", (
            f"gate blocked non-exact prompt {prompt!r}"
        )


def test_gate_degrades_to_context_when_relevance_is_short(store: Memory) -> None:
    """Failing the gate must still inject context, not fall silent."""
    config = HookConfig(mode="gate", gate_min_relevance=0.99)
    output = run_hook(_payload(KNOWLEDGE_BASE[0][0]), config, memory=store)
    assert output is not None
    assert "decision" not in output
    assert KNOWLEDGE_BASE[0][1] in output["hookSpecificOutput"]["additionalContext"]


def test_gate_message_names_the_bypass_token(store: Memory) -> None:
    config = HookConfig(mode="gate")
    output = run_hook(_payload(KNOWLEDGE_BASE[0][0]), config, memory=store)
    assert output is not None and output.get("decision") == "block"
    assert config.bypass_token in output["reason"]


def test_bypass_token_reaches_the_model(store: Memory) -> None:
    config = HookConfig(mode="gate")
    prompt = f"{KNOWLEDGE_BASE[0][0]} {config.bypass_token}"
    assert run_hook(_payload(prompt), config, memory=store) is None


def test_relevance_separates_exact_from_the_rest(store: Memory) -> None:
    """Pin the measured separation the default floor relies on."""
    exact = [decision_relevance(store.resolve(q)) for q, _, _ in KNOWLEDGE_BASE]
    others = [
        decision_relevance(store.resolve(q)) for q in PARAPHRASES + UNRELATED
    ]
    assert min(exact) > max(others), (
        "exact repeats no longer separate from paraphrases/unrelated queries; "
        "the gate's relevance floor needs recalibrating"
    )
    assert min(exact) > HookConfig().gate_min_relevance


def test_unknown_relevance_never_opens_the_gate() -> None:
    class _Bare:
        action = type("A", (), {"value": "replay"})()
        response = "cached answer"
        confidence = 0.99
        scores: dict = {}
        context: list = []
        memory = None

    assert decision_relevance(_Bare()) == 0.0
    assert build_output(_Bare(), HookConfig(mode="gate")) is not None
    assert build_output(_Bare(), HookConfig(mode="gate")).get("decision") != "block"


# ----------------------------------------------------------------------
# Event routing and fail-open
# ----------------------------------------------------------------------


def test_other_hook_events_are_ignored(store: Memory) -> None:
    payload = _payload(KNOWLEDGE_BASE[0][0])
    payload["hook_event_name"] = "PreToolUse"
    assert run_hook(payload, HookConfig(), memory=store) is None


@pytest.mark.parametrize(
    "stdin_text",
    ["", "   ", "not json at all", "[1, 2, 3]", '"a bare string"', "{"],
)
def test_main_fails_open_on_bad_stdin(
    stdin_text: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr("sys.stdin", _FakeStdin(stdin_text))
    assert hook_main(["user-prompt-submit", "--data-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_when_the_store_is_unusable(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    """An unreadable store must cost the user nothing but the hook's latency."""
    broken = tmp_path / "not-a-directory"
    broken.write_text("this is a file, not a persist dir")
    monkeypatch.setattr(
        "sys.stdin", _FakeStdin(json.dumps(_payload("What test runner does this use?")))
    )
    assert hook_main(["user-prompt-submit", "--data-dir", str(broken)]) == 0
    assert capsys.readouterr().out == ""


def test_main_fails_open_when_resolve_raises(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    def _boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("backend exploded mid-resolve")

    monkeypatch.setattr("agent_memory.hooks.run_hook", _boom)
    monkeypatch.setattr(
        "sys.stdin", _FakeStdin(json.dumps(_payload("What test runner does this use?")))
    )
    assert hook_main(["user-prompt-submit", "--data-dir", str(tmp_path)]) == 0
    assert capsys.readouterr().out == ""


def test_main_emits_valid_json_on_a_hit(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    tmp_path: Path,
) -> None:
    memory = Memory(persist_dir=str(tmp_path), collection_name="main_hit")
    memory.remember(
        "What test runner does this repo use?",
        "pytest via `make test`.",
        type="fact",
    )
    monkeypatch.setattr(
        "sys.stdin",
        _FakeStdin(json.dumps(_payload("What test runner does this repo use?"))),
    )
    assert hook_main(
        ["user-prompt-submit", "--data-dir", str(tmp_path),
         "--collection", "main_hit"]
    ) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert parsed["hookSpecificOutput"]["hookEventName"] == HOOK_EVENT
    assert "pytest" in parsed["hookSpecificOutput"]["additionalContext"]


class _FakeStdin:
    def __init__(self, text: str) -> None:
        self._text = text

    def read(self) -> str:
        return self._text


# ----------------------------------------------------------------------
# Config resolution
# ----------------------------------------------------------------------


def test_env_configures_the_hook() -> None:
    config = HookConfig.from_env(
        {
            "AGENT_MEMORY_HOOK_MODE": "gate",
            "AGENT_MEMORY_DIR": "/tmp/store",
            "AGENT_MEMORY_HOOK_GATE_MIN_RELEVANCE": "0.97",
            "AGENT_MEMORY_HOOK_ACTIONS": "replay,restore",
            "AGENT_MEMORY_HOOK_SCOPE": "project,team",
            "AGENT_MEMORY_HOOK_SEMANTIC": "off",
            "AGENT_MEMORY_HOOK_TOP_K": "7",
        }
    )
    assert config.mode == "gate"
    assert config.data_dir == "/tmp/store"
    assert config.gate_min_relevance == 0.97
    assert config.actions == ["replay", "restore"]
    assert config.scope == ["project", "team"]
    assert config.semantic is False
    assert config.top_k == 7


def test_malformed_env_values_fall_back_to_defaults() -> None:
    config = HookConfig.from_env(
        {
            "AGENT_MEMORY_HOOK_GATE_MIN_RELEVANCE": "not-a-number",
            "AGENT_MEMORY_HOOK_TOP_K": "",
        }
    )
    assert config.gate_min_relevance == HookConfig().gate_min_relevance
    assert config.top_k == HookConfig().top_k


def test_explicit_overrides_beat_the_environment() -> None:
    config = HookConfig.from_env(
        {"AGENT_MEMORY_HOOK_MODE": "gate"}, mode="context", top_k=None
    )
    assert config.mode == "context"
    assert config.top_k == HookConfig().top_k, "None overrides must not clobber"

"""Pre-inference memory gate for coding agents that support lifecycle hooks.

Why this module exists
----------------------
The MCP server (:mod:`mcp_server.server`) exposes ``resolve_memory`` as a
*tool*.  A tool runs **after** the host has already made an inference call to
decide to call it, and the host makes another to relay the result.  So a REPLAY
delivered through MCP never removes an LLM call — at best it collapses the
grep/read/re-derive loop that would otherwise follow.

A ``UserPromptSubmit`` hook runs **before** the host's first inference call for
a turn.  That is the only place outside the model where a stored answer can
actually replace a model call rather than merely inform one.  This module
implements that gate.

Two modes, because they buy different things
-------------------------------------------
``context`` (default)
    REPLAY / RESTORE / VERIFY are injected as ``additionalContext`` next to the
    user's prompt.  The inference call still happens.  What you save is the
    re-derivation work: the agent starts the turn already knowing the
    convention instead of discovering it with four tool calls.

``gate`` (opt-in)
    A high-confidence REPLAY *blocks* the prompt and answers the user from the
    store.  This is the only configuration that removes an LLM call outright,
    and it is off by default on purpose: a wrong REPLAY here means the user's
    prompt never reaches the model at all.  Failure is visible (the user sees
    the memory answer and can retry with the bypass token) but it is still a
    hijacked turn, so it demands a much higher confidence floor than REPLAY's
    own threshold.

Operational rules this module holds to
--------------------------------------
* **Fail open, always.**  A memory layer must never be the reason a prompt
  fails.  Every error path — bad stdin, unreadable store, missing backend,
  import failure — exits 0 with no output, and the prompt proceeds untouched.
* **Cheap on a miss.**  The prefilter runs before :class:`~agent_memory.Memory`
  is imported, so prompts that cannot usefully match cost a bare interpreter
  start rather than the ~0.9s import plus ~0.5s embedder load.
* **Factual phrasing.**  Injected text is written as statements about the store
  ("A stored memory matches…"), not as instructions to the model.  Text framed
  as an out-of-band command trips prompt-injection defenses and gets surfaced
  to the user instead of used as context.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any

HOOK_EVENT = "UserPromptSubmit"

#: Hosts cap injected strings at 10k characters and spill the remainder to a
#: file. Staying comfortably under the cap keeps the memory inline, where the
#: model actually reads it, instead of behind a path it has to choose to open.
MAX_CONTEXT_CHARS = 8000

#: Prompts longer than this are pasted logs, diffs, or multi-paragraph specs.
#: They are not recall-shaped questions, and embedding them wastes the latency
#: budget on every turn for a match that will not come.
DEFAULT_MAX_PROMPT_CHARS = 2000

#: Below this a prompt is "ok", "yes", "continue" — conversational
#: continuations that carry no retrievable intent of their own.
DEFAULT_MIN_PROMPT_CHARS = 12

#: Relevance floor for ``gate`` mode — the hybrid semantic/keyword signal, not
#: the composite decision score.
#:
#: Gating on the composite score looks natural and is wrong twice over. First,
#: it is unreachable: for a memory that has not been reused the composite
#: ceiling is ``0.55·relevance + 0.15·recency + 0.20·confidence`` = 0.90 at
#: perfect relevance, and measured exact matches land at 0.87–0.88, so a
#: "conservative" 0.90 composite floor never fires at all. Second, 35% of that
#: score is recency and stored confidence, which are near-identical for every
#: fresh candidate and therefore carry no evidence that this memory answers
#: *this* query — and the usage term means the same memory clears a composite
#: floor more easily the more it is reused, so the gate would quietly grow more
#: aggressive over a store's lifetime.
#:
#: Relevance has neither problem: it is the discriminating signal and it is
#: stable across reuse. Measured on a seeded store (8 memories, 8 verbatim
#: repeats / 8 paraphrases / 8 unrelated queries):
#:
#: * verbatim repeats:  0.9375 – 0.9692
#: * paraphrases:       0.4748 – 0.8134
#: * unrelated queries: 0.3408 – 0.7543
#:
#: Any floor in [0.90, 0.93] gated 8/8 repeats with zero false gates. 0.90 is
#: the default because it keeps 0.087 of headroom above the highest non-exact
#: match — the direction that matters, since failing to gate degrades to
#: ``context`` mode while a false gate takes the user's turn.
DEFAULT_GATE_MIN_RELEVANCE = 0.90

#: Typing this anywhere in a prompt disables the gate for that turn. The escape
#: hatch matters most in ``gate`` mode, where a user who disagrees with a
#: replayed answer needs a way to reach the model.
DEFAULT_BYPASS_TOKEN = "no-memory"


@dataclass
class HookConfig:
    """Resolved hook settings.

    Built by :meth:`from_env`, which layers environment variables over these
    defaults so the same hook entry in ``settings.json`` can be retuned per
    project without editing the command line.
    """

    mode: str = "context"
    data_dir: str = ".agent_memory"
    backend: str = "sqlite"
    collection: str | None = None
    top_k: int = 3
    scope: list[str] | None = None
    gate_min_relevance: float = DEFAULT_GATE_MIN_RELEVANCE
    max_prompt_chars: int = DEFAULT_MAX_PROMPT_CHARS
    min_prompt_chars: int = DEFAULT_MIN_PROMPT_CHARS
    bypass_token: str = DEFAULT_BYPASS_TOKEN
    semantic: bool = True
    #: Actions that produce injected context in ``context`` mode. VERIFY is
    #: included because a stale-but-relevant memory is still worth naming —
    #: the injected text tells the agent to check it rather than trust it.
    actions: list[str] = field(
        default_factory=lambda: ["replay", "restore", "verify"]
    )

    @classmethod
    def from_env(
        cls, env: dict[str, str] | None = None, **overrides: Any
    ) -> HookConfig:
        """Build a config from ``AGENT_MEMORY_HOOK_*`` vars, then explicit overrides.

        Precedence is overrides > environment > dataclass defaults, so CLI flags
        win over a project's exported settings.
        """
        src = os.environ if env is None else env
        cfg = cls()

        def _str(key: str, current: str) -> str:
            return src.get(key, current)

        def _float(key: str, current: float) -> float:
            try:
                return float(src[key]) if key in src else current
            except (TypeError, ValueError):
                return current

        def _int(key: str, current: int) -> int:
            try:
                return int(src[key]) if key in src else current
            except (TypeError, ValueError):
                return current

        cfg.mode = _str("AGENT_MEMORY_HOOK_MODE", cfg.mode).strip().lower()
        cfg.data_dir = _str(
            "AGENT_MEMORY_DIR", _str("AGENT_MEMORY_HOOK_DATA_DIR", cfg.data_dir)
        )
        cfg.backend = _str("AGENT_MEMORY_HOOK_BACKEND", cfg.backend)
        cfg.collection = src.get("AGENT_MEMORY_COLLECTION", cfg.collection)
        cfg.top_k = _int("AGENT_MEMORY_HOOK_TOP_K", cfg.top_k)
        cfg.gate_min_relevance = _float(
            "AGENT_MEMORY_HOOK_GATE_MIN_RELEVANCE", cfg.gate_min_relevance
        )
        cfg.max_prompt_chars = _int(
            "AGENT_MEMORY_HOOK_MAX_PROMPT_CHARS", cfg.max_prompt_chars
        )
        cfg.min_prompt_chars = _int(
            "AGENT_MEMORY_HOOK_MIN_PROMPT_CHARS", cfg.min_prompt_chars
        )
        cfg.bypass_token = _str("AGENT_MEMORY_HOOK_BYPASS_TOKEN", cfg.bypass_token)

        if "AGENT_MEMORY_HOOK_SEMANTIC" in src:
            cfg.semantic = src["AGENT_MEMORY_HOOK_SEMANTIC"].strip().lower() not in (
                "0",
                "false",
                "no",
                "off",
            )
        if "AGENT_MEMORY_HOOK_SCOPE" in src:
            scopes = [s.strip() for s in src["AGENT_MEMORY_HOOK_SCOPE"].split(",")]
            cfg.scope = [s for s in scopes if s] or None
        if "AGENT_MEMORY_HOOK_ACTIONS" in src:
            acts = [
                a.strip().lower()
                for a in src["AGENT_MEMORY_HOOK_ACTIONS"].split(",")
            ]
            cfg.actions = [a for a in acts if a]

        for key, value in overrides.items():
            if value is not None:
                setattr(cfg, key, value)
        return cfg


# ----------------------------------------------------------------------
# Prefilter — runs before Memory is imported
# ----------------------------------------------------------------------


def should_skip(prompt: str, config: HookConfig) -> str | None:
    """Return a reason to skip resolution, or ``None`` to proceed.

    Kept free of any :mod:`agent_memory` import so a skipped prompt never pays
    the package import or the embedder load. On a session where most prompts
    are long or conversational, this is the difference between adding ~50ms and
    adding ~1.4s to every turn.
    """
    text = prompt.strip()
    if not text:
        return "empty prompt"
    if config.bypass_token and config.bypass_token.lower() in text.lower():
        return f"bypass token {config.bypass_token!r} present"
    if text.startswith("/"):
        # Slash commands expand through UserPromptExpansion; the raw "/foo"
        # text carries no intent worth matching against the store.
        return "slash command"
    if text.startswith("!"):
        return "bash passthrough"
    if len(text) < config.min_prompt_chars:
        return f"prompt shorter than {config.min_prompt_chars} chars"
    if len(text) > config.max_prompt_chars:
        return f"prompt longer than {config.max_prompt_chars} chars"
    return None


# ----------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def render_replay_context(decision: Any) -> str:
    """Describe a REPLAY match as project information, not as an instruction."""
    entry = decision.memory
    lines = [
        "## Agent Memory — recalled from an earlier session",
        "",
        f"A stored memory closely matches this request "
        f"(confidence {decision.confidence:.2f}).",
        "",
    ]
    if entry is not None:
        lines.append(f'Question recorded earlier: "{entry.query}"')
        lines.append(f"Answer recorded earlier: {entry.response}")
        lines.append("")
        lines.append(
            f"Stored {entry.created_at:%Y-%m-%d}, reused {entry.access_count}×, "
            f"type {entry.type.value}, stored confidence {entry.confidence:.0%}."
        )
    elif decision.response:
        lines.append(f"Answer recorded earlier: {decision.response}")
    lines.append("")
    lines.append(
        "This is recalled knowledge from a previous session, not a fresh "
        "observation of the current working tree. Where it disagrees with what "
        "the repository actually contains now, the repository is correct."
    )
    return "\n".join(lines)


def render_restore_context(decision: Any, *, verify: bool = False) -> str:
    """Describe RESTORE/VERIFY matches as candidate prior knowledge."""
    heading = (
        "## Agent Memory — related memories needing verification"
        if verify
        else "## Agent Memory — related memories from earlier sessions"
    )
    lines = [
        heading,
        "",
        f"{len(decision.context)} stored "
        f"{'memory' if len(decision.context) == 1 else 'memories'} partially "
        f"match this request (best confidence {decision.confidence:.2f}). "
        "They are related but not close enough to answer it directly.",
        "",
    ]
    for i, result in enumerate(decision.context, 1):
        entry = result.entry
        lines.append(
            f"{i}. (score {result.final_score:.2f}, type {entry.type.value}, "
            f"stored {entry.created_at:%Y-%m-%d})"
        )
        lines.append(f'   Earlier question: "{_truncate(entry.query, 200)}"')
        lines.append(f"   Earlier answer: {_truncate(entry.response, 600)}")
    lines.append("")
    if verify:
        lines.append(
            "These memories are of a type that goes stale (facts, tool output, "
            "workflows) and scored below the verification threshold. Confirm "
            "them against the current repository or tooling before relying on "
            "them."
        )
    else:
        lines.append(
            "This is recalled knowledge from previous sessions, offered as "
            "background. It has not been re-checked against the current state "
            "of the repository."
        )
    return "\n".join(lines)


def render_gate_message(decision: Any, config: HookConfig) -> str:
    """User-facing text for a gated turn, where no model call will be made."""
    entry = decision.memory
    lines = [
        f"✅ Answered from Agent Memory (confidence {decision.confidence:.2f}) "
        f"— no model call was made.",
        "",
        decision.response or "",
    ]
    if entry is not None:
        lines += [
            "",
            f"— matched \"{_truncate(entry.query, 120)}\", stored "
            f"{entry.created_at:%Y-%m-%d}, reused {entry.access_count}×.",
        ]
    lines += [
        "",
        f"To ask the model instead, resend including “{config.bypass_token}”.",
    ]
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Core
# ----------------------------------------------------------------------


def decision_relevance(decision: Any) -> float:
    """Return the hybrid semantic/keyword relevance behind a decision.

    This is the same quantity :meth:`DefaultPolicy.select_action` calls
    ``relevance`` and compares against ``exact_relevance``: the blend
    ``max(0.7·semantic + 0.3·keyword, 0.7·keyword + 0.3·semantic)``, exposed on
    the decision as ``scores["semantic_score"]``. Falls back to recomputing it
    from the top context entry if the score breakdown is absent, and to 0.0 if
    neither is available — an unknown relevance must never open the gate.
    """
    scores = getattr(decision, "scores", None) or {}
    value = scores.get("semantic_score")
    if isinstance(value, (int, float)):
        return float(value)

    context = getattr(decision, "context", None) or []
    if context:
        top = context[0]
        semantic = float(getattr(top, "semantic_score", 0.0) or 0.0)
        keyword = float(getattr(top, "keyword_score", 0.0) or 0.0)
        return max(
            0.7 * semantic + 0.3 * keyword,
            0.7 * keyword + 0.3 * semantic,
        )
    return 0.0


def _load_memory(config: HookConfig) -> Any:
    """Import and construct :class:`~agent_memory.Memory` lazily."""
    from agent_memory import Memory

    kwargs: dict[str, Any] = {
        "persist_dir": config.data_dir,
        "backend": config.backend,
    }
    if config.collection:
        kwargs["collection_name"] = config.collection
    if not config.semantic:
        kwargs["enable_embeddings"] = False
    return Memory(**kwargs)


def build_output(decision: Any, config: HookConfig) -> dict[str, Any] | None:
    """Translate a decision into hook JSON, or ``None`` to stay silent.

    Returning ``None`` is the common case — NONE decisions, and actions the
    operator excluded — and it matters that it produces no output at all rather
    than an empty context block, which would still cost the model tokens to
    read and dismiss.
    """
    action = decision.action.value

    # Gate mode needs two independent keys turned at once. The policy must
    # already have chosen REPLAY — which means the composite cleared 0.85 and
    # the top hit beat its nearest disagreeing runner-up — *and* the match must
    # be near-exact on the discriminating signal. Either alone is too weak to
    # justify answering without the model: REPLAY fires on strong paraphrases,
    # and high relevance alone ignores stale or low-confidence entries.
    if (
        config.mode == "gate"
        and action == "replay"
        and decision.response
        and decision_relevance(decision) >= config.gate_min_relevance
    ):
        message = render_gate_message(decision, config)
        return {
            "decision": "block",
            "reason": message,
            "systemMessage": message,
            # Documented for UserPromptSubmit but its nesting is not pinned
            # down in the reference, so it is sent at top level. If the host
            # ignores it the only effect is that the block message also echoes
            # the user's prompt — cosmetic, not behavioural.
            "suppressOriginalPrompt": True,
            "hookSpecificOutput": {"hookEventName": HOOK_EVENT},
        }

    if action not in config.actions:
        return None

    if action == "replay":
        context = render_replay_context(decision)
    elif action == "restore":
        context = render_restore_context(decision) if decision.context else ""
    elif action == "verify":
        context = (
            render_restore_context(decision, verify=True) if decision.context else ""
        )
    else:
        return None

    if not context.strip():
        return None

    return {
        "hookSpecificOutput": {
            "hookEventName": HOOK_EVENT,
            "additionalContext": _truncate(context, MAX_CONTEXT_CHARS),
        }
    }


def run_hook(
    payload: dict[str, Any],
    config: HookConfig,
    *,
    memory: Any | None = None,
) -> dict[str, Any] | None:
    """Resolve one hook payload into hook JSON, or ``None`` to stay silent.

    *memory* is injectable so tests can exercise the decision-to-JSON contract
    against a seeded store without going through the environment.

    Raises nothing for expected conditions; :func:`main` is responsible for
    converting unexpected failures into a silent pass.
    """
    event = payload.get("hook_event_name")
    if event and event != HOOK_EVENT:
        return None

    prompt = payload.get("prompt") or ""
    if should_skip(prompt, config):
        return None

    mem = memory if memory is not None else _load_memory(config)
    decision = mem.resolve(prompt, top_k=config.top_k, scope=config.scope)
    return build_output(decision, config)


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``agent-memory hook``.

    Always returns 0. A blocking decision is expressed through the JSON
    ``decision`` field rather than exit code 2, because exit 2 routes the
    message through stderr where it is shown as an error — the wrong framing
    for a memory that answered the question successfully.
    """
    from agent_memory.cli import build_hook_config  # local: keeps import cheap

    try:
        config = build_hook_config(argv)
    except SystemExit:
        raise
    except Exception:
        return 0

    try:
        raw = sys.stdin.read()
    except Exception:
        return 0

    try:
        payload = json.loads(raw) if raw.strip() else {}
        if not isinstance(payload, dict):
            return 0
    except (ValueError, TypeError):
        return 0

    try:
        output = run_hook(payload, config)
    except Exception:
        # Fail open: a memory layer is never worth a failed prompt.
        return 0

    if output:
        sys.stdout.write(json.dumps(output))
    return 0

# Pre-inference hook (coding agents)

The MCP server exposes `resolve_memory` as a **tool**. A tool runs after the
host has already spent an inference call deciding to call it, and spends another
relaying the result. So a REPLAY delivered over MCP never removes a model call —
at best it collapses the grep/read/re-derive loop that would have followed.

A `UserPromptSubmit` hook runs **before** the host's first inference call for a
turn. That is the only place outside the model where a stored answer can replace
a model call instead of merely informing one.

```bash
# Inject matching memory alongside every prompt (safe default)
agent-memory hook-install

# Let a near-exact REPLAY answer without calling the model at all
agent-memory hook-install --mode gate --lexical
```

Restart the session, or run `/hooks`, to pick it up.

---

## The two modes buy different things

| | `context` (default) | `gate` (opt-in) |
|---|---|---|
| What it emits | `additionalContext` next to the prompt | `decision: "block"` + the stored answer |
| Model call happens? | Yes | **No** |
| What you save | the re-derivation loop | the whole turn |
| Risk if the match is wrong | a stale hint the agent can override | the prompt never reaches the model |
| Fires on | REPLAY, RESTORE, VERIFY | near-exact REPLAY only |

`context` is the default because its failure mode is recoverable: a wrong hint
sits next to the prompt and the agent can contradict it from the actual repo.
`gate` mode's failure mode is not — the user's prompt is erased and answered from
the store. That is why it is opt-in, keyed on a stricter signal, and escapable.

### Escaping the gate

Any prompt containing `no-memory` (configurable) bypasses the hook entirely for
that turn. Gated replies name the token so the user always has a way back to the
model:

```
✅ Answered from Agent Memory (confidence 0.89) — no model call was made.

pytest via `make test`; lint and types via `make check`.

— matched "What test runner does this repo use?", stored 2026-09-29, reused 4×.

To ask the model instead, resend including "no-memory".
```

---

## Why the gate keys on relevance, not confidence

Gating on `decision.confidence` is the obvious choice and it is wrong twice
over.

**It is unreachable.** For a memory that has not been reused, the composite
ceiling is `0.55·relevance + 0.15·recency + 0.20·confidence + 0.10·usage`, which
tops out at 0.90 with perfect relevance and zero reuse. Measured verbatim
repeats land at **0.87–0.88**. A "conservative" 0.90 confidence floor never
fires at all.

**It measures the wrong thing.** Recency and stored confidence are 35% of that
score and are near-identical for every fresh candidate, so they carry no
evidence that a memory answers *this* query — the same reasoning behind the
relevance gates in [`policy.py`](../agent_memory/policy.py). The usage term is
worse: it means a memory clears a composite floor more easily the more it is
reused, so a confidence-gated hook grows quietly more aggressive over a store's
lifetime.

The gate therefore requires **two independent keys**:

1. the policy already chose `REPLAY` (composite ≥ 0.85 *and* the top hit beat
   its nearest disagreeing runner-up), and
2. hybrid relevance ≥ `--gate-min-relevance` (default **0.90**).

Measured on a seeded 8-memory store (8 verbatim repeats, 8 paraphrases, 7
unrelated queries):

| Query class | Relevance range | Gate fires |
|---|---|---|
| Verbatim repeats | 0.9375 – 0.9692 | 8/8 |
| Paraphrases | 0.4748 – 0.8134 | 0 |
| Unrelated | 0.3408 – 0.7543 | 0 |

Any floor in `[0.90, 0.93]` gated 8/8 repeats with zero false gates. 0.90 keeps
0.087 of headroom above the highest non-exact match — the direction that
matters, since failing to gate degrades to `context` mode while a false gate
takes the turn. `tests/test_hooks.py` pins this separation, so a retrieval change
that collapses it fails CI rather than silently loosening the gate.

---

## Latency

The hook runs before every prompt, so its cost is paid on every turn. Measured
on this repo (macOS, warm filesystem, 2-memory store, 5 runs averaged):

| Path | Latency |
|---|---|
| Prefilter skip (short prompt, slash command, bypass token) | **303 ms** |
| Resolve, `--lexical` | **315 ms** |
| Resolve, semantic (ONNX MiniLM) | **898 ms** |

Two things keep this bounded:

**The prefilter runs before the store is constructed.** Empty prompts, `/`
commands, `!` passthroughs, prompts under 12 characters, prompts over 2000
characters (pasted logs and diffs, which are not recall-shaped), and anything
containing the bypass token all return before anything is loaded.

**chromadb is no longer imported eagerly.** It is a core dependency reachable
from `agent_memory.store`, and a module-level import there cost ~600 ms on every
`import agent_memory` — for a backend the default SQLite path never touches. It
is now imported inside `ChromaDBStore.__init__`, which cut the hook's skip path
from 786 ms to 303 ms and sped up every other entry point too.

### `--lexical` is the better choice for `gate` mode

On the calibration fixture, lexical-only retrieval is both faster and *cleaner*
for gating, because a verbatim repeat is a perfect keyword match:

| | Exact relevance | Non-exact max | Separation | Gate |
|---|---|---|---|---|
| Semantic | 0.9464 – 0.9737 | 0.7936 | +0.153 | 8/8, 0 false |
| Lexical | **1.0000** | 0.7500 | **+0.250** | 8/8, 0 false |

Semantic embeddings remain the default for `context` mode, where the value is
catching paraphrases rather than exact repeats — the repo's LongMemEval numbers
(98.1% semantic vs 96.0% lexical Recall@5 on `_S`) reflect retrieval breadth at
a scale this 8-memory fixture cannot speak to.

---

## Failure is always silent

A memory layer is never worth a failed prompt. Every error path — unparseable
stdin, a missing or corrupt store, an unavailable backend, an exception inside
`resolve()` — exits 0 with no output, and the prompt proceeds untouched. The hook
never uses exit code 2, because that routes its message through stderr as an
error, which is the wrong framing for a memory that answered successfully.

Host-side timeouts fail open the same way: a `UserPromptSubmit` command hook that
hits its timeout is cancelled, its output discarded, and the prompt continues
without the context. `hook-install` sets `timeout: 10` rather than relying on the
30-second default, so a pathological store costs ten seconds once, not thirty.

---

## Configuration

Flags on `agent-memory hook`, or the matching environment variables:

| Flag | Env var | Default | Meaning |
|---|---|---|---|
| `--mode` | `AGENT_MEMORY_HOOK_MODE` | `context` | `context` or `gate` |
| `--data-dir` | `AGENT_MEMORY_DIR` | `.agent_memory` | Store location |
| `--backend` | `AGENT_MEMORY_HOOK_BACKEND` | `sqlite` | Storage backend |
| `--collection` | `AGENT_MEMORY_COLLECTION` | `agent_memories` | Collection name |
| `--top-k` | `AGENT_MEMORY_HOOK_TOP_K` | `3` | Candidates retrieved |
| `--scope` | `AGENT_MEMORY_HOOK_SCOPE` | all | Comma-separated scopes |
| `--actions` | `AGENT_MEMORY_HOOK_ACTIONS` | `replay,restore,verify` | Which actions inject context |
| `--gate-min-relevance` | `AGENT_MEMORY_HOOK_GATE_MIN_RELEVANCE` | `0.90` | Gate-mode relevance floor |
| `--max-prompt-chars` | `AGENT_MEMORY_HOOK_MAX_PROMPT_CHARS` | `2000` | Skip longer prompts |
| `--min-prompt-chars` | `AGENT_MEMORY_HOOK_MIN_PROMPT_CHARS` | `12` | Skip shorter prompts |
| `--bypass-token` | `AGENT_MEMORY_HOOK_BYPASS_TOKEN` | `no-memory` | One-turn escape hatch |
| `--lexical` | `AGENT_MEMORY_HOOK_SEMANTIC=off` | semantic on | Disable embeddings |

Precedence is flags > environment > defaults.

### Manual `settings.json`

`hook-install` edits the file for you, preserving existing hooks and replacing
any previous `agent-memory` entry rather than stacking duplicates. To write it
by hand:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "agent-memory hook user-prompt-submit --data-dir .agent_memory",
            "timeout": 10,
            "statusMessage": "Checking agent memory…"
          }
        ]
      }
    ]
  }
}
```

Preview without writing: `agent-memory hook-install --dry-run`.

---

## The hook only reads

Nothing here writes to the store. An empty store makes the hook a no-op — which
is the state a fresh checkout is in. Populate it with any of:

```bash
# CLI
agent-memory remember "What test runner does this repo use?" \
  "pytest via \`make test\`." --type fact --tags project
```

- the MCP `remember_memory` tool, so the agent stores its own conclusions
- `Memory.remember()` or `Memory.from_conversation()` from your own code
  (`from_conversation()` is rule-based and costs **0 LLM calls**)

Automatic capture on turn end — a `Stop` hook calling `from_conversation()` — is
the natural complement to this gate and is not implemented yet.

---

## Verifying it works

Injected context and gate decisions do not produce visible transcript entries.
To confirm delivery, run the host with `--debug`, or exercise the hook directly:

```bash
echo '{"hook_event_name":"UserPromptSubmit","prompt":"What test runner does this repo use?"}' \
  | agent-memory hook user-prompt-submit --data-dir .agent_memory
```

Empty output means no memory matched — that is the expected result on a miss,
not a failure.

# Agent Memory

[![CI](https://github.com/TheProdSDE/agent-memory-sdk/actions/workflows/ci.yml/badge.svg)](https://github.com/TheProdSDE/agent-memory-sdk/actions)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/agent-memory-sdk.svg)](https://pypi.org/project/agent-memory-sdk/)
[![MCP Registry](https://badge.mcpx.dev?type=server&name=io.github.theprodsde%2Fagent-memory)](https://registry.modelcontextprotocol.io/servers/io.github.theprodsde/agent-memory)

**Persistent semantic memory for AI agents with intelligent decision-making.**

![Agent Memory CLI demo: exact query replays, shared-word trap correctly returns none](docs/assets/demo.gif)

> **🚀 Created by:** [TheProdSDE](https://github.com/TheProdSDE)

---

## The problem

Most AI memory systems retrieve and inject past context into every prompt.
This leads to wasted tokens, inconsistent responses, and agents that blindly
replay stale or wrong answers.

Agent Memory adds a **decision layer**:

```mermaid
flowchart TD
    A[User Query] --> B[Resolve Memory]
    B --> C[Decision Engine]
    C -->|High confidence match| D[🔄 Replay — return stored answer]
    C -->|Moderate match| E[📋 Restore — inject as context]
    C -->|Needs validation| F[✅ Verify — validate before reuse]
    C -->|No match| G[❌ None — answer from scratch]

    style D fill:#0d47a1,color:#fff
    style E fill:#e65100,color:#fff
    style F fill:#1b5e20,color:#fff
    style G fill:#b71c1c,color:#fff
```

Every `resolve()` returns an **explicit action** with a scored, explainable rationale —
not just a retrieved chunk.
Adversarial eval: **25/25 (100%)** on trap queries — see [benchmarks](docs/benchmarks.md).

---

## When to use it — real use cases

| Use case | Without memory | With Agent Memory | Saving |
|----------|---------------|-------------------|--------|
| **Support bot** handling 10k identical FAQ queries/day | Every query costs 1 LLM call | ~75% REPLAY on repeated questions, 0 LLM calls | **75% cost reduction** |
| **Coding agent** that re-derives project conventions each session | Wastes 2–5 LLM calls per session to "remember" conventions | Workflows are REPLAYED instantly on first query | **No re-derivation overhead** |
| **Research agent** building knowledge over multiple sessions | Each session starts cold; re-reads the same sources | Facts and summaries are RESTORED as context | **Persistent cross-session knowledge** |
| **Customer onboarding** bot answering the same steps repeatedly | Always generates a response | High-confidence workflows are REPLAYED verbatim | **Consistent identical answers** |
| **Tool-output caching** for expensive API calls | Calls the external API every time | Results stored with TTL; REPLAY within TTL, re-call after | **Reduced external API cost** |
| **Policy-compliance agent** that must verify facts before replaying | Silent hallucination risk on stale data | `requires_verification=True` ensures VERIFY fires; stale facts are never replayed silently | **Auditability + safety** |

### Where Agent Memory saves real money

A GPT-4o call costs ~$0.005. A support agent handling 50,000 queries/day with 70% repeat rate:
- Without memory: 50,000 × $0.005 = **$250/day**
- With Agent Memory: 15,000 LLM calls + cache misses = **$75/day**
- **Saving: ~ $175/day (~$64k/year)**
- Savings can depends on how much query are repeated and similar to the previous questions.

A REPLAY costs ~0.05ms of in-process computation. An LLM call takes 300–2,000ms and costs tokens.

### Is it right for your use case?

**Good fit:**
- Agent answers the same or similar questions across sessions
- You have fact-sensitive answers that can go stale (prices, limits, policies)
- Multiple agents or services share a knowledge base
- You need audit trails — knowing *which* memory answered and why

**Not the right tool:**
- Document RAG over a corpus of files → use a vector database for that
- Replacing your application's source-of-truth database
- Agents that never repeat similar queries

---

## Features at a glance

| Feature | What it does |
|---------|-------------|
| **Decision engine** | Every `resolve()` returns REPLAY / RESTORE / VERIFY / NONE — never silent injection |
| **Explainability** | `decision.explain()` shows per-component scores: semantic, recency, confidence, usage |
| **Hybrid retrieval** | BM25 FTS5 + optional vector KNN + RRF fusion — fast and accurate |
| **4 backends** | SQLite (default, zero-setup) · ChromaDB · Redis · PostgreSQL |
| **Framework adapters** | Drop-in `BaseMemory` for LangChain and LlamaIndex |
| **MCP server** | Works with Cursor, Claude Code, VS Code via Model Context Protocol |
| **REST API** | FastAPI server with 9 endpoints + Swagger UI |
| **Dashboard** | Streamlit UI — stats, memory browser, live resolve sandbox |
| **Multi-agent** | SHARED / NAMESPACED / ISOLATED memory across multiple agents |
| **Confidence learning** | Event-driven confidence updates + half-life temporal decay |
| **Memory graph** | Relationship edges, path-finding, clusters, PageRank importance |
| **Async API** | `aremember`, `aresolve`, `alist`, … — all operations have async counterparts |
| **TTL & states** | Automatic expiry, archiving, near-duplicate consolidation |

→ Full feature reference: **[docs/features.md](docs/features.md)**

---

## Performance

All numbers are **measured** — no projections. Charts generated from real benchmark runs.

### Latency at scale (diverse unique content, no LRU cache)

![resolve() latency vs store size](docs/assets/stress_latency_scale.png)

<!-- PERF:LATENCY:START -->
| Store size | p50 | p95 | p99 | Notes |
|-----------|-----|-----|-----|-------|
| Any size (cache hit) | **0.007ms** | 0.010ms | — | LRU cache, 60–80% of production queries |
| 500 | **4.2ms** | 5.2ms | 5.6ms | |
| 1,000 | **5.8ms** | 17ms | 26ms | |
| 5,000 | **6.6ms** | 7.7ms | 8.7ms | |
| 10,000 | **8.9ms** | 10ms | 10ms | |
| 50,000 | **5.5ms** | 8.0ms | 9.3ms | |
| 100,000 | **7.4ms** | 12ms | 14ms | |
| 1,000,000 (template-repeated) | 130ms | 310ms | — | Worst case: 32K copies/template |
<!-- PERF:LATENCY:END -->

> **Key insight:** latency scales with **match count per query**, not total store size. A 1M-entry store with diverse unique memories performs near the 10K numbers.

### Tuning levers (all measured — shipped by default)

![Pareto frontier: latency vs implementation effort](docs/assets/stress_pareto_frontier.png)

| Applied by default | Impact |
|-------------------|--------|
| **LRU cache** (5s TTL, 256 entries) | 10ms → **0.007ms** for repeated queries |
| **Bloom filter** (NONE fast-path) | 0.46ms → **0.010ms** at `keyword_search` level |
| **Stop-word FTS5 filter** | 12.4ms → **4.3ms** — stops "how/do/i/my" from matching 80% of corpus |
| **`touch()` commits immediately** | Releases write lock after every REPLAY — no stall for concurrent writers |
| **PRAGMA cache_size=32MB + mmap** | -4ms vs default 2MB cache |
| **`_RRFBucket` at module level** | -0.35ms/call — was recreated inside `fuse()` each call |
| **Dynamic IDF stop words** (≥5K docs) | Filters corpus-saturated terms automatically |

Points on the Pareto frontier above cannot improve latency without increasing implementation effort. LRU cache and Bloom filter are on the frontier — they ship by default.

### Seeding throughput

<!-- PERF:SEEDING:START -->
| Mode | 10,000 | 100,000 | 1,000,000 |
|-----|-----|-----|-----|
| Standard (per-row commit) | ~1min (118/s) | ~15min (110/s) | ~5.6h (50/s) |
| **Fast-seed** (`--fast-seed`) | **~1s (8,316/s)** | **~13s (7,854/s)** | **~25s (39,913/s)** |
<!-- PERF:SEEDING:END -->

→ Full methodology, charts, and tuning guide: **[docs/stress-testing.md](docs/stress-testing.md)**

---

## When to use it

**Use Agent Memory when:**
- You want an agent to remember past interactions without injecting all of them into every prompt
- You need explicit control over *when* memory is used (replay exact answers vs inject as context vs verify first)
- You have different memory trust levels (user preferences vs potentially-stale facts vs tool outputs)
- Multiple processes, services, or agents share the same memory store
- You need audit trails — every replay is traceable to a specific stored entry with a score breakdown

**Don't use it for:**
- Document RAG (search over a corpus of files) — use a vector database for that; Agent Memory stores query→answer *experiences*
- A replacement for your database — it stores transient agent knowledge, not your application's source-of-truth data

---

## Quick Start

```bash
pip install agent-memory-sdk
```

```python
from agent_memory import Memory, MemoryAction

memory = Memory(persist_dir=".agent_memory")

# Store once after a good answer
memory.remember(
    "How do I reset my password?",
    "Go to Settings → Security → Reset Password.",
    type="conversation", tags=["auth"],
)

# Decide before every LLM call
decision = memory.resolve("How do I reset my password?")

if decision.action == MemoryAction.REPLAY:
    return decision.response          # exact match — no LLM call needed

if decision.action == MemoryAction.RESTORE:
    context = memory.format_restore_context(decision)
    return call_llm(query, system_extra=context)

# VERIFY or NONE — validate or answer fresh
```

→ Full integration pattern and API reference: **[docs/usage.md](docs/usage.md)**

---

## Local Setup

### Option 1 — SQLite (zero dependencies, recommended to start)

```bash
pip install agent-memory-sdk

# Store something
agent-memory remember "How do I reset my password?" \
  "Go to Settings → Security → Reset Password." \
  --type conversation --tags auth,faq

# Ask it back
agent-memory resolve "I forgot my password"
# ✅ REPLAY  confidence: 0.87
# matched: "How do I reset my password?"  stored: 2026-01-01  reused 1×
# response: Go to Settings → Security → Reset Password.

# See what's stored
agent-memory stats
```

### Option 2 — Redis or Postgres backend

```bash
# Spin up the services
docker compose -f docker-compose.dev.yml up -d

# Install the backend extra
pip install "agent-memory-sdk[redis]"      # or [postgres]

# Use it
agent-memory --backend redis remember "API limit" "1000 req/min" --type fact
agent-memory --backend redis resolve "What is the rate limit?"
```

### Option 3 — Streamlit dashboard (visual exploration)

```bash
pip install "agent-memory-sdk[dashboard]"

# Seed demo data (optional)
python scripts/seed_demo.py --data-dir .agent_memory

# Open the dashboard
AGENT_MEMORY_DIR=.agent_memory agent-memory-dashboard
# → http://localhost:8501
```

### Option 4 — Development / from source

```bash
git clone https://github.com/TheProdSDE/agent-memory-sdk.git
cd agent-memory-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
make test          # run all tests
make check         # lint + type check
```

---

## Dashboard

An interactive Streamlit dashboard for exploring memories, testing the resolve sandbox, and monitoring stats.

![Agent Memory dashboard slideshow: stats, memory table, replay/verify/none resolve results](docs/assets/dashboard_demo.gif)

| Stats — KPIs + charts | Memories — searchable table |
|---|---|
| ![Stats tab: 31 total, donut chart by state, bar chart by type](docs/assets/01_stats.png) | ![Memories tab: 29 rows with type, scope, confidence, access count](docs/assets/02_memories.png) |

| Resolve → REPLAY | Resolve → VERIFY |
|---|---|
| ![REPLAY badge, confidence 0.88, full response shown](docs/assets/03_resolve_replay.png) | ![VERIFY badge, context entry with fact response](docs/assets/04_resolve_verify.png) |

```bash
pip install "agent-memory-sdk[dashboard]"
AGENT_MEMORY_DIR=.agent_memory agent-memory-dashboard   # → http://localhost:8501

# Seed demo data (optional — run only when you want it)
python scripts/seed_demo.py --data-dir .agent_memory
```

---

## MCP Server

Agent Memory is published on the [MCP Registry](https://registry.modelcontextprotocol.io/servers/io.github.theprodsde/agent-memory) — install it in any MCP-compatible client with zero manual setup.

### Add to your MCP client

**Cursor** — add to `.cursor/mcp.json` in your project, or `~/.cursor/mcp.json` globally:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uvx",
      "args": ["agent-memory-sdk"]
    }
  }
}
```

**Claude Desktop** — add to `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or `%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uvx",
      "args": ["agent-memory-sdk"]
    }
  }
}
```

**Claude Code** — add to `.claude/settings.json` in your project:

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uvx",
      "args": ["agent-memory-sdk"]
    }
  }
}
```

`uvx` installs the package on first run — no `pip install` needed.

### Custom storage location

```json
{
  "mcpServers": {
    "agent-memory": {
      "command": "uvx",
      "args": ["agent-memory-sdk"],
      "env": {
        "AGENT_MEMORY_DIR": "/path/to/your/memory",
        "AGENT_MEMORY_COLLECTION": "my_project"
      }
    }
  }
}
```

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_MEMORY_DIR` | `~/.agent_memory` | Directory for the persistent SQLite store |
| `AGENT_MEMORY_COLLECTION` | `agent_memories` | Collection name (one DB per collection) |

### Tools exposed

| Tool | What it does |
|------|-------------|
| `resolve_memory` | Retrieve memory and get an explicit decision: **replay** / **restore** / **verify** / **none** — with confidence score and reasoning |
| `remember_memory` | Store a query/response pair with optional tags, type, scope, confidence, TTL |
| `list_memories` | Paginated list of stored memories with scope and archive filters |
| `get_memory_by_id` | Fetch a single memory entry by ID |
| `forget_memory` | Permanently delete a memory |
| `archive_memory` | Archive a memory (excluded from retrieval, not deleted) |
| `consolidate_memories` | Merge near-duplicate memories into summary entries |

→ Full MCP setup guide and Docker config: **[docs/mcp.md](docs/mcp.md)**

---

## Integrations

`agent-memory-sdk` is the core — every integration delegates to `Memory`.

| Integration | Install extra | Example |
|-------------|--------------|---------|
| **Core SDK** (SQLite) | *(none)* | [basic_usage.py](examples/basic_usage.py) |
| **LangChain** `BaseMemory` | `[langchain]` | [langchain_integration.py](examples/langchain_integration.py) |
| **LlamaIndex** `BaseMemory` | `[llamaindex]` | [llamaindex_integration.py](examples/llamaindex_integration.py) |
| **Redis** backend | `[redis]` | [redis_backend.py](examples/redis_backend.py) |
| **PostgreSQL** backend | `[postgres]` | [postgres_backend.py](examples/postgres_backend.py) |
| **Multi-agent** isolation | *(none)* | [multi_agent.py](examples/multi_agent.py) |
| **FastAPI** REST server | `[api]` | [rest_api.py](examples/rest_api.py) |
| **Confidence + Graph** | *(none)* | [confidence_and_graph.py](examples/confidence_and_graph.py) |
| **Benchmark harness** | *(none)* | [benchmark_harness.py](examples/benchmark_harness.py) |

→ Setup instructions and code snippets for each: **[examples/README.md](examples/README.md)**

---

## Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.10+ |
| **Storage** | SQLite · ChromaDB · Redis · PostgreSQL |
| **Retrieval** | BM25 FTS5 + Vector KNN + RRF fusion |
| **Interfaces** | MCP · FastAPI · Streamlit · CLI |
| **Adapters** | LangChain `BaseMemory` · LlamaIndex `BaseMemory` |
| **Search DSA** | Bloom filter (NONE fast-path) · Dynamic IDF stop words · RRF fusion |
| **Testing** | pytest (270 tests) · ruff · mypy |
| **CI/CD** | GitHub Actions — test matrix 3.10–3.13 → release gate → PyPI |

No API keys required — everything runs locally.

---

## Documentation

| Doc | Contents |
|-----|----------|
| [docs/usage.md](docs/usage.md) | Integration pattern, API reference, MemoryEntry / MemoryDecision fields |
| [docs/features.md](docs/features.md) | Decision actions, hybrid retrieval, types, scopes, TTL, graph, multi-agent |
| [docs/mcp.md](docs/mcp.md) | MCP server setup for Cursor, Claude Code, VS Code; Docker config |
| [docs/cli.md](docs/cli.md) | CLI commands, REST API server, dashboard launch, eval dataset format |
| [docs/roadmap.md](docs/roadmap.md) | All shipped features, what's next, GitHub Project board |
| [docs/release.md](docs/release.md) | CI-automated release process, versioning, rollback |
| [docs/architecture.md](docs/architecture.md) | Retrieval pipeline, scoring policy, system design |
| [docs/comparison.md](docs/comparison.md) | Feature matrix vs Redis, mem0, Zep, LangMem, LlamaIndex, MemGPT |
| [docs/stress-testing.md](docs/stress-testing.md) | 10K / 100K / 1M latency benchmarks with methodology |
| [docs/benchmarks.md](docs/benchmarks.md) | Eval results and reproduce commands |
| [docs/why-decision-layer.md](docs/why-decision-layer.md) | The failure mode this project exists to fix |
| [examples/README.md](examples/README.md) | Index of all runnable examples |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Dev setup, test commands, PR checklist |

---

## Status & Roadmap

All planned features through v0.5.0 are **shipped**.
Track what's next on the **[GitHub Project →](https://github.com/users/theprodsde/projects/2)**

→ **[docs/roadmap.md](docs/roadmap.md)**

## Release

Tag-triggered, fully CI-gated: `git tag v0.x.y && git push origin v0.x.y`

→ **[docs/release.md](docs/release.md)**

---

## Contributing

See **[CONTRIBUTING.md](CONTRIBUTING.md)** for dev setup, test commands, and the PR checklist.

---

## License

MIT — see [LICENSE](LICENSE).

---

## Support

- **Issues:** [GitHub Issues](https://github.com/TheProdSDE/agent-memory-sdk/issues)
- **Discussions:** [GitHub Discussions](https://github.com/TheProdSDE/agent-memory-sdk/discussions)
- **Email:** theprodsde@gmail.com

---

> **Agent Memory helps agents decide: Replay → Restore → Verify → Ignore**
>
> Built with ❤️ by [TheProdSDE](https://github.com/TheProdSDE)

mcp-name: io.github.theprodsde/agent-memory

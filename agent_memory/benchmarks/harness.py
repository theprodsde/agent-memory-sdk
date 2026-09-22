"""LongMemEval / LoCoMo benchmark harness.

Evaluates retrieval quality of a memory store against multi-session
question-answering datasets.  The dataset format is a superset of our
existing eval format, adding ``expected_content`` and ``session_id`` fields
so we can compute recall and MRR alongside the existing action-accuracy metric.

Example dataset shape::

    {
      "name": "longmemeval_mini",
      "description": "...",
      "sessions": [
        {
          "session_id": "s1",
          "events": [
            {"query": "Q", "response": "A", "type": "fact", "tags": [...]}
          ]
        }
      ],
      "questions": [
        {
          "query": "What did the user say about X?",
          "expected_content": "some snippet that should appear in the answer",
          "expected_memory_queries": ["Q"],  // optional: queries of relevant memories
          "difficulty": "easy"
        }
      ]
    }
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from agent_memory.manager import Memory
from agent_memory.models import MemoryAction

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkEvent:
    query: str
    response: str
    session_id: str = ""
    type: str = "conversation"
    scope: str = "user"
    tags: list[str] = field(default_factory=list)
    confidence: float = 1.0
    requires_verification: bool = False
    ttl: str | int | float | None = None


@dataclass
class BenchmarkQuestion:
    query: str
    expected_content: str = ""
    expected_memory_queries: list[str] = field(default_factory=list)
    difficulty: str = "medium"
    session_id: str = ""


@dataclass
class BenchmarkDataset:
    name: str
    description: str = ""
    events: list[BenchmarkEvent] = field(default_factory=list)
    questions: list[BenchmarkQuestion] = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> BenchmarkDataset:
        """Load a dataset from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        events: list[BenchmarkEvent] = []

        # Flat event list
        for e in data.get("events", []):
            events.append(_parse_event(e))

        # Sessioned format: {"sessions": [{"session_id": ..., "events": [...]}]}
        for session in data.get("sessions", []):
            sid = session.get("session_id", "")
            for e in session.get("events", []):
                ev = _parse_event(e)
                if not ev.session_id:
                    ev.session_id = sid
                events.append(ev)

        # Backward compat: "memories" key from old eval format
        for m in data.get("memories", []):
            events.append(_parse_event(m))

        questions = [
            BenchmarkQuestion(
                query=q["query"],
                expected_content=q.get("expected_content", ""),
                expected_memory_queries=q.get("expected_memory_queries", []),
                difficulty=q.get("difficulty", "medium"),
                session_id=q.get("session_id", ""),
            )
            for q in data.get("questions", data.get("cases", []))
        ]

        return cls(
            name=data.get("name", Path(path).stem),
            description=data.get("description", ""),
            events=events,
            questions=questions,
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BenchmarkDataset:
        """Build a BenchmarkDataset directly from a dict (no file I/O)."""
        import json
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8"
        ) as f:
            json.dump(data, f)
            tmp_path = f.name
        return cls.load(Path(tmp_path))


def _parse_event(e: dict[str, Any]) -> BenchmarkEvent:
    return BenchmarkEvent(
        query=e.get("query", ""),
        response=e.get("response", ""),
        session_id=e.get("session_id", ""),
        type=e.get("type", "conversation"),
        scope=e.get("scope", "user"),
        tags=e.get("tags", []),
        confidence=float(e.get("confidence", 1.0)),
        requires_verification=bool(e.get("requires_verification", False)),
        ttl=e.get("ttl"),
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HarnessConfig:
    """Tuning knobs for the benchmark run."""

    top_k: int = 5
    """How many memories to retrieve per question."""

    recall_at_k: list[int] = field(default_factory=lambda: [1, 3, 5])
    """Recall@k values to compute."""

    content_match_threshold: float = 0.5
    """Fraction of expected_content words that must appear in the retrieved
    response for a question to count as a content-recall hit."""

    include_none_action: bool = False
    """Count NONE action as a retrieval failure even if content matches."""


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class HarnessMetrics:
    recall_at_k: dict[int, float] = field(default_factory=dict)
    mrr: float = 0.0
    action_accuracy: float = 0.0
    content_recall: float = 0.0
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "recall_at_k": {f"R@{k}": round(v, 4) for k, v in self.recall_at_k.items()},
            "mrr": round(self.mrr, 4),
            "action_accuracy": round(self.action_accuracy, 4),
            "content_recall": round(self.content_recall, 4),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "p95_latency_ms": round(self.p95_latency_ms, 2),
        }


@dataclass
class HarnessResult:
    dataset: str
    total_questions: int = 0
    total_events: int = 0
    metrics: HarnessMetrics = field(default_factory=HarnessMetrics)
    failures: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total_questions": self.total_questions,
            "total_events": self.total_events,
            "metrics": self.metrics.to_dict(),
            "failures": self.failures[:20],  # cap to keep output readable
        }

    def format(self) -> str:
        m = self.metrics
        lines = [
            f"Dataset: {self.dataset}",
            f"  Events seeded:   {self.total_events}",
            f"  Questions asked: {self.total_questions}",
            f"  Content recall:  {m.content_recall:.1%}",
            f"  Action accuracy: {m.action_accuracy:.1%}",
            f"  MRR:             {m.mrr:.4f}",
        ]
        for k, v in sorted(m.recall_at_k.items()):
            lines.append(f"  Recall@{k}:       {v:.1%}")
        lines.append(f"  Avg latency:     {m.avg_latency_ms:.1f} ms")
        lines.append(f"  P95 latency:     {m.p95_latency_ms:.1f} ms")
        if self.failures:
            lines.append(f"  Failures:        {len(self.failures)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Harness
# ---------------------------------------------------------------------------


class BenchmarkHarness:
    """Run a LongMemEval- or LoCoMo-style evaluation against a Memory store.

    Usage::

        harness = BenchmarkHarness(memory)
        dataset = BenchmarkDataset.load("longmemeval_mini.json")
        result = harness.run(dataset)
        print(result.format())
    """

    def __init__(self, memory: Memory, config: HarnessConfig | None = None) -> None:
        self.memory = memory
        self.config = config or HarnessConfig()

    def seed(self, dataset: BenchmarkDataset) -> int:
        """Seed all events from *dataset* into the memory store.

        Returns the number of events stored.
        """
        count = 0
        for event in dataset.events:
            self.memory.remember(
                query=event.query,
                response=event.response,
                type=event.type,
                scope=event.scope,
                tags=event.tags,
                confidence=event.confidence,
                requires_verification=event.requires_verification,
                ttl=event.ttl,
                metadata={"session_id": event.session_id} if event.session_id else {},
            )
            count += 1
        return count

    def run(self, dataset: BenchmarkDataset) -> HarnessResult:
        """Seed *dataset* and evaluate all questions. Returns a HarnessResult."""
        result = HarnessResult(dataset=dataset.name)
        result.total_events = self.seed(dataset)
        result.total_questions = len(dataset.questions)

        if not dataset.questions:
            return result

        cfg = self.config
        latencies: list[float] = []
        reciprocal_ranks: list[float] = []
        recall_hits: dict[int, int] = {k: 0 for k in cfg.recall_at_k}
        content_hits = 0
        action_hits = 0

        for question in dataset.questions:
            t0 = time.perf_counter()
            decision = self.memory.resolve(question.query, top_k=cfg.top_k)
            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            # Collect all retrieved memory texts
            retrieved: list[str] = []
            if decision.memory:
                retrieved.append(
                    f"{decision.memory.query}\n{decision.memory.response}"
                )
            for ctx in decision.context:
                retrieved.append(
                    f"{ctx.entry.query}\n{ctx.entry.response}"
                )

            # --- Content recall: expected_content words in any retrieved text ---
            content_ok = False
            if question.expected_content:
                expected_words = set(question.expected_content.lower().split())
                for text in retrieved:
                    text_words = set(text.lower().split())
                    overlap = len(expected_words & text_words) / max(
                        len(expected_words), 1
                    )
                    if overlap >= cfg.content_match_threshold:
                        content_ok = True
                        break
            else:
                content_ok = decision.action != MemoryAction.NONE

            if content_ok:
                content_hits += 1

            # --- Action accuracy: NONE is a failure when expected_content set ---
            action_ok = not (
                question.expected_content
                and decision.action == MemoryAction.NONE
                and not cfg.include_none_action
            )
            if action_ok:
                action_hits += 1

            # --- Recall@k: whether any expected memory query appears at rank k ---
            if question.expected_memory_queries:
                rank_found: int | None = None
                for rank, ctx_result in enumerate(decision.context, start=1):
                    if any(
                        ctx_result.entry.query.lower().startswith(eq.lower()[:20])
                        for eq in question.expected_memory_queries
                    ):
                        rank_found = rank
                        break
                if (
                    decision.memory
                    and rank_found is None
                    and any(
                        decision.memory.query.lower().startswith(eq.lower()[:20])
                        for eq in question.expected_memory_queries
                    )
                ):
                    rank_found = 1

                rr = (1.0 / rank_found) if rank_found else 0.0
                reciprocal_ranks.append(rr)
                for k in cfg.recall_at_k:
                    if rank_found is not None and rank_found <= k:
                        recall_hits[k] += 1
            else:
                # No explicit memory queries → use content_ok as recall proxy
                reciprocal_ranks.append(1.0 if content_ok else 0.0)
                for k in cfg.recall_at_k:
                    if content_ok:
                        recall_hits[k] += 1

            if not content_ok:
                result.failures.append(
                    {
                        "query": question.query,
                        "action": decision.action.value,
                        "expected_content": question.expected_content[:80],
                        "retrieved_count": len(retrieved),
                        "difficulty": question.difficulty,
                    }
                )

        n = len(dataset.questions)
        latencies_sorted = sorted(latencies)
        result.metrics = HarnessMetrics(
            recall_at_k={k: recall_hits[k] / n for k in cfg.recall_at_k},
            mrr=sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0,
            action_accuracy=action_hits / n,
            content_recall=content_hits / n,
            avg_latency_ms=sum(latencies) / len(latencies),
            p95_latency_ms=latencies_sorted[int(0.95 * len(latencies_sorted)) - 1],
        )
        return result

    def run_from_file(self, path: str | Path) -> HarnessResult:
        """Convenience: load dataset from *path* and run."""
        return self.run(BenchmarkDataset.load(path))


def format_harness_report(results: list[HarnessResult]) -> str:
    """Format a list of HarnessResult objects into a readable report."""
    lines = ["Agent Memory Benchmark Report", "=" * 40, ""]
    for r in results:
        lines.append(r.format())
        lines.append("")
    if len(results) > 1:
        all_n = sum(r.total_questions for r in results)
        avg_cr = sum(r.metrics.content_recall * r.total_questions for r in results) / max(
            all_n, 1
        )
        lines.append(f"Overall content recall: {avg_cr:.1%} across {all_n} questions")
    return "\n".join(lines)

"""LongMemEval / LoCoMo benchmark harness — evaluate retrieval quality.

agent-memory-sdk ships a built-in harness that measures how well the
memory store retrieves relevant information across multi-session datasets.

Metrics computed:
  Recall@k     — fraction of questions where the right memory ranked ≤ k
  MRR          — mean reciprocal rank of the first correct retrieval
  Content recall — fraction of expected answer words found in retrieved text
  Action accuracy — fraction of questions not incorrectly returning NONE
  P95 latency  — 95th-percentile resolve latency in milliseconds
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

from agent_memory import BenchmarkDataset, BenchmarkHarness, Memory
from agent_memory.benchmarks.harness import HarnessConfig

# ── Build a small in-line dataset (normally you'd load a JSON file) ───────────
DATASET = {
    "name": "customer_support_mini",
    "description": "Support bot memory benchmark — 3 sessions, 5 questions",
    "sessions": [
        {
            "session_id": "s1",
            "events": [
                {
                    "query":    "How do I reset my password?",
                    "response": "Settings → Security → Reset Password. Link expires in 30 min.",
                    "type":     "conversation",
                    "tags":     ["auth", "password"],
                },
                {
                    "query":    "What payment methods do you accept?",
                    "response": "Visa, Mastercard, PayPal, and Apple Pay.",
                    "type":     "fact",
                    "tags":     ["billing", "payments"],
                },
            ],
        },
        {
            "session_id": "s2",
            "events": [
                {
                    "query":    "What is the API rate limit?",
                    "response": "Free: 100 req/min. Pro: 1000 req/min.",
                    "type":     "fact",
                    "tags":     ["api", "limits"],
                    "requires_verification": True,
                },
                {
                    "query":    "How do I invite team members?",
                    "response": "Settings → Team → Invite. Invitations expire after 7 days.",
                    "type":     "workflow",
                    "tags":     ["team", "onboarding"],
                },
                {
                    "query":    "What is the enterprise SLA?",
                    "response": "99.99% uptime. P1 response: 15 minutes.",
                    "type":     "fact",
                    "tags":     ["sla", "enterprise"],
                    "confidence": 0.9,
                },
            ],
        },
    ],
    "questions": [
        {
            "query":                   "I forgot my password, what should I do?",
            "expected_content":        "Reset Password",
            "expected_memory_queries": ["How do I reset my password?"],
            "difficulty":              "easy",
        },
        {
            "query":                   "Which payment options are available?",
            "expected_content":        "Visa",
            "expected_memory_queries": ["What payment methods do you accept?"],
            "difficulty":              "easy",
        },
        {
            "query":                   "How many requests can I make per minute on the free plan?",
            "expected_content":        "100",
            "expected_memory_queries": ["What is the API rate limit?"],
            "difficulty":              "medium",
        },
        {
            "query":                   "How do I add a new user to my workspace?",
            "expected_content":        "Invite",
            "expected_memory_queries": ["How do I invite team members?"],
            "difficulty":              "medium",
        },
        {
            "query":                   "What uptime guarantee do enterprise customers get?",
            "expected_content":        "99.99%",
            "expected_memory_queries": ["What is the enterprise SLA?"],
            "difficulty":              "hard",
        },
    ],
}

# ── Save dataset to a temp file (or use BenchmarkDataset.from_dict directly) ──
with tempfile.TemporaryDirectory() as tmp:
    dataset_path = Path(tmp) / "support_mini.json"
    dataset_path.write_text(json.dumps(DATASET), encoding="utf-8")

    # ── Load dataset ──────────────────────────────────────────────────────────
    dataset = BenchmarkDataset.load(dataset_path)
    print(f"Dataset: '{dataset.name}'")
    print(f"  Events:    {len(dataset.events)}")
    print(f"  Questions: {len(dataset.questions)}")

    # ── Create memory store and harness ───────────────────────────────────────
    memory = Memory(persist_dir=tmp, collection_name="bench")
    config = HarnessConfig(
        top_k=5,
        recall_at_k=[1, 3, 5],
        content_match_threshold=0.4,
    )
    harness = BenchmarkHarness(memory, config=config)

    # ── Run benchmark ─────────────────────────────────────────────────────────
    result = harness.run(dataset)

    # ── Print report ──────────────────────────────────────────────────────────
    print()
    print(result.format())

    # ── Structured output (for CI / dashboards) ───────────────────────────────
    print("\nJSON output:")
    print(json.dumps(result.to_dict(), indent=2))

    # ── Run multiple datasets and get a combined report ───────────────────────
    # result2 = harness.run_from_file("locomo_mini.json")
    # report = format_harness_report([result, result2])
    # print(report)

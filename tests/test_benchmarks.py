"""Tests for the LongMemEval / LoCoMo benchmark harness."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_memory.benchmarks.harness import (
    BenchmarkDataset,
    BenchmarkHarness,
    HarnessConfig,
    format_harness_report,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def memory(tmp_path):
    from agent_memory.manager import Memory

    return Memory(persist_dir=tmp_path, collection_name="bench_test")


SAMPLE_DATASET = {
    "name": "sample_bench",
    "description": "Minimal test dataset",
    "sessions": [
        {
            "session_id": "s1",
            "events": [
                {
                    "query": "What is the user's favorite programming language?",
                    "response": "Python",
                    "type": "fact",
                    "tags": ["preference", "python"],
                },
                {
                    "query": "What framework does the user prefer for web apps?",
                    "response": "FastAPI",
                    "type": "fact",
                    "tags": ["preference", "web"],
                },
            ],
        }
    ],
    "questions": [
        {
            "query": "What programming language does the user like?",
            "expected_content": "Python",
            "expected_memory_queries": [
                "What is the user's favorite programming language?"
            ],
            "difficulty": "easy",
        },
        {
            "query": "What web framework does the user prefer?",
            "expected_content": "FastAPI",
            "difficulty": "easy",
        },
    ],
}


@pytest.fixture()
def dataset_file(tmp_path) -> Path:
    path = tmp_path / "sample_bench.json"
    path.write_text(json.dumps(SAMPLE_DATASET), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# BenchmarkDataset loading
# ---------------------------------------------------------------------------


def test_load_from_file(dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    assert dataset.name == "sample_bench"
    assert len(dataset.events) == 2
    assert len(dataset.questions) == 2


def test_load_flat_events(tmp_path):
    data = {
        "name": "flat",
        "events": [{"query": "q", "response": "r"}],
        "questions": [],
    }
    path = tmp_path / "flat.json"
    path.write_text(json.dumps(data))
    dataset = BenchmarkDataset.load(path)
    assert len(dataset.events) == 1


def test_load_legacy_memories_key(tmp_path):
    data = {
        "name": "legacy",
        "memories": [{"query": "q", "response": "r"}],
        "cases": [{"query": "q?", "expected_content": "r"}],
    }
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(data))
    dataset = BenchmarkDataset.load(path)
    assert len(dataset.events) == 1
    assert len(dataset.questions) == 1


# ---------------------------------------------------------------------------
# Harness seeding
# ---------------------------------------------------------------------------


def test_seed_populates_memory(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    harness = BenchmarkHarness(memory)
    count = harness.seed(dataset)
    assert count == 2
    assert memory.store.count == 2


def test_run_seeds_and_evaluates(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    harness = BenchmarkHarness(memory)
    result = harness.run(dataset)
    assert result.total_events == 2
    assert result.total_questions == 2
    assert result.dataset == "sample_bench"


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def test_metrics_structure(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    harness = BenchmarkHarness(memory)
    result = harness.run(dataset)
    m = result.metrics
    assert 0.0 <= m.content_recall <= 1.0
    assert 0.0 <= m.action_accuracy <= 1.0
    assert 0.0 <= m.mrr <= 1.0
    assert m.avg_latency_ms >= 0
    assert m.p95_latency_ms >= 0
    for k in [1, 3, 5]:
        assert k in m.recall_at_k
        assert 0.0 <= m.recall_at_k[k] <= 1.0


def test_to_dict_structure(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    result = BenchmarkHarness(memory).run(dataset)
    d = result.to_dict()
    assert "dataset" in d
    assert "metrics" in d
    assert "total_questions" in d
    assert "R@1" in d["metrics"]["recall_at_k"]


def test_format_output(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    result = BenchmarkHarness(memory).run(dataset)
    text = result.format()
    assert "sample_bench" in text
    assert "recall" in text.lower() or "Recall" in text


# ---------------------------------------------------------------------------
# Config knobs
# ---------------------------------------------------------------------------


def test_custom_config(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    cfg = HarnessConfig(top_k=1, recall_at_k=[1])
    result = BenchmarkHarness(memory, config=cfg).run(dataset)
    assert 1 in result.metrics.recall_at_k


# ---------------------------------------------------------------------------
# Run from file convenience method
# ---------------------------------------------------------------------------


def test_run_from_file(memory, dataset_file):
    harness = BenchmarkHarness(memory)
    result = harness.run_from_file(dataset_file)
    assert result.total_questions == 2


# ---------------------------------------------------------------------------
# Report formatter
# ---------------------------------------------------------------------------


def test_format_harness_report(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    result = BenchmarkHarness(memory).run(dataset)
    report = format_harness_report([result])
    assert "sample_bench" in report
    assert "Benchmark Report" in report


def test_format_harness_report_multiple(memory, dataset_file):
    dataset = BenchmarkDataset.load(dataset_file)
    harness = BenchmarkHarness(memory)
    r1 = harness.run(dataset)
    r2 = harness.run(dataset)
    report = format_harness_report([r1, r2])
    assert "Overall" in report


# ---------------------------------------------------------------------------
# Empty dataset edge case
# ---------------------------------------------------------------------------


def test_empty_dataset(memory, tmp_path):
    data = {"name": "empty", "events": [], "questions": []}
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(data))
    dataset = BenchmarkDataset.load(path)
    result = BenchmarkHarness(memory).run(dataset)
    assert result.total_questions == 0
    assert result.total_events == 0

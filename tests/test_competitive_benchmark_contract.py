"""Tests for the public competitive benchmark result contract."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.competitive.validate_result import validate_result


FIXTURE = (
    Path(__file__).parents[1]
    / "benchmarks"
    / "competitive"
    / "examples"
    / "agent_memory_longmemeval_s_lexical.json"
)


def _fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def test_agent_memory_fixture_satisfies_contract() -> None:
    assert validate_result(_fixture()) == []


def test_contract_requires_reproducibility_metadata() -> None:
    result = _fixture()
    del result["dataset"]["sha256"]
    del result["environment"]["command"]

    errors = validate_result(result)

    assert "dataset.sha256 must be a lowercase 64-character SHA256" in errors
    assert "environment.command must be a non-empty string" in errors


def test_contract_rejects_invalid_comparative_metrics() -> None:
    result = _fixture()
    result["metrics"]["recall_at_5"] = 1.1
    result["configuration"]["cache_state"] = "unknown"

    errors = validate_result(result)

    assert "metrics.recall_at_5 must be a number between 0 and 1" in errors
    assert any(error.startswith("configuration.cache_state") for error in errors)

"""Validate competitive benchmark result artifacts without external dependencies.

Usage:
    python -m benchmarks.competitive.validate_result path/to/result.json
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

_DEPLOYMENTS = {"local", "self_hosted", "hosted"}
_RETRIEVAL_UNITS = {"turn", "turn_pair", "session", "fact", "custom"}
_CACHE_STATES = {"cold", "warm", "disabled"}
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


def _require_string(data: dict[str, Any], key: str, errors: list[str], path: str) -> None:
    if not isinstance(data.get(key), str) or not data[key]:
        errors.append(f"{path}.{key} must be a non-empty string")


def _require_mapping(data: dict[str, Any], key: str, errors: list[str]) -> dict[str, Any]:
    value = data.get(key)
    if not isinstance(value, dict):
        errors.append(f"{key} must be an object")
        return {}
    return value


def validate_result(result: dict[str, Any]) -> list[str]:
    """Return contract violations for a competitive benchmark result."""
    errors: list[str] = []
    if result.get("schema_version") != "1.0":
        errors.append("schema_version must be '1.0'")

    system = _require_mapping(result, "system", errors)
    for key in ("name", "version"):
        _require_string(system, key, errors, "system")
    if system.get("deployment") not in _DEPLOYMENTS:
        errors.append(f"system.deployment must be one of {sorted(_DEPLOYMENTS)}")

    dataset = _require_mapping(result, "dataset", errors)
    for key in ("name", "release"):
        _require_string(dataset, key, errors, "dataset")
    if not isinstance(dataset.get("sha256"), str) or not _SHA256.fullmatch(dataset["sha256"]):
        errors.append("dataset.sha256 must be a lowercase 64-character SHA256")
    if dataset.get("retrieval_unit") not in _RETRIEVAL_UNITS:
        errors.append(f"dataset.retrieval_unit must be one of {sorted(_RETRIEVAL_UNITS)}")

    environment = _require_mapping(result, "environment", errors)
    for key in ("os", "hardware", "python", "command"):
        _require_string(environment, key, errors, "environment")

    configuration = _require_mapping(result, "configuration", errors)
    top_k = configuration.get("top_k")
    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        errors.append("configuration.top_k must be a positive integer")
    if configuration.get("cache_state") not in _CACHE_STATES:
        errors.append(f"configuration.cache_state must be one of {sorted(_CACHE_STATES)}")

    metrics = _require_mapping(result, "metrics", errors)
    for key in ("recall_at_5", "recall_at_10", "qa_accuracy", "wrong_replay_rate"):
        if key in metrics and (
            not isinstance(metrics[key], (int, float)) or not 0 <= metrics[key] <= 1
        ):
            errors.append(f"metrics.{key} must be a number between 0 and 1")
    for key in ("process_rss_mb", "ingest_seconds", "cost_usd"):
        if key in metrics and (
            not isinstance(metrics[key], (int, float)) or metrics[key] < 0
        ):
            errors.append(f"metrics.{key} must be a non-negative number")

    artifacts = _require_mapping(result, "artifacts", errors)
    _require_string(artifacts, "raw_results", errors, "artifacts")
    return errors


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m benchmarks.competitive.validate_result RESULT.json", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    try:
        result = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        print(f"cannot read {path}: {exc}", file=sys.stderr)
        return 2
    errors = validate_result(result)
    if errors:
        print("invalid competitive benchmark result:", file=sys.stderr)
        print("\n".join(f"- {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"valid competitive benchmark result: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

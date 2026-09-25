#!/usr/bin/env python
"""Generate stress charts from archived stress_test.py JSON results.

Usage:
    uv run python scripts/generate_stress_charts.py
    uv run python scripts/generate_stress_charts.py --results-dir benchmarks/stress/results

The script deliberately has no embedded benchmark values. Every plotted point
must come from a JSON result emitted by scripts/stress_test.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_results(results_dir: Path) -> list[dict]:
    results = []
    for path in sorted(results_dir.glob("*.json")):
        result = json.loads(path.read_text(encoding="utf-8"))
        result["_source"] = path.name
        results.append(result)
    if not results:
        raise SystemExit(f"No benchmark JSON files found in {results_dir}")
    return results


def _plot(results: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "Chart generation requires matplotlib. Install it with "
            "'uv pip install matplotlib' or add it to the local development environment."
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    fixture_note = "Bundled synthetic stress fixture; each point comes from archived JSON"

    latency = [result for result in results if result.get("n_queries", 0) >= 2]
    if latency:
        latency.sort(key=lambda result: result["n_memories"])
        sizes = [result["n_memories"] for result in latency]
        p50 = [result["median_ms"] for result in latency]
        p95 = [result["p95_ms"] for result in latency]
        p99 = [result["p99_ms"] for result in latency]

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.plot(sizes, p50, "o-", label="p50")
        axis.plot(sizes, p95, "s-", label="p95")
        axis.plot(sizes, p99, "^-", label="p99")
        axis.set_xscale("log")
        axis.set_xlabel("Stored memories")
        axis.set_ylabel("resolve() latency (ms)")
        axis.set_title("Stress latency by stored-memory count")
        axis.text(
            0.02,
            0.02,
            fixture_note,
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(True, alpha=0.25)
        axis.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "stress_latency_archived.png", dpi=160)
        plt.close(fig)

    seeded = [result for result in results if result.get("seed_rate") is not None]
    if seeded:
        seeded.sort(key=lambda result: result["n_memories"])
        sizes = [result["n_memories"] for result in seeded]
        rates = [result["seed_rate"] for result in seeded]

        fig, axis = plt.subplots(figsize=(9, 5))
        axis.bar([f"{size:,}" for size in sizes], rates)
        axis.set_xlabel("Entries seeded")
        axis.set_ylabel("Seed rate (entries/s)")
        axis.set_title("Fast-seed throughput from archived runs")
        axis.text(
            0.02,
            0.02,
            fixture_note,
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "stress_seeding_archived.png", dpi=160)
        plt.close(fig)

    resource_profiles = [
        result
        for result in results
        if result.get("query_rss_end_mb") is not None and result.get("p95_ms") is not None
    ]
    if resource_profiles:
        fig, axis = plt.subplots(figsize=(9, 5))
        for result in resource_profiles:
            axis.scatter(result["query_rss_end_mb"], result["p95_ms"], s=80)
            axis.annotate(
                f"{result['n_memories']:,}",
                (result["query_rss_end_mb"], result["p95_ms"]),
                xytext=(6, 6),
                textcoords="offset points",
            )
        axis.set_xlabel("Endpoint RSS after retrieval (MiB)")
        axis.set_ylabel("resolve() p95 latency (ms)")
        axis.set_title("Measured resource and latency profile")
        axis.text(
            0.02,
            0.02,
            "Not a Pareto frontier; no subjective effort or projected backend data",
            transform=axis.transAxes,
            fontsize=8,
            color="dimgray",
        )
        axis.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(output_dir / "stress_resource_latency_archived.png", dpi=160)
        plt.close(fig)

    if not latency and not seeded:
        raise SystemExit("Archived results contain neither latency nor seed metrics")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("benchmarks/stress/results"),
        help="directory containing stress_test.py JSON output",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/assets"),
        help="directory for generated PNG charts",
    )
    args = parser.parse_args()
    results = _load_results(args.results_dir)
    _plot(results, args.output_dir)
    print(f"Generated charts from {len(results)} archived result(s) in {args.output_dir}")


if __name__ == "__main__":
    main()

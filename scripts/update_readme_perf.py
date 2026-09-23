#!/usr/bin/env python
"""Rewrite the performance tables in README.md from benchmark data.

Run after every benchmark / stress-test update:

    python scripts/update_readme_perf.py

Data source: scripts/generate_stress_charts.py  (DIVERSE_DATA, std_rates, fast_rates)
Targets    : README.md sections between <!-- PERF:*:START --> / <!-- PERF:*:END --> anchors

Adding a new store-size measurement to DIVERSE_DATA in generate_stress_charts.py is
all that's needed — this script picks it up automatically on the next run.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).parent.parent
README = REPO / "README.md"
CHARTS = REPO / "scripts" / "generate_stress_charts.py"


# ---------------------------------------------------------------------------
# Load data from generate_stress_charts.py without running its plot code
# ---------------------------------------------------------------------------

def _load_chart_data() -> dict:
    """Extract only the DATA constants from the charts script via regex + eval.

    The constants live at the top of the file before any matplotlib calls.
    We pull out the named assignments and eval them in a safe namespace rather
    than importing the whole module (which would try to run matplotlib code).
    """
    src = CHARTS.read_text(encoding="utf-8")

    # Grab the block between the data sentinel comment and the first Chart comment
    m = re.search(
        r"# ── DATA.*?\n(.*?)# =+\s*\n# Chart 1",
        src,
        re.DOTALL,
    )
    if not m:
        # Fallback: grab everything up to the first plt.subplots call
        m = re.search(r"# ── DATA.*?\n(.*?)fig, ax = plt", src, re.DOTALL)
    if not m:
        sys.exit("Error: could not find data block in generate_stress_charts.py")

    data_block = m.group(1)

    # Also grab the seeding arrays that come later (they use plain names)
    for extra_name in ("std_rates", "fast_rates", "x_vals"):
        pattern = rf"^({extra_name}\s*=\s*.+?)(?=\n\S|\Z)"
        em = re.search(pattern, src, re.MULTILINE | re.DOTALL)
        if em:
            data_block += "\n" + em.group(1)

    ns: dict = {}
    exec(data_block, {"__builtins__": {}}, ns)  # noqa: S102 — controlled input
    return ns


# ---------------------------------------------------------------------------
# Table builders
# ---------------------------------------------------------------------------

def _fmt(ms: float) -> str:
    """Format a latency value for the table."""
    if ms < 1:
        return f"{ms:.3f}ms"
    if ms < 10:
        return f"{ms:.1f}ms"
    return f"{ms:.0f}ms"


def _duration(rate: float, n: int) -> str:
    """Convert entries/s + count to a human-readable duration string (no ~ prefix)."""
    secs = n / rate
    if secs < 60:
        return f"{secs:.0f}s"
    if secs < 3600:
        return f"{secs/60:.0f}min"
    return f"{secs/3600:.1f}h"


def build_latency_table(data: dict) -> str:
    rows = [
        "| Store size | p50 | p95 | p99 | Notes |",
        "|-----------|-----|-----|-----|-------|",
        "| Any size (cache hit) | **0.007ms** | 0.010ms | — "
        "| LRU cache, 60–80% of production queries |",
    ]
    for scale, avg, p50, p90, p95, p99 in data["DIVERSE_DATA"]:
        size = f"{scale:,}"
        rows.append(f"| {size} | **{_fmt(p50)}** | {_fmt(p95)} | {_fmt(p99)} | |")

    # Template-repeated worst case from TEMPLATE_DATA (first entry is smallest scale)
    tpl = data["TEMPLATE_DATA"]
    if tpl:
        scale_tpl, p50_tpl, p95_tpl = tpl[-1]  # largest measured scale
        rows.append(
            f"| {scale_tpl:,} (template-repeated) | {_fmt(p50_tpl)} | {_fmt(p95_tpl)} | — "
            "| Worst case: 32K copies/template |"
        )
    return "\n".join(rows)


def build_seeding_table(data: dict) -> str:
    x_vals = data["x_vals"]          # [10_000, 100_000, 1_000_000]
    std_rates = data["std_rates"]
    fast_rates = data["fast_rates"]

    std_cells = " | ".join(
        f"~{_duration(r, n)} ({r:,.0f}/s)" for r, n in zip(std_rates, x_vals)
    )
    fast_cells = " | ".join(
        f"**~{_duration(r, n)} ({r:,.0f}/s)**" for r, n in zip(fast_rates, x_vals)
    )
    headers = " | ".join(f"{n:,}" for n in x_vals)
    sep = "|" + "|".join("-----" for _ in range(len(x_vals) + 1)) + "|"

    return "\n".join([
        f"| Mode | {headers} |",
        sep,
        f"| Standard (per-row commit) | {std_cells} |",
        f"| **Fast-seed** (`--fast-seed`) | {fast_cells} |",
    ])


# ---------------------------------------------------------------------------
# README injector
# ---------------------------------------------------------------------------

def _replace_section(text: str, tag: str, new_content: str) -> str:
    pattern = (
        rf"(<!-- PERF:{tag}:START -->)\n"
        rf".*?"
        rf"(<!-- PERF:{tag}:END -->)"
    )
    replacement = rf"\1\n{new_content}\n\2"
    result, n = re.subn(pattern, replacement, text, flags=re.DOTALL)
    if n == 0:
        sys.exit(f"Error: anchor <!-- PERF:{tag}:START/END --> not found in README.md")
    return result


def main() -> None:
    data = _load_chart_data()

    readme = README.read_text(encoding="utf-8")
    readme = _replace_section(readme, "LATENCY", build_latency_table(data))
    readme = _replace_section(readme, "SEEDING", build_seeding_table(data))
    README.write_text(readme, encoding="utf-8")

    print("✓ README.md performance tables updated")
    print(f"  Latency rows: {len(data['DIVERSE_DATA'])} sizes + cache hit + template worst-case")
    print(f"  Seeding rows: {len(data['x_vals'])} sizes × 2 modes")


if __name__ == "__main__":
    main()

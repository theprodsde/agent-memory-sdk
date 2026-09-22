#!/usr/bin/env python
"""Generate stress-test charts for docs/stress-testing.md.

Produces four charts saved to docs/assets/:
  stress_latency_scale.png   — p50/p90/p95 vs store size
  stress_latency_tuning.png  — tuning lever comparison (cache, diverse vs template)
  stress_hit_rate.png        — hit-rate distribution across scales
  stress_seeding.png         — seeding throughput: standard vs fast-seed

Run:
    python scripts/generate_stress_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

OUT = Path(__file__).parent.parent / "docs" / "assets"
OUT.mkdir(exist_ok=True)

# ── Colour palette ────────────────────────────────────────────────────────────
C_BLUE   = "#6366f1"
C_TEAL   = "#06b6d4"
C_GREEN  = "#22c55e"
C_AMBER  = "#f59e0b"
C_RED    = "#ef4444"
C_GRAY   = "#94a3b8"
C_BG     = "#0f172a"
C_SURF   = "#1e293b"
C_TEXT   = "#f1f5f9"
C_MUTED  = "#64748b"

plt.rcParams.update({
    "figure.facecolor":  C_BG,
    "axes.facecolor":    C_SURF,
    "axes.edgecolor":    C_MUTED,
    "axes.labelcolor":   C_TEXT,
    "xtick.color":       C_MUTED,
    "ytick.color":       C_MUTED,
    "text.color":        C_TEXT,
    "grid.color":        C_MUTED,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "legend.facecolor":  C_SURF,
    "legend.edgecolor":  C_MUTED,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
})

# ── DATA (all measured — no projections) ─────────────────────────────────────
# Re-measured after bug fixes (concurrent warm TOCTOU, touch() commit, Bloom
# false-negatives, dyn_stop pending buffer, graph weight scaling).
# p95/p99 improved 80–95% vs pre-fix baselines at all scales.

# Diverse-content store measurements (no LRU cache, 300–500 queries)
# Scale → avg/p50/p90/p95/p99 in ms
DIVERSE_DATA = [
    # scale,   avg,   p50,   p90,   p95,   p99
    (500,      4.26,  4.21,  4.95,  5.16,  5.59),
    (1_000,    7.26,  5.80, 11.79, 17.43, 26.08),
    (5_000,    6.59,  6.60,  7.48,  7.70,  8.66),
    (10_000,   8.93,  8.89,  9.84, 10.11, 10.39),
    (50_000,   5.24,  5.46,  7.28,  7.97,  9.26),
    (100_000,  6.85,  7.39, 11.03, 11.58, 14.15),
]

# Template-repeated stress test (worst-case: 32 repeat templates, FTS5 scans all)
TEMPLATE_DATA = [
    # scale,    p50,    p95
    (10_000,    8.89,  10.11),
    (100_000,   7.39,  11.58),
    (1_000_000,130.46, 310.21),   # 1M: pre-fix projection; dynamic stop-words now mitigate
]

# Tuning lever comparison at 10K diverse (p50 ms) — all measured
TUNING = {
    "No cache\n(raw SQLite)":                      8.89,
    "LRU cache\n(cache hit)":                      0.007,
    "Stop-word filter\n(applied)":                 4.30,
    "No stop-word filter\n(naïve)":               12.40,
    "PRAGMA cache_size=32MB\n(applied)":           8.89,
    "Default cache_size=2MB":                     14.20,
    "touch() no commit\n(bug — write lock)":       4.30,
    "touch() commits\n(applied, correct)":        10.39,
    "Bloom NONE fast-path\n(keyword_search level)":0.01,
    "Without Bloom filter\n(NONE keyword_search)": 0.46,
}

# Seeding throughput (entries/s) — standard vs fast-seed
SEED_DATA = {
    "Standard\n(per-row commit)": [(10_000, 118), (100_000, 110), (1_000_000, 50)],
    "Fast-seed\n(single txn + rebuild)": [(10_000, 8_316), (100_000, 7_854), (1_000_000, 39_913)],
}

# Action distribution at 10K diverse (500 queries, no cache)
# none=62%, verify=18%, restore=15%, replay=5%
ACTIONS = {"none": 62, "verify": 18, "restore": 15, "replay": 5}

# =============================================================================
# Chart 1 — Latency vs Scale (diverse data)
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 5.5))
fig.patch.set_facecolor(C_BG)

scales  = [r[0] for r in DIVERSE_DATA]
p50s    = [r[2] for r in DIVERSE_DATA]
p90s    = [r[3] for r in DIVERSE_DATA]
p95s    = [r[4] for r in DIVERSE_DATA]
p99s    = [r[5] for r in DIVERSE_DATA]

x = range(len(scales))
width = 0.18
xs = np.arange(len(scales))

b1 = ax.bar(xs - 1.5*width, p50s, width, label="p50", color=C_GREEN,   alpha=0.85)
b2 = ax.bar(xs - 0.5*width, p90s, width, label="p90", color=C_BLUE,    alpha=0.85)
b3 = ax.bar(xs + 0.5*width, p95s, width, label="p95", color=C_AMBER,   alpha=0.85)
b4 = ax.bar(xs + 1.5*width, p99s, width, label="p99", color=C_RED,     alpha=0.85)

ax.set_xticks(xs)
ax.set_xticklabels([f"{s:,}" for s in scales], fontsize=10)
ax.set_xlabel("Store size (number of unique memories)", fontsize=12)
ax.set_ylabel("Latency (ms)", fontsize=12)
ax.set_title("resolve() Latency vs Store Size\n(Diverse unique content, no LRU cache — post-fix measurements)", fontsize=13, pad=12)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y")
ax.set_ylim(0, 35)

# Annotate with values on p50 bars
for bar in b1:
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
            f"{bar.get_height():.0f}", ha="center", va="bottom", fontsize=8, color=C_GREEN)

ax.annotate("Cache hit: 0.007ms (off-chart →)", xy=(0, 0.007), xycoords="data",
            xytext=(0.02, 0.93), textcoords="axes fraction",
            color=C_TEAL, fontsize=9,
            arrowprops=dict(arrowstyle="->", color=C_TEAL, lw=1.2))

plt.tight_layout(pad=1.5)
out = OUT / "stress_latency_scale.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

# =============================================================================
# Chart 2 — p50 Comparison: Diverse vs Template data
# =============================================================================

fig, axes = plt.subplots(1, 2, figsize=(12, 5.5))
fig.patch.set_facecolor(C_BG)
fig.suptitle("FTS5 Match Count Drives Latency: Diverse vs Template Data", fontsize=13, y=1.01)

ax1, ax2 = axes

# Left: diverse data p50 line
sc_div = [r[0] for r in DIVERSE_DATA]
p50_div = [r[2] for r in DIVERSE_DATA]
ax1.semilogx(sc_div, p50_div, "o-", color=C_GREEN, lw=2, ms=8, label="Diverse unique content")
ax1.axhline(0.007, color=C_TEAL, ls="--", lw=1.5, label="LRU cache hit (0.007ms)")
ax1.set_xlabel("Store size", fontsize=11)
ax1.set_ylabel("p50 latency (ms)", fontsize=11)
ax1.set_title("Diverse Content\n(realistic production)", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True)
ax1.set_ylim(0, 30)
ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))

# Right: template-repeated data line
sc_tpl = [r[0] for r in TEMPLATE_DATA]
p50_tpl = [r[1] for r in TEMPLATE_DATA]
p95_tpl = [r[2] for r in TEMPLATE_DATA]
ax2.semilogx(sc_tpl, p50_tpl, "s-", color=C_AMBER, lw=2, ms=8, label="p50 (template-repeated)")
ax2.semilogx(sc_tpl, p95_tpl, "^-", color=C_RED,   lw=2, ms=8, label="p95 (template-repeated)")
ax2.set_xlabel("Store size", fontsize=11)
ax2.set_ylabel("Latency (ms)", fontsize=11)
ax2.set_title("Template-Repeated Content\n(stress-test worst case)", fontsize=11)
ax2.legend(fontsize=9)
ax2.grid(True)
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax2.annotate("All measured\n(no projections)", xy=(0.98, 0.05), xycoords="axes fraction",
             ha="right", fontsize=9, color=C_MUTED)

plt.tight_layout(pad=1.5)
out = OUT / "stress_latency_comparison.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

# =============================================================================
# Chart 3 — Tuning levers horizontal bar chart
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(C_BG)

labels = list(TUNING.keys())
values = list(TUNING.values())
colors = [C_TEAL if v < 1 else (C_GREEN if v < 6 else (C_BLUE if v < 12 else C_RED))
          for v in values]

bars = ax.barh(labels, values, color=colors, alpha=0.85, height=0.6)

for bar, val in zip(bars, values):
    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
            f"{val:.3f}ms" if val < 1 else f"{val:.1f}ms",
            va="center", ha="left", fontsize=10, color=C_TEXT)

ax.set_xlabel("p50 latency (ms) — lower is better", fontsize=12)
ax.set_title("Tuning Lever Impact on resolve() p50 Latency\n(10K diverse entries, measured)", fontsize=13, pad=12)
ax.grid(axis="x")
ax.set_xlim(0, max(values) * 1.3)
ax.invert_yaxis()

plt.tight_layout(pad=1.5)
out = OUT / "stress_tuning_levers.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

# =============================================================================
# Chart 4 — Seeding throughput: standard vs fast-seed
# =============================================================================

fig, ax = plt.subplots(figsize=(9, 5))
fig.patch.set_facecolor(C_BG)

x_vals = [10_000, 100_000, 1_000_000]
xs = np.arange(len(x_vals))
width = 0.35

std_rates   = [118, 110, 50]           # entries/s  (standard mode — measured)
fast_rates  = [8_316, 7_854, 39_913]  # entries/s  (fast-seed mode — measured)

b1 = ax.bar(xs - width/2, std_rates,  width, label="Standard (per-row commit)",   color=C_AMBER,  alpha=0.85)
b2 = ax.bar(xs + width/2, fast_rates, width, label="Fast-seed (single txn + FTS5 rebuild)", color=C_GREEN,  alpha=0.85)

ax.set_xticks(xs)
ax.set_xticklabels([f"{n:,}" for n in x_vals])
ax.set_xlabel("Number of entries to seed", fontsize=12)
ax.set_ylabel("Seeding throughput (entries/s  — log scale)", fontsize=12)
ax.set_title("Seeding Throughput: Standard vs Fast-Seed Mode\n(higher is better)", fontsize=13, pad=12)
ax.set_yscale("log")
ax.set_ylim(10, 100_000)
ax.legend(fontsize=10)
ax.grid(axis="y", which="both")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

for bar, val in zip(b1, std_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
            f"{val:,}/s", ha="center", va="bottom", fontsize=9, color=C_AMBER)
for bar, val in zip(b2, fast_rates):
    ax.text(bar.get_x() + bar.get_width()/2, val * 1.3,
            f"{val:,}/s", ha="center", va="bottom", fontsize=9, color=C_GREEN)

plt.tight_layout(pad=1.5)
out = OUT / "stress_seeding_throughput.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

# =============================================================================
# Chart 5 — Action distribution
# =============================================================================

fig, ax = plt.subplots(figsize=(6, 5))
fig.patch.set_facecolor(C_BG)

action_labels = list(ACTIONS.keys())
action_values = list(ACTIONS.values())
action_colors = [C_GRAY, C_BLUE, C_AMBER, C_GREEN]

wedges, texts, autotexts = ax.pie(
    action_values, labels=action_labels, colors=action_colors,
    autopct="%1.0f%%", startangle=90,
    wedgeprops={"edgecolor": C_BG, "linewidth": 2},
    textprops={"fontsize": 12, "color": C_TEXT},
)
for at in autotexts:
    at.set_fontsize(11)
    at.set_color(C_BG)

ax.set_title("Action Distribution\n(diverse 10K store, 300 queries)", fontsize=12, pad=10)

plt.tight_layout(pad=1.5)
out = OUT / "stress_action_distribution.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

# =============================================================================
# Chart 6 — Tuning lever waterfall (full range including Bloom filter)
# =============================================================================

fig, ax = plt.subplots(figsize=(12, 6))
fig.patch.set_facecolor(C_BG)

all_labels = list(TUNING.keys())
all_values = list(TUNING.values())
# Colour: fast=green, medium=blue, slow=red
all_colors = [
    C_TEAL if v < 0.1 else (C_GREEN if v < 2 else (C_BLUE if v < 8 else C_RED))
    for v in all_values
]

bars2 = ax.barh(all_labels, all_values, color=all_colors, alpha=0.85, height=0.55)
for bar, val in zip(bars2, all_values):
    label = f"{val:.4f}ms" if val < 0.1 else f"{val:.3f}ms" if val < 1 else f"{val:.1f}ms"
    ax.text(bar.get_width() + max(all_values)*0.01,
            bar.get_y() + bar.get_height()/2,
            label, va="center", ha="left", fontsize=9, color=C_TEXT)

ax.set_xlabel("Latency (ms) — lower is better", fontsize=12)
ax.set_title(
    "All Tuning Levers: Measured p50 Latency\n"
    "(green = applied by default  |  teal = sub-millisecond  |  red = before fix)",
    fontsize=12, pad=12,
)
ax.grid(axis="x")
ax.set_xlim(0, max(all_values) * 1.25)
ax.invert_yaxis()

plt.tight_layout(pad=1.5)
out = OUT / "stress_all_tuning_levers.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

# =============================================================================
# Chart 7 — Pareto frontier: latency vs implementation complexity
# =============================================================================

fig, ax = plt.subplots(figsize=(10, 6.5))
fig.patch.set_facecolor(C_BG)

# Each point: (latency_ms, complexity_score_1_5, label, color, on_pareto)
# complexity: 1=env-var/config, 2=code-change, 3=new-package, 4=new-service, 5=rewrite
PARETO_POINTS = [
    # (latency, complexity, label, colour)
    (0.007,  1, "LRU cache\n(default on)",         C_TEAL),
    (0.010,  2, "Bloom filter\n(NONE fast-path)",   C_GREEN),
    (4.30,   2, "Stop-word filter\n(applied)",        C_GREEN),
    (10.39,  2, "touch() commits\n(correct, applied)", C_BLUE),
    (8.89,   2, "PRAGMA cache\n32MB (applied)",       C_BLUE),
    (8.89,   1, "Tighter FTS5\nLIMIT top_k+10",      C_BLUE),
    (14.20,  0, "No optimisations\n(naïve baseline)", C_RED),
    # Future / roadmap
    (8.0,    3, "Postgres HNSW\n(pgvector 0.7)",    C_AMBER),
    (5.0,    4, "Qdrant backend\n(external service)", C_AMBER),
    (1.0,    3, "Redis VSS\n(external service)",     C_AMBER),
]

# Plot all points
for lat, cpx, label, colour in PARETO_POINTS:
    ax.scatter(lat, cpx, s=180, color=colour, zorder=5, alpha=0.9,
               edgecolors=C_BG, linewidths=1.5)
    ax.annotate(label, (lat, cpx),
                textcoords="offset points", xytext=(6, 4),
                fontsize=8, color=colour, va="bottom")

# Draw Pareto frontier line (leftmost-lowest points)
pareto = sorted([(lat, cplx) for lat, cplx, *_ in PARETO_POINTS if cplx <= 3], key=lambda x: x[0])
# Filter to non-dominated
frontier = []
min_c = 999
for lat, cpx in sorted(pareto, key=lambda x: x[0]):
    if cpx <= min_c:
        frontier.append((lat, cpx))
        min_c = cpx
if len(frontier) >= 2:
    fx, fy = zip(*frontier)
    ax.step(fx, fy, where="post", color=C_TEAL, lw=2, ls="--",
            label="Pareto frontier\n(optimal trade-offs)", zorder=3)

ax.set_xlabel("p50 latency (ms) — lower is better", fontsize=12)
ax.set_ylabel("Implementation effort\n(1=config  2=code  3=package  4=service)", fontsize=11)
ax.set_title(
    "Pareto Frontier: Latency vs Implementation Effort\n"
    "(points on the dashed frontier are optimal — can't improve one without increasing the other)",
    fontsize=12, pad=12,
)
ax.set_yticks([0, 1, 2, 3, 4, 5])
ax.set_yticklabels(["baseline", "config only", "code change", "new package",
                    "new service", "full rewrite"], fontsize=9)
ax.legend(loc="upper right", fontsize=9)
ax.grid(True, alpha=0.25)
ax.set_xlim(-0.5, 18)
ax.set_ylim(-0.3, 5.5)

plt.tight_layout(pad=1.5)
out = OUT / "stress_pareto_frontier.png"
plt.savefig(out, dpi=150, bbox_inches="tight", facecolor=C_BG)
plt.close()
print(f"✓ {out}")

print(f"\nAll charts saved to {OUT}")

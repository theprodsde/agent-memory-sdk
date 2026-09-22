"""Minimal status-page HTML for the FastAPI server's GET /.

This is intentionally thin — it shows live stats and links out to the
full Streamlit dashboard (``agent-memory-dashboard``).
"""
from __future__ import annotations

from typing import Any


def render_dashboard(stats: dict[str, Any]) -> str:
    total = stats.get("total", 0)
    by_state: dict[str, int] = stats.get("by_state", {})
    by_type: dict[str, int] = stats.get("by_type", {})
    total_access: int = stats.get("total_access_count", 0)

    state_rows = "".join(
        f"""<tr>
              <td class="label">{s}</td>
              <td class="num">{c}</td>
              <td><div class="bar" style="width:{min(100, round(c/max(total,1)*100))}%"></div></td>
            </tr>"""
        for s, c in sorted(by_state.items())
    ) or '<tr><td colspan="3" class="empty">No data</td></tr>'

    type_rows = "".join(
        f"""<tr>
              <td class="label">{t}</td>
              <td class="num">{c}</td>
              <td><div class="bar" style="width:{min(100, round(c/max(total,1)*100))}%"></div></td>
            </tr>"""
        for t, c in sorted(by_type.items(), key=lambda x: -x[1])
    ) or '<tr><td colspan="3" class="empty">No data</td></tr>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Agent Memory · Status</title>
  <style>
    :root {{
      --bg: #0f172a; --surface: #1e293b; --border: #334155;
      --text: #f1f5f9; --muted: #94a3b8;
      --accent: #6366f1; --accent-light: #818cf8;
      --green: #22c55e; --amber: #f59e0b; --red: #ef4444; --gray: #6b7280;
    }}
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Inter', 'Segoe UI', sans-serif;
      background: var(--bg); color: var(--text);
      min-height: 100vh; padding: 32px 24px;
    }}
    header {{ display: flex; align-items: center; gap: 12px; margin-bottom: 32px; }}
    header h1 {{ font-size: 1.5rem; font-weight: 700; }}
    .badge {{
      background: var(--accent); color: #fff;
      font-size: 0.7rem; font-weight: 600; padding: 2px 8px;
      border-radius: 999px; letter-spacing: .5px; text-transform: uppercase;
    }}
    .tiles {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 32px; }}
    .tile {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; padding: 20px 24px; min-width: 140px; flex: 1;
    }}
    .tile .label {{ font-size: .75rem; color: var(--muted); text-transform: uppercase;
                    letter-spacing: .6px; margin-bottom: 8px; }}
    .tile .value {{ font-size: 2.25rem; font-weight: 700; line-height: 1; }}
    .tile.accent {{ border-color: var(--accent); }}
    .tables {{ display: flex; flex-wrap: wrap; gap: 24px; margin-bottom: 32px; }}
    .card {{
      background: var(--surface); border: 1px solid var(--border);
      border-radius: 12px; overflow: hidden; flex: 1; min-width: 260px;
    }}
    .card h2 {{ padding: 16px 20px; font-size: .85rem; font-weight: 600;
                color: var(--muted); text-transform: uppercase; letter-spacing: .6px;
                border-bottom: 1px solid var(--border); }}
    table {{ width: 100%; border-collapse: collapse; }}
    td {{ padding: 10px 20px; font-size: .875rem; }}
    td.label {{ color: var(--text); width: 45%; }}
    td.num {{ color: var(--accent-light); font-variant-numeric: tabular-nums; width: 15%; }}
    .bar {{ height: 6px; background: var(--accent); border-radius: 3px;
             opacity: .7; min-width: 2px; }}
    tr:not(:last-child) td {{ border-bottom: 1px solid var(--border); }}
    .empty {{ color: var(--muted); font-style: italic; padding: 20px; }}
    .actions {{
      display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 32px;
    }}
    a.btn {{
      display: inline-flex; align-items: center; gap: 6px;
      padding: 10px 20px; border-radius: 8px; font-size: .875rem;
      font-weight: 500; text-decoration: none; transition: opacity .15s;
    }}
    a.btn:hover {{ opacity: .85; }}
    a.btn.primary {{ background: var(--accent); color: #fff; }}
    a.btn.secondary {{
      background: transparent; color: var(--accent-light);
      border: 1px solid var(--border);
    }}
    .tip {{
      background: var(--surface); border: 1px solid var(--border);
      border-left: 3px solid var(--accent); border-radius: 8px;
      padding: 14px 18px; font-size: .85rem; color: var(--muted);
    }}
    .tip code {{
      background: var(--bg); color: var(--accent-light);
      padding: 1px 6px; border-radius: 4px; font-size: .8rem;
    }}
  </style>
</head>
<body>
  <header>
    <span style="font-size:2rem">🧠</span>
    <div>
      <h1>Agent Memory SDK</h1>
      <span style="color:var(--muted);font-size:.85rem">REST API · Status page</span>
    </div>
    <span class="badge">v0.3.0</span>
  </header>

  <div class="tiles">
    <div class="tile accent">
      <div class="label">Total memories</div>
      <div class="value">{total}</div>
    </div>
    <div class="tile">
      <div class="label">Active</div>
      <div class="value" style="color:var(--green)">{by_state.get("active",0)}</div>
    </div>
    <div class="tile">
      <div class="label">Archived</div>
      <div class="value" style="color:var(--amber)">{by_state.get("archived",0)}</div>
    </div>
    <div class="tile">
      <div class="label">Expired</div>
      <div class="value" style="color:var(--red)">{by_state.get("expired",0)}</div>
    </div>
    <div class="tile">
      <div class="label">Total accesses</div>
      <div class="value">{total_access}</div>
    </div>
  </div>

  <div class="tables">
    <div class="card">
      <h2>By State</h2>
      <table>
        <tbody>{state_rows}</tbody>
      </table>
    </div>
    <div class="card">
      <h2>By Type</h2>
      <table>
        <tbody>{type_rows}</tbody>
      </table>
    </div>
  </div>

  <div class="actions">
    <a class="btn primary" href="/docs">📖 API Docs (Swagger)</a>
    <a class="btn secondary" href="/redoc">ReDoc</a>
    <a class="btn secondary" href="/stats">Stats JSON</a>
  </div>

  <div class="tip">
    <strong>Full dashboard:</strong>
    install the dashboard extra and run
    <code>pip install agent-memory-sdk[dashboard]</code>
    then <code>agent-memory-dashboard</code> for the interactive Streamlit UI
    (charts, memory browser, resolve sandbox, graph explorer).
  </div>
</body>
</html>"""

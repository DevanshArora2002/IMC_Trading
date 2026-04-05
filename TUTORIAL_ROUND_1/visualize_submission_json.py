#!/usr/bin/env python3
"""
Load an IMC Prosperity submission / run JSON and emit a static HTML report
with charts (no third-party deps). Usage:

  python3 visualize_submission_json.py 28169.json
  python3 visualize_submission_json.py 28169.json -o my_report.html
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
from pathlib import Path
from typing import Any


def _polyline_points(xs: list[float], ys: list[float], w: float, h: float, pad: float) -> str:
    if not xs:
        return ""
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    if max_x == min_x:
        max_x = min_x + 1
    if max_y == min_y:
        max_y = min_y + 1
    inner_w, inner_h = w - 2 * pad, h - 2 * pad
    parts: list[str] = []
    for x, y in zip(xs, ys):
        px = pad + (x - min_x) / (max_x - min_x) * inner_w
        py = pad + (1 - (y - min_y) / (max_y - min_y)) * inner_h
        parts.append(f"{px:.2f},{py:.2f}")
    return " ".join(parts)


def _svg_chart(
    title: str,
    xs: list[float],
    series: list[tuple[str, list[float]]],
    width: float = 720,
    height: float = 260,
) -> str:
    pad = 36.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#ca8a04", "#9333ea"]
    lines: list[str] = []
    all_y = [y for _, ys in series for y in ys]
    min_y, max_y = (min(all_y), max(all_y)) if all_y else (0.0, 1.0)
    if min_y == max_y:
        max_y = min_y + 1e-9
    min_x, max_x = min(xs), max(xs)
    if min_x == max_x:
        max_x = min_x + 1

    def tx(x: float) -> float:
        return pad + (x - min_x) / (max_x - min_x) * (width - 2 * pad)

    def ty(y: float) -> float:
        return pad + (1 - (y - min_y) / (max_y - min_y)) * (height - 2 * pad)

    for i, (name, ys) in enumerate(series):
        pts = " ".join(f"{tx(x):.2f},{ty(y):.2f}" for x, y in zip(xs, ys))
        color = colors[i % len(colors)]
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2" '
            f'points="{pts}"/><text x="{width - pad}" y="{pad + 14 + i * 16}" '
            f'font-size="11" text-anchor="end" fill="{color}">{name}</text>'
        )

    return f"""<figure class="chart">
<h3>{title}</h3>
<svg viewBox="0 0 {width} {height}" width="100%" style="max-width:{width}px">
  <rect x="0" y="0" width="{width}" height="{height}" fill="#fafafa" stroke="#e5e5e5"/>
  <text x="{pad}" y="{height - 8}" font-size="10" fill="#666">x: time (timestamp)</text>
  <text x="8" y="{pad}" font-size="10" fill="#666" transform="rotate(-90 8,{pad})">value</text>
  {"".join(lines)}
</svg>
</figure>"""


def build_html(data: dict[str, Any], source_name: str) -> str:
    activities = list(csv.DictReader(io.StringIO(data["activitiesLog"]), delimiter=";"))
    graph_rows = list(csv.DictReader(io.StringIO(data["graphLog"]), delimiter=";"))

    g_ts = [float(r["timestamp"]) for r in graph_rows]
    g_val = [float(r["value"]) for r in graph_rows]

    products: dict[str, list[tuple[float, float]]] = {}
    for r in activities:
        p = r["product"]
        t = float(r["timestamp"])
        mid = float(r["mid_price"]) if r.get("mid_price") not in (None, "") else float("nan")
        products.setdefault(p, []).append((t, mid))

    pos_rows = data.get("positions") or []
    pos_html = "<ul>" + "".join(
        f"<li><code>{x['symbol']}</code>: {x['quantity']}</li>" for x in pos_rows
    ) + "</ul>"

    mid_charts = []
    for pname, pts in sorted(products.items()):
        xs = [t for t, _ in pts]
        ys = [m for _, m in pts]
        mid_charts.append(_svg_chart(f"Mid price — {pname}", xs, [(pname, ys)]))

    pnl_chart = _svg_chart("Cumulative P&amp;L (from graphLog)", g_ts, [("P&amp;L", g_val)])

    # Per-timestamp P&amp;L from last row of each product (same timestamp appears twice)
    pnl_by_t: dict[float, float] = {}
    for r in activities:
        t = float(r["timestamp"])
        pnl = float(r["profit_and_loss"])
        pnl_by_t[t] = pnl  # same for both products per tick in this file
    pnl_ts = sorted(pnl_by_t)
    pnl_series = [pnl_by_t[t] for t in pnl_ts]
    pnl_snap_chart = _svg_chart("P&amp;L at each snapshot (activitiesLog)", pnl_ts, [("snapshot P&amp;L", pnl_series)])

    structure = """
    <pre class="diagram">
submission JSON
├── round          (e.g. tutorial round id)
├── status         (e.g. FINISHED)
├── profit         (final score / P&amp;L)
├── positions      (symbol → quantity at end)
├── graphLog       CSV: timestamp;value  → cumulative P&amp;L curve for the scoreboard
└── activitiesLog  CSV: per-timestamp order book + mid + P&amp;L columns
    </pre>
    """

    next_steps = """
    <h2>What you can do next with this file</h2>
    <ol>
      <li><strong>Compare to your bot</strong> — Align timestamps with your own logs to see
          when inventory or quotes diverged from the replay.</li>
      <li><strong>Spread and mid</strong> — From bid/ask levels, compute spread and depth;
          check whether your strategy was posting inside or outside the recorded book.</li>
      <li><strong>P&amp;L vs mid</strong> — If <code>profit_and_loss</code> moves with mid,
          you are likely marked on inventory; large jumps often mean fills.</li>
      <li><strong>Multiple runs</strong> — Diff two JSONs (profit, final positions, graphLog)
          after code changes to see regressions quickly.</li>
    </ol>
    """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Run report: {source_name}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; max-width: 900px; color: #111; }}
    h1 {{ font-size: 1.35rem; }}
    h2 {{ font-size: 1.1rem; margin-top: 2rem; }}
    h3 {{ font-size: 0.95rem; margin: 0 0 8px; }}
    .meta {{ background: #f4f4f5; padding: 12px 16px; border-radius: 8px; margin: 16px 0; }}
    .meta code {{ background: #e4e4e7; padding: 2px 6px; border-radius: 4px; }}
    pre.diagram {{ background: #18181b; color: #e4e4e7; padding: 16px; border-radius: 8px;
      font-size: 12px; overflow-x: auto; line-height: 1.45; }}
    figure.chart {{ margin: 24px 0; }}
    ol li {{ margin: 0.4em 0; }}
  </style>
</head>
<body>
  <h1>Submission / run visualization</h1>
  <p>Source: <code>{source_name}</code></p>
  <div class="meta">
    <strong>round</strong> <code>{data.get("round")}</code> &nbsp;|&nbsp;
    <strong>status</strong> <code>{data.get("status")}</code> &nbsp;|&nbsp;
    <strong>profit</strong> <code>{data.get("profit")}</code>
    <h2>Final positions</h2>
    {pos_html}
  </div>
  <h2>What is inside this JSON?</h2>
  {structure}
  {next_steps}
  <h2>Charts</h2>
  {pnl_chart}
  {pnl_snap_chart}
  {"".join(mid_charts)}
  <p style="color:#666;font-size:13px;margin-top:32px">
    Generated by <code>visualize_submission_json.py</code> (stdlib only). Open this file in a browser.
  </p>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser(description="Build HTML visualization for Prosperity run JSON.")
    ap.add_argument("json_path", type=Path, help="Path to e.g. 28169.json")
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output HTML path (default: <stem>_report.html next to JSON)",
    )
    args = ap.parse_args()
    path: Path = args.json_path
    out = args.output or path.with_name(f"{path.stem}_report.html")
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    html = build_html(data, path.name)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out.resolve()}")


if __name__ == "__main__":
    main()

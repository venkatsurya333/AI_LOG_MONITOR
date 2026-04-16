"""
report.py — Automated daily HTML anomaly report generator.
Produces a self-contained, single-file HTML report including:
  - Executive summary
  - Severity breakdown with visual bars
  - Hourly heatmap
  - Top anomalies table
  - Cluster summaries
  - Trend insights
  - Model metadata
Run standalone or import generate_report() from other modules.
"""

import os
import html
import time
import pandas as pd
from datetime import datetime

from config import ANOMALY_RESULTS_CSV
from trend_analysis import analyze_trends
from clustering import cluster_anomalies
from logger import log_info, log_error


SEVERITY_COLOR = {
    "CRITICAL": ("#f85149", "#3d0a08"),
    "HIGH":     ("#d29922", "#3d2a00"),
    "MEDIUM":   ("#58a6ff", "#031d4a"),
    "LOW":      ("#3fb950", "#07290d"),
    "INFO":     ("#8b949e", "#1c2128"),
}


def _load(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        log_info(f"Report: loaded {path} ({len(df)} rows)")
        return df
    except Exception as exc:
        log_error(f"Report: cannot load {path}: {exc}")
        return pd.DataFrame()


def _severity_bars_html(sev_counts: dict) -> str:
    total = sum(sev_counts.values()) or 1
    rows  = []
    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"):
        cnt = sev_counts.get(level, 0)
        pct = cnt / total * 100
        color, bg = SEVERITY_COLOR.get(level, ("#8b949e", "#1c2128"))
        rows.append(f"""
        <div class="bar-row">
          <span class="sev-label" style="color:{color}">{level}</span>
          <div class="bar-track">
            <div class="bar-fill" style="width:{pct:.1f}%;background:{color}"></div>
          </div>
          <span class="bar-count">{cnt}</span>
        </div>""")
    return "\n".join(rows)


def _heatmap_html(hourly: dict) -> str:
    mx   = max(hourly.values(), default=1)
    cols = []
    for h in range(24):
        v    = hourly.get(h, 0)
        pct  = v / mx if mx else 0
        if pct > 0.75:
            color = "#f85149"
        elif pct > 0.4:
            color = "#d29922"
        else:
            color = "#3fb950"
        opacity = 0.15 + pct * 0.85
        cols.append(
            f'<div class="hm-cell" style="background:{color};opacity:{opacity:.2f}" '
            f'title="{h:02d}:00 — {v} anomalies"></div>'
        )
    labels = "".join(
        f'<span class="hm-label">{h:02d}:00</span>'
        for h in (0, 6, 12, 18, 23)
    )
    return f"""
    <div class="hm-grid">{"".join(cols)}</div>
    <div class="hm-labels">{labels}</div>"""


def _top_anomalies_html(anomalies: pd.DataFrame, n: int = 20) -> str:
    if anomalies.empty:
        return "<p>No anomalies found.</p>"

    cols = ["timestamp", "severity", "anomaly_score_norm", "message"]
    cols = [c for c in cols if c in anomalies.columns]

    top  = anomalies.sort_values(
        "anomaly_score_norm" if "anomaly_score_norm" in anomalies.columns
        else anomalies.columns[0],
        ascending=False
    ).head(n)

    rows = ""
    for _, row in top.iterrows():
        sev   = str(row.get("severity", "INFO"))
        color, _ = SEVERITY_COLOR.get(sev, ("#8b949e", "#1c2128"))
        ts    = str(row.get("timestamp", row.get("@timestamp", "N/A")))[:19]
        score = f"{float(row.get('anomaly_score_norm', 0)):.4f}"
        msg   = html.escape(str(row.get("message", ""))[:120])
        rows += f"""
        <tr>
          <td class="td-mono">{ts}</td>
          <td><span class="sev-badge" style="color:{color};border-color:{color}40">{sev}</span></td>
          <td class="td-mono td-score">{score}</td>
          <td class="td-msg">{msg}</td>
        </tr>"""
    return f"""
    <table class="anomaly-table">
      <thead><tr><th>Timestamp</th><th>Severity</th><th>Score</th><th>Message</th></tr></thead>
      <tbody>{rows}</tbody>
    </table>"""


def _clusters_html(clusters: list[dict]) -> str:
    if not clusters:
        return "<p>No cluster data available.</p>"
    cards = ""
    for c in clusters:
        sev   = c.get("dominant_severity", "INFO")
        color, bg = SEVERITY_COLOR.get(sev, ("#8b949e","#1c2128"))
        kws   = " ".join(
            f'<span class="keyword">{html.escape(k)}</span>'
            for k in c.get("top_keywords", [])[:6]
        )
        sample = html.escape(c.get("sample_msgs", [""])[0][:140]) if c.get("sample_msgs") else ""
        cards += f"""
        <div class="cluster-card">
          <div class="cluster-header">
            <span class="cluster-id">cluster_{c['cluster_id']}</span>
            <span class="sev-badge" style="color:{color};border-color:{color}40">{sev}</span>
            <span class="cluster-size">{c['size']} anomalies</span>
          </div>
          <div class="keywords">{kws}</div>
          <div class="cluster-sample">{sample}</div>
        </div>"""
    return cards


def _trend_insights_html(insights: list[str]) -> str:
    items = "".join(f"<li>{html.escape(i)}</li>" for i in insights)
    return f"<ul class='insights-list'>{items}</ul>"


def generate_report(
    df: pd.DataFrame | None = None,
    output_path: str | None = None,
) -> str:
    """
    Generate a self-contained HTML anomaly report.

    Args:
        df:          anomaly_results DataFrame. Loads ANOMALY_RESULTS_CSV if None.
        output_path: where to save the report. Default: report_YYYYMMDD_HHMMSS.html

    Returns:
        Path to the saved HTML file.
    """
    if df is None:
        df = _load(ANOMALY_RESULTS_CSV)
    if df.empty:
        log_error("No data to generate report from")
        return ""

    if output_path is None:
        tag = time.strftime("%Y%m%d_%H%M%S")
        output_path = f"report_{tag}.html"

    anomalies = df[df.get("anomaly", pd.Series(dtype=int)) == -1] \
                if "anomaly" in df.columns else df
    sev_counts = anomalies["severity"].value_counts().to_dict() \
                 if "severity" in anomalies.columns else {}

    trend_data = analyze_trends(df=df)
    clusters   = cluster_anomalies(df=df)

    total    = len(df)
    n_anom   = len(anomalies)
    rate     = n_anom / total * 100 if total else 0
    gen_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # ── HTML template ─────────────────────────────────────────────────────────
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AI Log Monitor — Anomaly Report {time.strftime("%Y-%m-%d")}</title>
<style>
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:#0d1117;color:#e6edf3;font-size:13px;line-height:1.6;padding:24px}}
.container{{max-width:1000px;margin:0 auto}}
h1{{font-size:20px;font-weight:600;margin-bottom:4px;color:#fff}}
h2{{font-size:14px;font-weight:500;color:#8b949e;letter-spacing:.06em;
    text-transform:uppercase;margin:28px 0 12px;padding-bottom:6px;
    border-bottom:1px solid #21262d}}
.meta{{font-size:11px;color:#8b949e;margin-bottom:24px;font-family:monospace}}
.grid-4{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:24px}}
.metric{{background:#161b22;border:1px solid #21262d;border-radius:6px;padding:14px 16px}}
.metric-label{{font-size:10px;color:#8b949e;text-transform:uppercase;letter-spacing:.08em}}
.metric-val{{font-size:24px;font-weight:600;margin:4px 0 2px;font-family:monospace}}
.metric-sub{{font-size:10px;color:#8b949e;font-family:monospace}}
.card{{background:#161b22;border:1px solid #21262d;border-radius:6px;
       padding:16px 18px;margin-bottom:16px}}
.bar-row{{display:flex;align-items:center;gap:10px;margin:5px 0}}
.sev-label{{width:72px;font-family:monospace;font-size:11px;font-weight:500}}
.bar-track{{flex:1;height:10px;background:#21262d;border-radius:2px;overflow:hidden}}
.bar-fill{{height:100%;border-radius:2px}}
.bar-count{{width:40px;text-align:right;font-family:monospace;font-size:11px;color:#8b949e}}
.hm-grid{{display:grid;grid-template-columns:repeat(24,1fr);gap:2px;height:36px}}
.hm-cell{{border-radius:2px}}
.hm-labels{{display:flex;justify-content:space-between;margin-top:4px}}
.hm-label{{font-family:monospace;font-size:10px;color:#484f58}}
.anomaly-table{{width:100%;border-collapse:collapse;font-size:11px}}
.anomaly-table th{{text-align:left;padding:6px 8px;border-bottom:1px solid #21262d;
                   color:#8b949e;font-weight:500;font-size:10px;text-transform:uppercase}}
.anomaly-table tr:hover td{{background:#1c2128}}
.anomaly-table td{{padding:5px 8px;border-bottom:1px solid #161b22;vertical-align:top}}
.td-mono{{font-family:monospace;white-space:nowrap;color:#8b949e}}
.td-score{{color:#fff;font-weight:500}}
.td-msg{{color:#8b949e;max-width:400px;word-break:break-word}}
.sev-badge{{font-family:monospace;font-size:10px;font-weight:500;
            padding:1px 7px;border-radius:3px;border:1px solid;white-space:nowrap}}
.cluster-card{{background:#0d1117;border:1px solid #21262d;border-radius:5px;
               padding:12px 14px;margin:8px 0}}
.cluster-header{{display:flex;align-items:center;gap:10px;margin-bottom:7px}}
.cluster-id{{font-family:monospace;font-size:11px;color:#484f58}}
.cluster-size{{font-size:11px;color:#484f58;margin-left:auto}}
.keywords{{margin-bottom:6px}}
.keyword{{display:inline-block;font-family:monospace;font-size:10px;
           background:#21262d;color:#8b949e;border-radius:3px;
           padding:1px 6px;margin:1px}}
.cluster-sample{{font-family:monospace;font-size:10px;color:#484f58;
                  white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.insights-list{{padding-left:18px;color:#8b949e;font-size:12px;line-height:2}}
.insights-list li::marker{{color:#3fb950}}
.footer{{margin-top:32px;font-size:10px;color:#484f58;text-align:center;
         font-family:monospace;border-top:1px solid #21262d;padding-top:12px}}
.trend-badge{{display:inline-block;padding:2px 8px;border-radius:3px;
              font-size:11px;font-family:monospace;font-weight:500}}
.trend-inc{{background:rgba(248,81,73,.15);color:#f85149}}
.trend-dec{{background:rgba(63,185,80,.15);color:#3fb950}}
.trend-sta{{background:rgba(139,148,158,.15);color:#8b949e}}
</style>
</head>
<body>
<div class="container">

<h1>AI Log Monitor — Anomaly Report</h1>
<div class="meta">Generated: {gen_time} &nbsp;|&nbsp; Source: {ANOMALY_RESULTS_CSV}</div>

<div class="grid-4">
  <div class="metric">
    <div class="metric-label">Total Logs</div>
    <div class="metric-val">{total:,}</div>
    <div class="metric-sub">in dataset</div>
  </div>
  <div class="metric">
    <div class="metric-label">Anomalies</div>
    <div class="metric-val" style="color:#d29922">{n_anom:,}</div>
    <div class="metric-sub">{rate:.2f}% anomaly rate</div>
  </div>
  <div class="metric">
    <div class="metric-label">CRITICAL</div>
    <div class="metric-val" style="color:#f85149">{sev_counts.get("CRITICAL",0)}</div>
    <div class="metric-sub">immediate action</div>
  </div>
  <div class="metric">
    <div class="metric-label">Trend</div>
    <div class="metric-val" style="font-size:16px;padding-top:4px">
      <span class="trend-badge {'trend-inc' if trend_data['trend_direction']=='Increasing' else 'trend-dec' if trend_data['trend_direction']=='Decreasing' else 'trend-sta'}">
        {'↗' if trend_data['trend_direction']=='Increasing' else '↘' if trend_data['trend_direction']=='Decreasing' else '→'} {trend_data['trend_direction']}
      </span>
    </div>
    <div class="metric-sub">peak hour {trend_data.get('peak_hour','N/A'):02d}:00</div>
  </div>
</div>

<h2>Severity Breakdown</h2>
<div class="card">
{_severity_bars_html(sev_counts)}
</div>

<h2>Hourly Anomaly Heatmap</h2>
<div class="card">
{_heatmap_html(trend_data.get("hourly_counts", {}))}
</div>

<h2>Trend Insights</h2>
<div class="card">
{_trend_insights_html(trend_data.get("insights", []))}
</div>

<h2>Top {min(n_anom,20)} Anomalies by Score</h2>
<div class="card" style="overflow-x:auto">
{_top_anomalies_html(anomalies)}
</div>

<h2>Root Cause Clusters (KMeans)</h2>
<div class="card">
{_clusters_html(clusters)}
</div>

<div class="footer">
  AI Log Monitor &nbsp;|&nbsp; Isolation Forest + TF-IDF + Gemini
  &nbsp;|&nbsp; {gen_time}
</div>

</div>
</body>
</html>"""

    try:
        with open(output_path, "w", encoding="utf-8") as fh:
            fh.write(html_content)
        log_info(f"Report saved → {output_path}")
        print(f"Report saved → {output_path}")
    except Exception as exc:
        log_error(f"Failed to write report: {exc}")
        raise

    return output_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML anomaly report")
    parser.add_argument("--output", default=None, help="Output HTML path")
    args = parser.parse_args()

    path = generate_report(output_path=args.output)
    if path:
        print(f"Open in browser: file://{os.path.abspath(path)}")

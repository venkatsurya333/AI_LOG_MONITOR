"""
trend_analysis.py — Anomaly trend detection grouped by hour and day.
Detects increasing / decreasing / stable trends and returns
structured insight strings usable by the AI assistant.
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from collections import defaultdict

from config import ANOMALY_RESULTS_CSV
from logger import log_info, log_error


# ── Core trend engine ──────────────────────────────────────────────────────────

def analyze_trends(df: pd.DataFrame | None = None) -> dict:
    """
    Compute anomaly trends over time.

    Args:
        df: DataFrame with anomaly_results. Loads ANOMALY_RESULTS_CSV if None.

    Returns:
        dict with keys:
            hourly_counts    — {hour: count}
            daily_counts     — {weekday_name: count}
            peak_hour        — hour with most anomalies
            peak_day         — weekday with most anomalies
            trend_direction  — "Increasing" | "Decreasing" | "Stable"
            severity_summary — {severity: count}
            insights         — list[str]   (ready for the AI prompt)
    """
    df = _load(df)
    anomalies = df[df.get("anomaly", pd.Series(dtype=int)) == -1].copy() \
                if "anomaly" in df.columns \
                else df.copy()

    if anomalies.empty:
        log_info("No anomalies found for trend analysis")
        return _empty_result()

    anomalies = _ensure_time_cols(anomalies)

    hourly  = anomalies.groupby("hour").size().to_dict()
    daily   = anomalies.groupby("day_name").size().to_dict()

    peak_hour = max(hourly, key=hourly.get, default=0)
    peak_day  = max(daily,  key=daily.get,  default="N/A")

    trend = _compute_trend(anomalies)
    sev   = anomalies["severity"].value_counts().to_dict() \
            if "severity" in anomalies.columns else {}

    insights = _build_insights(hourly, daily, peak_hour, peak_day, trend, sev)

    result = {
        "hourly_counts":    hourly,
        "daily_counts":     daily,
        "peak_hour":        peak_hour,
        "peak_day":         peak_day,
        "trend_direction":  trend,
        "severity_summary": sev,
        "insights":         insights,
    }
    log_info(f"Trend analysis complete | trend={trend} | peak_hour={peak_hour}")
    return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df.copy()
    try:
        df = pd.read_csv(ANOMALY_RESULTS_CSV)
        log_info(f"Loaded {ANOMALY_RESULTS_CSV} for trend analysis ({len(df)} rows)")
        return df
    except Exception as exc:
        log_error(f"Cannot load {ANOMALY_RESULTS_CSV}: {exc}")
        return pd.DataFrame()


def _ensure_time_cols(df: pd.DataFrame) -> pd.DataFrame:
    if "hour" not in df.columns or "day_of_week" not in df.columns:
        ts_col = "@timestamp" if "@timestamp" in df.columns else "timestamp"
        if ts_col in df.columns:
            ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
            df["hour"]        = ts.dt.hour.fillna(0).astype(int)
            df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
        else:
            df["hour"]        = 0
            df["day_of_week"] = 0

    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df["day_name"] = df["day_of_week"].apply(
        lambda d: day_names[int(d)] if 0 <= int(d) <= 6 else "Unknown"
    )
    return df


def _compute_trend(df: pd.DataFrame) -> str:
    """Simple linear trend on hourly anomaly counts."""
    if "hour" not in df.columns:
        return "Stable"

    hourly_series = df.groupby("hour").size()
    if len(hourly_series) < 3:
        return "Stable"

    x = np.arange(len(hourly_series))
    y = hourly_series.values.astype(float)

    # Linear regression slope
    slope = np.polyfit(x, y, 1)[0]

    if slope > 0.5:
        return "Increasing"
    if slope < -0.5:
        return "Decreasing"
    return "Stable"


def _build_insights(
    hourly: dict,
    daily: dict,
    peak_hour: int,
    peak_day: str,
    trend: str,
    sev: dict,
) -> list[str]:
    insights = []

    if hourly:
        insights.append(
            f"Peak anomaly hour: {peak_hour:02d}:00 "
            f"({hourly.get(peak_hour, 0)} anomalies)"
        )

    if daily:
        insights.append(
            f"Most anomalous day: {peak_day} "
            f"({daily.get(peak_day, 0)} anomalies)"
        )

    insights.append(f"Overall anomaly trend: {trend}")

    for level in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
        cnt = sev.get(level, 0)
        if cnt:
            insights.append(f"{level} severity count: {cnt}")

    night_hours = {h: c for h, c in hourly.items() if h in range(0, 6)}
    if night_hours:
        night_total = sum(night_hours.values())
        if night_total > 0:
            insights.append(
                f"Off-hours anomalies (00:00–06:00): {night_total} — "
                "potential unauthorized activity"
            )

    return insights


def _empty_result() -> dict:
    return {
        "hourly_counts":    {},
        "daily_counts":     {},
        "peak_hour":        None,
        "peak_day":         None,
        "trend_direction":  "Stable",
        "severity_summary": {},
        "insights":         ["No anomaly data available for trend analysis."],
    }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = analyze_trends()
    print("\n── Trend Analysis ──────────────────────────")
    for ins in result["insights"]:
        print(f"  • {ins}")
    print(f"\nTrend direction : {result['trend_direction']}")
    print(f"Peak hour       : {result['peak_hour']}")
    print(f"Peak day        : {result['peak_day']}")

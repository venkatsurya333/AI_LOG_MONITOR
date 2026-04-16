"""
detect_anomalies.py — Quick anomaly summary from anomaly_results.csv.
Prints distribution, samples, and saves detected_anomalies.csv.
"""

import pandas as pd
from config import ANOMALY_RESULTS_CSV, DETECTED_ANOMALIES_CSV
from logger import log_info, log_error


def summarise_anomalies(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Load anomaly results, print a summary, and save anomaly-only CSV.

    Returns:
        DataFrame of anomalies only (anomaly == -1).
    """
    if df is None:
        try:
            df = pd.read_csv(ANOMALY_RESULTS_CSV)
            log_info(f"Loaded {ANOMALY_RESULTS_CSV} ({len(df)} rows)")
        except Exception as exc:
            log_error(f"Cannot load {ANOMALY_RESULTS_CSV}: {exc}")
            raise

    if "anomaly" not in df.columns:
        log_error("Column 'anomaly' missing")
        raise ValueError("Missing 'anomaly' column")

    anomalies = df[df["anomaly"] == -1].copy()
    normal    = df[df["anomaly"] ==  1]

    print("\n── Anomaly Detection Summary ─────────────────────────────")
    print(f"  Total logs    : {len(df)}")
    print(f"  Normal        : {len(normal)}")
    print(f"  Anomalies     : {len(anomalies)}")
    print(f"  Anomaly rate  : {len(anomalies)/max(len(df),1)*100:.2f}%")

    if "severity" in anomalies.columns:
        print("\n  Severity Breakdown:")
        for sev, cnt in anomalies["severity"].value_counts().items():
            bar = "█" * min(cnt, 40)
            print(f"    {sev:<10} {cnt:>5}  {bar}")

    if "anomaly_score_norm" in anomalies.columns:
        print(f"\n  Avg norm score : {anomalies['anomaly_score_norm'].mean():.4f}")
        print(f"  Max norm score : {anomalies['anomaly_score_norm'].max():.4f}")

    print("\n  Top 10 Anomaly Samples:")
    cols = ["timestamp" if "timestamp" in anomalies.columns else "@timestamp",
            "severity", "message"]
    cols = [c for c in cols if c in anomalies.columns]

    for _, row in anomalies.head(10).iterrows():
        ts  = row.get("timestamp", row.get("@timestamp", "N/A"))
        sev = row.get("severity", "?")
        msg = str(row.get("message", ""))[:100]
        print(f"    [{sev}] {ts} — {msg}")

    print("──────────────────────────────────────────────────────────\n")

    anomalies.to_csv(DETECTED_ANOMALIES_CSV, index=False)
    log_info(f"Saved {DETECTED_ANOMALIES_CSV} ({len(anomalies)} rows)")
    print(f"  Saved → {DETECTED_ANOMALIES_CSV}")

    return anomalies


if __name__ == "__main__":
    summarise_anomalies()

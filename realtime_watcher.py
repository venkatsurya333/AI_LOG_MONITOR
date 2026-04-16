"""
realtime_watcher.py — Watches Elasticsearch for new logs every N seconds
and immediately runs prediction on new batches.
"""

import time
import re
import gc
import pandas as pd
from elasticsearch import Elasticsearch
from config import ES_HOST, ES_USER, ES_PASS, ES_SOURCE_INDEX
from feature_engineering import engineer_features
from predict import predict_anomalies
from upload import upload_anomalies
from logger import log_info, log_error

WATCH_INTERVAL = 15   # seconds
WINDOW         = "now-5m"

# Regex to strip syslog timestamp prefix:
# "2026-04-13T10:12:13.693780+05:30 kvs-VirtualBox sshd: actual message"
SYSLOG_PREFIX = re.compile(
    r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d+[+-]\d{2}:\d{2}\s+\S+\s+'
)

def clean_message(raw: str) -> str:
    """Strip syslog timestamp+hostname prefix, return clean message text."""
    return SYSLOG_PREFIX.sub('', str(raw)).strip()

def connect():
    return Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        request_timeout=30,
    )

def fetch_recent(es):
    try:
        resp = es.search(
            index=ES_SOURCE_INDEX,
            body={
                "size": 500,
                "query": {
                    "range": {
                        "@timestamp": {"gte": WINDOW, "lte": "now"}
                    }
                },
                "sort": [{"@timestamp": {"order": "asc"}}],
            }
        )
        hits = resp.get("hits", {}).get("hits", [])
        if not hits:
            return pd.DataFrame()

        records = []
        for h in hits:
            src = h.get("_source", {})
            raw_msg = src.get("message", "")
            message = clean_message(raw_msg)
            if not message:
                continue
            records.append({
                "@timestamp": src.get("@timestamp", ""),
                "message":    message,
                "host":       src.get("host", {}).get("name", ""),
                "log_file":   src.get("log", {}).get("file", {}).get("path", ""),
                "input_type": src.get("input", {}).get("type", ""),
                "agent_name": src.get("agent", {}).get("name", ""),
            })

        return pd.DataFrame(records) if records else pd.DataFrame()

    except Exception as exc:
        log_error(f"Fetch error: {exc}")
        return pd.DataFrame()

def run():
    print(f"\n🔴 LIVE MODE — checking every {WATCH_INTERVAL}s (Ctrl+C to stop)\n")
    es = connect()
    print(f"✅ Connected to Elasticsearch")
    print(f"📡 Watching: {ES_SOURCE_INDEX}")
    print(f"⏱  Window: {WINDOW}\n")

    while True:
        df = fetch_recent(es)

        if df.empty:
            print(f"[{time.strftime('%H:%M:%S')}] No new logs in {WINDOW} — watching...")
        else:
            print(f"[{time.strftime('%H:%M:%S')}] {len(df)} new logs — running prediction...")
            try:
                X, df_feat  = engineer_features(df=df, fit_vectorizer=False)
                df_result   = predict_anomalies(df=df_feat)
                anomalies   = df_result[df_result["anomaly"] == -1]
                normals     = df_result[df_result["anomaly"] ==  1]

                if not anomalies.empty:
                    print(f"  ⚠️  {len(anomalies)} anomalies | {len(normals)} normal")
                    print()

                    # Show top 5 by score
                    top = anomalies.sort_values(
                        "anomaly_score_norm", ascending=False
                    ).head(5)

                    for _, row in top.iterrows():
                        sev   = row.get("severity", "?")
                        score = row.get("anomaly_score_norm", 0)
                        msg   = str(row.get("message", ""))[:90]
                        icon  = {"CRITICAL":"🔴","HIGH":"🟠","MEDIUM":"🟡","LOW":"🟢"}.get(sev,"⚪")
                        print(f"    {icon} [{sev:<8}] score={score:.3f} | {msg}")

                    print()
                    upload_anomalies(df=df_result)
                    print(f"  ✅ Uploaded to Elasticsearch + Telegram alert sent")
                else:
                    print(f"  ✅ All {len(df)} logs normal — no anomalies")

            except FileNotFoundError:
                print("  ❌ Model not found — run: python3 train_model.py first")
            except Exception as exc:
                log_error(f"Pipeline error: {exc}")
                print(f"  ❌ Error: {exc}")

        print()
        gc.collect()
        time.sleep(WATCH_INTERVAL)

if __name__ == "__main__":
    run()

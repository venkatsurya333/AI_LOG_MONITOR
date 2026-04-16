"""
fetch_logs.py — Incremental log fetching from Elasticsearch via Filebeat index.
Improved version with robust message extraction and clean dataset generation.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
from elasticsearch import Elasticsearch

from config import (
    ES_HOST, ES_USER, ES_PASS,
    ES_SOURCE_INDEX, FETCH_SIZE,
    FETCH_WINDOW_DEFAULT, LAST_RUN_FILE,
    LOGS_DATASET_CSV,
)
from logger import log_info, log_error, log_warning


# ─────────────────────────────────────────────────────────
# Elasticsearch connection
# ─────────────────────────────────────────────────────────
def _connect() -> Elasticsearch:
    try:
        es = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            verify_certs=False,
            request_timeout=30,
        )
        if not es.ping():
            raise ConnectionError("Elasticsearch ping failed")

        log_info("Connected to Elasticsearch (fetch_logs)")
        return es

    except Exception as exc:
        log_error(f"Elasticsearch connection failed: {exc}")
        raise


# ─────────────────────────────────────────────────────────
# Last-run timestamp handling
# ─────────────────────────────────────────────────────────
def _read_last_timestamp() -> str:
    if os.path.exists(LAST_RUN_FILE):
        try:
            with open(LAST_RUN_FILE, "r") as fh:
                ts = fh.read().strip()
                if ts:
                    return ts
        except Exception as exc:
            log_warning(f"Could not read last_run file: {exc}")

    return FETCH_WINDOW_DEFAULT


def _write_last_timestamp(ts: str) -> None:
    try:
        with open(LAST_RUN_FILE, "w") as fh:
            fh.write(str(ts))
        log_info(f"Updated last run timestamp: {ts}")
    except Exception as exc:
        log_error(f"Could not write last_run file: {exc}")


# ─────────────────────────────────────────────────────────
# Message extraction (VERY IMPORTANT)
# ─────────────────────────────────────────────────────────
def _extract_message(source: dict) -> str:
    import re

    candidates = [
        source.get("message"),
        source.get("log", {}).get("original"),
        source.get("event", {}).get("original"),
    ]

    raw = ""
    for candidate in candidates:
        if candidate and str(candidate).strip():
            raw = str(candidate).strip()
            break

    if not raw:
        return ""

    # Skip Filebeat internal monitoring noise entirely
    noise_keywords = [
        '"log.level"', '"log.logger"', '"monitoring"',
        'Non-zero metrics', 'logSnapshot', 'libbeat',
        'filebeat[', 'harvester', 'pipeline.go',
    ]
    if any(kw in raw for kw in noise_keywords):
        return ""

    # Strip syslog prefix:
    # "2026-04-13T12:07:04.175259+05:30 kvs-VirtualBox sshd: actual message"
    syslog_pattern = re.compile(
        r'^\d{4}-\d{2}-\d{2}T[\d:\.]+[+-]\d{2}:\d{2}\s+'
        r'[\w\-\.]+\s+'
        r'[\w\-\.\[\]]+:\s*'
    )
    cleaned = syslog_pattern.sub('', raw).strip()

    return cleaned if cleaned else raw
# ─────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────
def fetch_logs() -> pd.DataFrame:
    """
    Fetch logs incrementally from Elasticsearch.
    """

    es = _connect()
    last_ts = _read_last_timestamp()

    log_info(f"Fetching logs from: {last_ts}")

    query = {
        "size": FETCH_SIZE,
        "query": {
            "range": {
                "@timestamp": {
                    "gte": last_ts,
                    "lte": "now"
                }
            }
        },
        "sort": [{"@timestamp": {"order": "asc"}}]
    }

    try:
        response = es.search(index=ES_SOURCE_INDEX, body=query)
        log_info("Elasticsearch query executed")
    except Exception as exc:
        log_error(f"Search failed: {exc}")
        raise

    hits = response.get("hits", {}).get("hits", [])

    if not hits:
        log_info("No new logs found")
        return pd.DataFrame()

    records = []
    skipped = 0

    for hit in hits:
        source = hit.get("_source", {})

        message = _extract_message(source)

        if not message:
            skipped += 1
            continue

        record = {
            "@timestamp": source.get("@timestamp", ""),
            "message": message,
            "message_short": message[:200],   # 🔥 IMPORTANT FIX
            "host": source.get("host", {}).get("name", ""),
            "log_file": source.get("log", {}).get("file", {}).get("path", ""),
            "input_type": source.get("input", {}).get("type", ""),
            "agent_name": source.get("agent", {}).get("name", ""),
        }

        records.append(record)

    if skipped:
        log_warning(f"Skipped {skipped} logs with empty message")

    if not records:
        log_warning("No valid logs found after processing")
        return pd.DataFrame()

    df = pd.DataFrame(records)

    # remove duplicates
    before = len(df)
    df = df.drop_duplicates(subset=["@timestamp", "message"])
    after = len(df)

    if before != after:
        log_info(f"Removed {before - after} duplicate logs")

    log_info(f"Final logs fetched: {len(df)}")

    # save dataset
    df.to_csv(LOGS_DATASET_CSV, index=False)
    log_info(f"Saved dataset → {LOGS_DATASET_CSV}")

    # update last timestamp
    latest_ts = df["@timestamp"].max()
    _write_last_timestamp(latest_ts)

    # preview
    log_info("Sample logs:")
    for msg in df["message_short"].head(3):
        log_info(f"→ {msg}")

    return df


# ─────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = fetch_logs()

    if df.empty:
        print("No new logs fetched.")
    else:
        print(f"\nFetched {len(df)} logs successfully.")
        print(df.head())

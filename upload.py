"""
upload.py — Upload detected anomalies to Elasticsearch.
Uses MD5-based document IDs to prevent duplicates.
Fires Telegram alerts for CRITICAL and HIGH severity events.
"""

import warnings
warnings.filterwarnings("ignore")

import hashlib
import pandas as pd
from elasticsearch import Elasticsearch

from config import (
    ES_HOST, ES_USER, ES_PASS,
    ES_ANOMALY_INDEX,
    ANOMALY_RESULTS_CSV,
)
from alerts import send_alert
from logger import log_info, log_error, log_warning


def _connect() -> Elasticsearch:
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False,
        request_timeout=30,
    )
    if not es.ping():
        raise ConnectionError("Elasticsearch ping failed")
    return es


def _make_doc_id(timestamp: str, message: str) -> str:
    return hashlib.md5(f"{timestamp}{message}".encode()).hexdigest()


def _parse_timestamp(raw: str) -> str | None:
    try:
        return pd.to_datetime(raw).isoformat()
    except Exception:
        return None


def upload_anomalies(df: pd.DataFrame | None = None) -> tuple[int, int]:
    """
    Upload anomalies from DataFrame (or ANOMALY_RESULTS_CSV) to Elasticsearch.

    Returns:
        (uploaded_count, skipped_count)
    """
    if df is None:
        try:
            df = pd.read_csv(ANOMALY_RESULTS_CSV)
            log_info(f"Loaded {ANOMALY_RESULTS_CSV} ({len(df)} rows)")
        except Exception as exc:
            log_error(f"Cannot load {ANOMALY_RESULTS_CSV}: {exc}")
            raise

    if "anomaly" not in df.columns:
        log_error("Column 'anomaly' missing from DataFrame")
        raise ValueError("Missing 'anomaly' column")

    anomalies = df[df["anomaly"] == -1].copy()
    total = len(anomalies)
    log_info(f"Anomalies to upload: {total}")

    if total == 0:
        log_info("No anomalies to upload")
        return 0, 0

    es = _connect()
    log_info("Connected to Elasticsearch (upload)")

    uploaded = 0
    skipped  = 0

    for _, row in anomalies.iterrows():
        try:
            message   = str(row.get("message", ""))
            raw_ts    = str(row.get("@timestamp", row.get("timestamp", ""))).strip()
            timestamp = _parse_timestamp(raw_ts)

            if not timestamp:
                log_warning(f"Invalid timestamp '{raw_ts}' — skipping row")
                skipped += 1
                continue

            doc_id   = _make_doc_id(timestamp, message)
            severity = str(row.get("severity", "INFO"))
            score    = float(row.get("anomaly_score",      0.0))
            norm     = float(row.get("anomaly_score_norm", 0.0))

            doc = {
                "timestamp":          timestamp,
                "message":            message,
                "severity":           severity,
                "anomaly":            int(row.get("anomaly", 0)),
                "anomaly_score":      score,
                "anomaly_score_norm": norm,
                "msg_length":         int(row.get("msg_length",       0)),
                "error_flag":         int(row.get("contains_error",   0)),
                "login_flag":         int(row.get("contains_login",   0)),
                "warning_flag":       int(row.get("contains_warning", 0)),
                "digit_count":        int(row.get("digit_count",      0)),
                "word_count":         int(row.get("word_count",       0)),
                "hour":               int(row.get("hour",             0)),
                "day_of_week":        int(row.get("day_of_week",      0)),
		"message_short": row["message_short"],
            }

            es.index(index=ES_ANOMALY_INDEX, id=doc_id, document=doc)
            uploaded += 1

            # Alert only for actionable severities
            if severity in ("CRITICAL", "HIGH"):
                send_alert(
                    message=message,
                    severity=severity,
                    score=norm,
                    timestamp=timestamp,
                )

            log_info(f"Uploaded {doc_id} | {severity} | score={norm:.4f}")

        except Exception as exc:
            log_error(f"Row upload error: {exc}")
            skipped += 1

    log_info(f"Upload complete — uploaded={uploaded} skipped={skipped}")
    return uploaded, skipped


if __name__ == "__main__":
    up, sk = upload_anomalies()
    print(f"Uploaded: {up} | Skipped: {sk}")

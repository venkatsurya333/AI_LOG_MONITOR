import warnings
warnings.filterwarnings("ignore")

from logger import log_info, log_error
from elasticsearch import Elasticsearch
import pandas as pd
import hashlib
from dotenv import load_dotenv
import os
from alerts import send_alert

# -------------------------
# LOAD ENV VARIABLES
# -------------------------
load_dotenv()

ES_HOST = os.getenv("ES_HOST")
ES_USER = os.getenv("ES_USER")
ES_PASS = os.getenv("ES_PASS")

# -------------------------
# CONNECT TO ELASTICSEARCH
# -------------------------
try:
    es = Elasticsearch(
        ES_HOST,
        basic_auth=(ES_USER, ES_PASS),
        verify_certs=False
    )
    log_info("Connected to Elasticsearch (upload_anomalies)")
except Exception as e:
    log_error(f"Elasticsearch connection failed: {e}")
    exit()

# -------------------------
# LOAD DATA
# -------------------------
try:
    df = pd.read_csv("anomaly_results.csv")
    log_info("Loaded anomaly_results.csv")
except Exception as e:
    log_error(f"Error loading file: {e}")
    exit()

# -------------------------
# FILTER ANOMALIES
# -------------------------
if "anomaly" not in df.columns:
    log_error("Column 'anomaly' missing")
    exit()

anomalies = df[df["anomaly"] == -1]

log_info(f"Total anomalies to upload: {len(anomalies)}")

# -------------------------
# UPLOAD LOOP
# -------------------------
count = 0
skipped = 0

for _, row in anomalies.iterrows():

    try:
        # -------------------------
        # SAFE FIELD EXTRACTION
        # -------------------------
        message = str(row.get("message", ""))

        timestamp = str(row.get("@timestamp", "")).strip()

        try:
            timestamp = pd.to_datetime(timestamp).isoformat()
        except:
            skipped += 1
            continue

        # -------------------------
        # CREATE UNIQUE ID
        # -------------------------
        unique_string = timestamp + message
        doc_id = hashlib.md5(unique_string.encode()).hexdigest()

        # -------------------------
        # FLAGS
        # -------------------------
        error_flag = int(row.get("contains_error", 0))
        login_flag = int(row.get("contains_login", 0))
        warning_flag = int(row.get("contains_warning", 0))

        # -------------------------
        # ANOMALY SCORE (NEW)
        # -------------------------
        score = float(row.get("anomaly_score", 0.0))

        # -------------------------
        # IMPROVED SEVERITY LOGIC
        # -------------------------
        if score < -0.2:
            severity = "CRITICAL"
        elif score < -0.1:
            severity = "HIGH"
        elif error_flag == 1:
            severity = "HIGH"
        elif login_flag == 1:
            severity = "MEDIUM"
        elif warning_flag == 1:
            severity = "LOW"
        else:
            severity = "INFO"

        # -------------------------
        # CREATE DOCUMENT
        # -------------------------
        doc = {
            "timestamp": timestamp,
            "message": message,
            "msg_length": int(row.get("msg_length", 0)),
            "error_flag": error_flag,
            "login_flag": login_flag,
            "warning_flag": warning_flag,
            "digit_count": int(row.get("digit_count", 0)),
            "anomaly": int(row.get("anomaly", 0)),
            "anomaly_score": score,
            "severity": severity
        }

        # -------------------------
        # SEND TO ELASTICSEARCH
        # -------------------------
        es.index(
            index="ai-log-anomalies",
            id=doc_id,
            document=doc
        )

        count += 1

        # -------------------------
        # SEND ALERT
        # -------------------------
        send_alert(f"{severity} ALERT:\n{message}")

        log_info(f"Uploaded anomaly: {doc_id} | Severity: {severity} | Score: {score}")

    except Exception as e:
        log_error(f"Error processing row: {e}")
        skipped += 1

# -------------------------
# FINAL OUTPUT
# -------------------------
log_info(f"Uploaded {count} anomalies successfully")
log_info(f"Skipped rows: {skipped}")

print(f"Uploaded: {count}, Skipped: {skipped}")


import os
from dotenv import load_dotenv

load_dotenv()

ES_HOST = os.getenv("ES_HOST", "https://localhost:9200")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASS = os.getenv("ES_PASS", "")

ES_SOURCE_INDEX  = os.getenv("ES_SOURCE_INDEX",  ".ds-filebeat-*")
ES_ANOMALY_INDEX = os.getenv("ES_ANOMALY_INDEX", "ai-log-anomalies")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL   = os.getenv("GEMINI_MODEL",   "gemini-2.5-flash")

TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN",   "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SCHEDULE_INTERVAL_SECONDS = int(os.getenv("SCHEDULE_INTERVAL_SECONDS", "300"))

MODEL_PATH      = os.getenv("MODEL_PATH",      "model.pkl")
VECTORIZER_PATH = os.getenv("VECTORIZER_PATH", "vectorizer.pkl")
FEATURES_PATH   = os.getenv("FEATURES_PATH",   "features.npy")

ISOLATION_FOREST_CONTAMINATION = float(os.getenv("IF_CONTAMINATION", "0.02"))
ISOLATION_FOREST_N_ESTIMATORS  = int(os.getenv("IF_N_ESTIMATORS",    "200"))
TFIDF_MAX_FEATURES             = int(os.getenv("TFIDF_MAX_FEATURES",  "100"))

ALERT_COOLDOWN_SECONDS = int(os.getenv("ALERT_COOLDOWN_SECONDS", "60"))

FETCH_WINDOW_DEFAULT = os.getenv("FETCH_WINDOW_DEFAULT", "now-10m")
FETCH_SIZE           = int(os.getenv("FETCH_SIZE", "2000"))
LAST_RUN_FILE        = os.getenv("LAST_RUN_FILE", "last_run.txt")

LOGS_DATASET_CSV       = "logs_dataset.csv"
ANOMALY_RESULTS_CSV    = "anomaly_results.csv"
DETECTED_ANOMALIES_CSV = "detected_anomalies.csv"

KMEANS_N_CLUSTERS = int(os.getenv("KMEANS_N_CLUSTERS", "5"))

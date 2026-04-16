"""
predict.py — Run anomaly prediction on logs_dataset.csv.
Produces anomaly_results.csv with labels, scores, normalized scores, and severity.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib

from config import MODEL_PATH, ANOMALY_RESULTS_CSV, LOGS_DATASET_CSV
from feature_engineering import engineer_features
from logger import log_info, log_error


# ── Severity classification ────────────────────────────────────────────────────
def _classify_severity(row):
    score = row.get("anomaly_score_norm", 0.0)

    error_flag   = int(row.get("contains_error", 0))
    login_flag   = int(row.get("contains_login", 0))
    warning_flag = int(row.get("contains_warning", 0))

    # Strong anomalies
    if score > 0.85:
        return "CRITICAL"

    if score > 0.65:
        return "HIGH"

    # Inject diversity
    if score > 0.45:
        return "MEDIUM"

    if score > 0.25:
        return "LOW"

    return "INFO"
# ── Core prediction function ───────────────────────────────────────────────────

def predict_anomalies(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Predict anomalies on the given DataFrame (or loads LOGS_DATASET_CSV).

    Returns:
        DataFrame with added columns: anomaly, anomaly_score,
        anomaly_score_norm, severity.
    """
    # Feature engineering (inference mode — load existing vectorizer)
    X_final, df = engineer_features(df=df, fit_vectorizer=False)

    # Load model
    try:
        model = joblib.load(MODEL_PATH)
        log_info(f"Model loaded from {MODEL_PATH}")
    except Exception as exc:
        log_error(f"Cannot load model: {exc}")
        raise

    # Predict
    df["anomaly"]       = model.predict(X_final)          # +1 normal / -1 anomaly
    df["anomaly_score"] = model.decision_function(X_final) # raw score

    # Normalize score to [0, 1] — 1.0 = most anomalous
    s_min, s_max = df["anomaly_score"].min(), df["anomaly_score"].max()
    span = s_max - s_min if (s_max - s_min) > 0 else 1.0
    df["anomaly_score_norm"] = 1.0 - (df["anomaly_score"] - s_min) / span

    # Severity classification
    df["severity"] = df.apply(_classify_severity, axis=1)

    # Save
    df.to_csv(ANOMALY_RESULTS_CSV, index=False)
    log_info(f"Results saved → {ANOMALY_RESULTS_CSV}")

    counts = df["anomaly"].value_counts().to_dict()
    sev    = df["severity"].value_counts().to_dict()
    log_info(f"Anomaly distribution: {counts}")
    log_info(f"Severity distribution: {sev}")

    return df


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    result = predict_anomalies()
    anomalies = result[result["anomaly"] == -1]
    print(f"Total logs : {len(result)}")
    print(f"Anomalies  : {len(anomalies)}")
    print("\nSeverity breakdown:")
    print(result["severity"].value_counts())

import pandas as pd
import numpy as np
import joblib
from logger import log_info, log_error

from logger import log_info, log_error

try:
    # -------------------------
    # LOAD DATA
    # -------------------------
    df = pd.read_csv("logs_dataset.csv")
    log_info("Loaded logs_dataset.csv")

    df["message"] = df["message"].fillna("").astype(str)

    # -------------------------
    # BASIC FEATURES
    # -------------------------
    df["msg_length"] = df["message"].apply(len)

    df["contains_error"] = df["message"].str.contains(
        "error|fail|critical", case=False, regex=True
    ).astype(int)

    df["contains_login"] = df["message"].str.contains(
        "login|ssh|password", case=False, regex=True
    ).astype(int)

    df["contains_warning"] = df["message"].str.contains(
        "warning", case=False
    ).astype(int)

    df["digit_count"] = df["message"].str.count(r"\d")

    X_basic = df[[
        "msg_length",
        "contains_error",
        "contains_login",
        "contains_warning",
        "digit_count"
    ]].values

    # -------------------------
    # LOAD MODEL + VECTORIZER
    # -------------------------
    vectorizer = joblib.load("vectorizer.pkl")
    model = joblib.load("model.pkl")

    log_info("Model and vectorizer loaded")

    # -------------------------
    # NLP FEATURES
    # -------------------------
    X_text = vectorizer.transform(df["message"]).toarray()

    # -------------------------
    # COMBINE FEATURES
    # -------------------------
    X_final = np.hstack((X_basic, X_text))

    # -------------------------
    # PREDICT
    # -------------------------
    df["anomaly"] = model.predict(X_final)
    # -------------------------
    # ADD ANOMALY SCORE (NEW)
    # -------------------------
    df["anomaly_score"] = model.decision_function(X_final)

    # Normalize score (optional but powerful)
    df["anomaly_score_norm"] = (df["anomaly_score"] - df["anomaly_score"].min()) / (df["anomaly_score"].max() - df["anomaly_score"].min())

    log_info("Anomaly scores calculated")
    # -------------------------
    # SAVE RESULTS
    # -------------------------
    df.to_csv("anomaly_results.csv", index=False)
    log_info("Saved anomaly_results.csv")

    # -------------------------
    # LOG SUMMARY
    # -------------------------
    counts = df["anomaly"].value_counts().to_dict()
    log_info(f"Anomaly distribution: {counts}")

except Exception as e:
    log_error(f"Prediction failed: {e}")

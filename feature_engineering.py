"""
feature_engineering.py — Advanced feature engineering pipeline.
Produces a combined numpy feature matrix from logs_dataset.csv.
Features: basic text stats + time-based + message frequency + TF-IDF NLP.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    LOGS_DATASET_CSV,
    VECTORIZER_PATH,
    FEATURES_PATH,
    TFIDF_MAX_FEATURES,
)
from logger import log_info, log_error


# ── Public API ─────────────────────────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame | None = None,
    fit_vectorizer: bool = True,
) -> tuple[np.ndarray, pd.DataFrame]:
    """
    Build the full feature matrix.

    Args:
        df:              Pre-loaded DataFrame. If None, loads LOGS_DATASET_CSV.
        fit_vectorizer:  True → fit & save vectorizer (training).
                         False → load existing vectorizer (inference).

    Returns:
        (X_final, df) — feature matrix and the enriched DataFrame.
    """
    df = _load(df)
    df = _clean(df)
    df = _basic_features(df)
    df = _time_features(df)
    df = _frequency_features(df)
    X_basic = _select_basic(df)
    X_text  = _tfidf_features(df, fit=fit_vectorizer)
    X_final = np.hstack((X_basic, X_text))

    np.save(FEATURES_PATH, X_final)
    log_info(f"Feature matrix saved → {FEATURES_PATH} | shape={X_final.shape}")

    return X_final, df


# ── Private helpers ────────────────────────────────────────────────────────────

def _load(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df.copy()
    try:
        df = pd.read_csv(LOGS_DATASET_CSV)
        log_info(f"Loaded {LOGS_DATASET_CSV} ({len(df)} rows)")
        return df
    except Exception as exc:
        log_error(f"Cannot load {LOGS_DATASET_CSV}: {exc}")
        raise


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df["message"] = df.get("message", pd.Series(dtype=str)).fillna("").astype(str)
    return df


def _basic_features(df: pd.DataFrame) -> pd.DataFrame:
    msg = df["message"]
    df["msg_length"]       = msg.str.len()
    df["contains_error"]   = msg.str.contains(r"error|fail|critical|exception", case=False, regex=True).astype(int)
    df["contains_login"]   = msg.str.contains(r"login|ssh|password|auth",       case=False, regex=True).astype(int)
    df["contains_warning"] = msg.str.contains(r"warning|warn",                  case=False, regex=True).astype(int)
    df["digit_count"]      = msg.str.count(r"\d")
    df["word_count"]       = msg.str.split().str.len().fillna(0).astype(int)
    df["uppercase_ratio"]  = msg.apply(
        lambda s: sum(1 for c in s if c.isupper()) / max(len(s), 1)
    )
    return df


def _time_features(df: pd.DataFrame) -> pd.DataFrame:
    ts_col = "@timestamp" if "@timestamp" in df.columns else "timestamp"
    if ts_col not in df.columns:
        log_error("No timestamp column found — time features set to 0")
        df["hour"]        = 0
        df["day_of_week"] = 0
        return df

    try:
        ts = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
        df["hour"]        = ts.dt.hour.fillna(0).astype(int)
        df["day_of_week"] = ts.dt.dayofweek.fillna(0).astype(int)
    except Exception as exc:
        log_error(f"Time feature extraction failed: {exc}")
        df["hour"]        = 0
        df["day_of_week"] = 0

    return df


def _frequency_features(df: pd.DataFrame) -> pd.DataFrame:
    freq = df["message"].value_counts()
    df["message_frequency"] = df["message"].map(freq).fillna(1).astype(int)
    return df


_BASIC_COLS = [
    "msg_length",
    "contains_error",
    "contains_login",
    "contains_warning",
    "digit_count",
    "word_count",
    "uppercase_ratio",
    "hour",
    "day_of_week",
    "message_frequency",
]


def _select_basic(df: pd.DataFrame) -> np.ndarray:
    missing = [c for c in _BASIC_COLS if c not in df.columns]
    if missing:
        log_error(f"Missing basic feature columns: {missing}")
        raise ValueError(f"Missing columns: {missing}")
    return df[_BASIC_COLS].fillna(0).values.astype(float)


def _tfidf_features(df: pd.DataFrame, fit: bool) -> np.ndarray:
    if fit:
        vectorizer = TfidfVectorizer(
            max_features=TFIDF_MAX_FEATURES,
            sublinear_tf=True,
            min_df=2,
            strip_accents="unicode",
        )
        X_text = vectorizer.fit_transform(df["message"])
        joblib.dump(vectorizer, VECTORIZER_PATH)
        log_info(f"Vectorizer fitted & saved → {VECTORIZER_PATH}")
    else:
        try:
            vectorizer = joblib.load(VECTORIZER_PATH)
            log_info(f"Vectorizer loaded from {VECTORIZER_PATH}")
        except Exception as exc:
            log_error(f"Could not load vectorizer: {exc}")
            raise
        X_text = vectorizer.transform(df["message"])

    return X_text.toarray()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    X, df = engineer_features(fit_vectorizer=True)
    print(f"Feature matrix shape: {X.shape}")
    print(df[_BASIC_COLS].head())

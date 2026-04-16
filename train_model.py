"""
train_model.py — Train (or retrain) the Isolation Forest anomaly detector.
Loads feature matrix produced by feature_engineering.py.
"""

import numpy as np
import joblib
from sklearn.ensemble import IsolationForest

from config import (
    FEATURES_PATH,
    MODEL_PATH,
    ISOLATION_FOREST_CONTAMINATION,
    ISOLATION_FOREST_N_ESTIMATORS,
)
from logger import log_info, log_error


def train_model(X: np.ndarray | None = None) -> IsolationForest:
    """
    Fit an IsolationForest and persist it.

    Args:
        X: Feature matrix. If None, loads from FEATURES_PATH.

    Returns:
        Fitted IsolationForest model.
    """
    if X is None:
        try:
            X = np.load(FEATURES_PATH)
            log_info(f"Loaded features from {FEATURES_PATH} | shape={X.shape}")
        except Exception as exc:
            log_error(f"Cannot load features: {exc}")
            raise

    if X.ndim != 2 or X.shape[0] == 0:
        raise ValueError(f"Invalid feature matrix shape: {X.shape}")

    log_info(
        f"Training IsolationForest | n_estimators={ISOLATION_FOREST_N_ESTIMATORS}"
        f" | contamination={ISOLATION_FOREST_CONTAMINATION}"
    )

    model = IsolationForest(
        n_estimators=ISOLATION_FOREST_N_ESTIMATORS,
        contamination=ISOLATION_FOREST_CONTAMINATION,
        max_samples="auto",
        max_features=1.0,
        bootstrap=False,
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X)

    joblib.dump(model, MODEL_PATH)
    log_info(f"Model saved → {MODEL_PATH}")

    return model


if __name__ == "__main__":
    model = train_model()
    print(f"Model trained successfully. Saved to {MODEL_PATH}")

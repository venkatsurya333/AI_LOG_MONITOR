"""
clustering.py — Root cause detection via KMeans clustering on TF-IDF features.
Groups anomalies into thematic clusters and summarises dominant error patterns.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

from config import (
    ANOMALY_RESULTS_CSV,
    VECTORIZER_PATH,
    KMEANS_N_CLUSTERS,
)
from logger import log_info, log_error, log_warning


def cluster_anomalies(df: pd.DataFrame | None = None) -> list[dict]:
    """
    Cluster anomaly messages using KMeans on TF-IDF vectors.

    Args:
        df: DataFrame with anomaly_results. Loads ANOMALY_RESULTS_CSV if None.

    Returns:
        List of cluster dicts:
            {
              cluster_id:   int,
              size:         int,
              top_keywords: list[str],
              sample_msgs:  list[str],
              dominant_severity: str,
              summary:      str,
            }
    """
    df = _load(df)
    anomalies = df[df["anomaly"] == -1].copy() if "anomaly" in df.columns else df.copy()

    if anomalies.empty:
        log_info("No anomalies to cluster")
        return []

    anomalies["message"] = anomalies["message"].fillna("").astype(str)

    n_clusters = min(KMEANS_N_CLUSTERS, len(anomalies))
    if n_clusters < 2:
        log_warning("Too few anomalies for clustering — skipping")
        return []

    # Vectorize messages (use fitted vectorizer if available, else fit fresh)
    try:
        vectorizer = joblib.load(VECTORIZER_PATH)
        log_info(f"Vectorizer loaded from {VECTORIZER_PATH}")
    except Exception:
        log_warning("Vectorizer not found — fitting a temporary one for clustering")
        vectorizer = TfidfVectorizer(max_features=100, sublinear_tf=True)
        vectorizer.fit(anomalies["message"])

    X = vectorizer.transform(anomalies["message"]).toarray()

    # KMeans
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    anomalies["cluster"] = km.fit_predict(X)
    log_info(f"KMeans clustering done | k={n_clusters}")

    # Build summaries
    feature_names = vectorizer.get_feature_names_out()
    summaries = []

    for cid in range(n_clusters):
        subset = anomalies[anomalies["cluster"] == cid]
        size   = len(subset)

        # Top keywords from cluster centroid
        centroid = km.cluster_centers_[cid]
        top_idx  = centroid.argsort()[-8:][::-1]
        keywords = [feature_names[i] for i in top_idx]

        # Sample messages
        samples = subset["message"].head(3).tolist()

        # Dominant severity
        dom_sev = (
            subset["severity"].mode()[0]
            if "severity" in subset.columns and not subset.empty
            else "UNKNOWN"
        )

        summary = (
            f"Cluster {cid} ({size} anomalies) — "
            f"keywords: [{', '.join(keywords[:5])}] — "
            f"dominant severity: {dom_sev}"
        )

        summaries.append({
            "cluster_id":         cid,
            "size":               size,
            "top_keywords":       keywords,
            "sample_msgs":        samples,
            "dominant_severity":  dom_sev,
            "summary":            summary,
        })

        log_info(summary)

    return summaries


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is not None:
        return df.copy()
    try:
        df = pd.read_csv(ANOMALY_RESULTS_CSV)
        log_info(f"Loaded {ANOMALY_RESULTS_CSV} ({len(df)} rows)")
        return df
    except Exception as exc:
        log_error(f"Cannot load {ANOMALY_RESULTS_CSV}: {exc}")
        return pd.DataFrame()


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    clusters = cluster_anomalies()
    if not clusters:
        print("No clusters generated.")
    else:
        for c in clusters:
            print(f"\n[Cluster {c['cluster_id']}] size={c['size']} | sev={c['dominant_severity']}")
            print(f"  Keywords : {', '.join(c['top_keywords'])}")
            print(f"  Sample   : {c['sample_msgs'][0] if c['sample_msgs'] else 'N/A'}")

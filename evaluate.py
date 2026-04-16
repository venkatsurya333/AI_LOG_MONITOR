"""
evaluate.py — Model evaluation and diagnostics.
Computes anomaly rate, score distribution stats, severity breakdown,
confusion matrix (using error keywords as weak labels), and
produces a clean CLI report.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from config import ANOMALY_RESULTS_CSV, MODEL_PATH, VECTORIZER_PATH
from logger import log_info, log_error


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_results(path: str = ANOMALY_RESULTS_CSV) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        log_info(f"Loaded {path} ({len(df)} rows)")
        return df
    except Exception as exc:
        log_error(f"Cannot load {path}: {exc}")
        raise


def _basic_stats(df: pd.DataFrame) -> dict:
    total   = len(df)
    anom    = (df["anomaly"] == -1).sum()
    normal  = total - anom
    rate    = anom / total if total else 0

    score_col = "anomaly_score"
    norm_col  = "anomaly_score_norm"

    stats = {
        "total":        total,
        "anomalies":    int(anom),
        "normal":       int(normal),
        "anom_rate":    rate,
    }

    if score_col in df.columns:
        a = df[df["anomaly"] == -1][score_col]
        n = df[df["anomaly"] ==  1][score_col]
        stats["score_anom_mean"] = float(a.mean()) if len(a) else 0
        stats["score_anom_std"]  = float(a.std())  if len(a) else 0
        stats["score_norm_mean"] = float(n.mean()) if len(n) else 0
        stats["score_norm_std"]  = float(n.std())  if len(n) else 0
        stats["score_min"]       = float(df[score_col].min())
        stats["score_max"]       = float(df[score_col].max())
        stats["score_threshold"] = float(df[df["anomaly"]==-1][score_col].max()) \
                                   if len(a) else 0

    if norm_col in df.columns:
        a_norm = df[df["anomaly"] == -1][norm_col]
        stats["norm_score_mean"] = float(a_norm.mean()) if len(a_norm) else 0
        stats["norm_score_p95"]  = float(a_norm.quantile(0.95)) if len(a_norm) else 0

    return stats


def _severity_breakdown(df: pd.DataFrame) -> dict[str, int]:
    if "severity" not in df.columns:
        return {}
    return df[df["anomaly"] == -1]["severity"].value_counts().to_dict()


def _weak_label_confusion(df: pd.DataFrame) -> dict:
    """
    Use keyword flags as weak positive labels to estimate precision/recall.
    A log with contains_error=1 is treated as a 'true positive' candidate.
    """
    if "contains_error" not in df.columns:
        return {}

    df = df.copy()
    df["weak_label"] = (
        (df.get("contains_error", 0) == 1) |
        (df.get("contains_login",  0) == 1) |
        (df.get("contains_warning",0) == 1)
    ).astype(int)

    df["predicted_anom"] = (df["anomaly"] == -1).astype(int)

    tp = ((df["predicted_anom"] == 1) & (df["weak_label"] == 1)).sum()
    fp = ((df["predicted_anom"] == 1) & (df["weak_label"] == 0)).sum()
    fn = ((df["predicted_anom"] == 0) & (df["weak_label"] == 1)).sum()
    tn = ((df["predicted_anom"] == 0) & (df["weak_label"] == 0)).sum()

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall    = tp / (tp + fn) if (tp + fn) else 0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    return {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "weak_precision": precision,
        "weak_recall":    recall,
        "weak_f1":        f1,
    }


def _feature_importance(df: pd.DataFrame) -> dict:
    """
    Proxy for feature relevance: mean absolute value of each basic feature
    among anomalies vs normals. Higher difference = more discriminative.
    """
    basic = ["msg_length","contains_error","contains_login","contains_warning",
             "digit_count","word_count","uppercase_ratio","hour","day_of_week",
             "message_frequency"]
    present = [c for c in basic if c in df.columns]

    if not present:
        return {}

    anom   = df[df["anomaly"] == -1][present].mean()
    normal = df[df["anomaly"] ==  1][present].mean()
    diff   = (anom - normal).abs().sort_values(ascending=False)
    return diff.to_dict()


def _histogram_bar(values: pd.Series, n_bins: int = 10, width: int = 32) -> str:
    """ASCII histogram of a numeric series."""
    if values.empty:
        return "  (no data)"
    mn, mx = values.min(), values.max()
    if mn == mx:
        return f"  all values = {mn:.4f}"
    bins = np.linspace(mn, mx, n_bins + 1)
    counts, edges = np.histogram(values, bins=bins)
    mx_count = max(counts) or 1
    lines = []
    for i, c in enumerate(counts):
        bar   = "█" * int(c / mx_count * width)
        label = f"{edges[i]:+.3f}"
        lines.append(f"  {label:>8}  {bar} {c}")
    return "\n".join(lines)


# ── Main report ────────────────────────────────────────────────────────────────

def evaluate(path: str = ANOMALY_RESULTS_CSV) -> dict:
    df = _load_results(path)

    if "anomaly" not in df.columns:
        raise ValueError("Column 'anomaly' missing — run predict.py first")

    stats   = _basic_stats(df)
    sev     = _severity_breakdown(df)
    conf    = _weak_label_confusion(df)
    feat    = _feature_importance(df)
    scores  = df[df["anomaly"]==-1]["anomaly_score"] if "anomaly_score" in df.columns else pd.Series()

    # ── Print report ──────────────────────────────────────────────────────────
    W = 60
    hr = "─" * W

    print(f"\n{'═'*W}")
    print(f"  AI Log Monitor — Model Evaluation Report")
    print(f"{'═'*W}\n")

    print(f"  {hr}")
    print(f"  DATASET SUMMARY")
    print(f"  {hr}")
    print(f"  Total logs       : {stats['total']:>8,}")
    print(f"  Normal           : {stats['normal']:>8,}  ({100-stats['anom_rate']*100:.2f}%)")
    print(f"  Anomalies        : {stats['anomalies']:>8,}  ({stats['anom_rate']*100:.2f}%)")

    print(f"\n  {hr}")
    print(f"  SCORE DISTRIBUTION  (raw decision_function)")
    print(f"  {hr}")
    if "score_min" in stats:
        print(f"  Score range      : [{stats['score_min']:+.4f}, {stats['score_max']:+.4f}]")
        print(f"  Anomaly mean     : {stats['score_anom_mean']:+.4f}  σ={stats['score_anom_std']:.4f}")
        print(f"  Normal mean      : {stats['score_norm_mean']:+.4f}  σ={stats['score_norm_std']:.4f}")
        print(f"  Decision boundary: {stats['score_threshold']:+.4f}")
    if "norm_score_mean" in stats:
        print(f"  Norm score mean  : {stats['norm_score_mean']:.4f}")
        print(f"  Norm score p95   : {stats['norm_score_p95']:.4f}")

    if not scores.empty:
        print(f"\n  Score histogram (anomalies only):")
        print(_histogram_bar(scores))

    print(f"\n  {hr}")
    print(f"  SEVERITY BREAKDOWN")
    print(f"  {hr}")
    for level in ("CRITICAL","HIGH","MEDIUM","LOW","INFO"):
        cnt = sev.get(level, 0)
        if cnt or level in ("CRITICAL","HIGH"):
            bar = "█" * min(cnt, 40)
            print(f"  {level:<10} {cnt:>6}  {bar}")

    if conf:
        print(f"\n  {hr}")
        print(f"  WEAK-LABEL QUALITY ESTIMATES")
        print(f"  (using error/login/warning keywords as proxy labels)")
        print(f"  {hr}")
        print(f"  True Positives   : {conf['tp']:>6}")
        print(f"  False Positives  : {conf['fp']:>6}  (flagged but no keyword)")
        print(f"  False Negatives  : {conf['fn']:>6}  (keyword present but not flagged)")
        print(f"  True Negatives   : {conf['tn']:>6}")
        print(f"  Weak Precision   : {conf['weak_precision']:.4f}")
        print(f"  Weak Recall      : {conf['weak_recall']:.4f}")
        print(f"  Weak F1          : {conf['weak_f1']:.4f}")

    if feat:
        print(f"\n  {hr}")
        print(f"  FEATURE DISCRIMINABILITY")
        print(f"  (mean |anomaly - normal| per feature, descending)")
        print(f"  {hr}")
        for fname, diff in list(feat.items())[:8]:
            bar = "█" * int(min(diff / max(feat.values(), default=1), 1) * 24)
            print(f"  {fname:<24} {diff:.4f}  {bar}")

    print(f"\n{'═'*W}\n")

    result = {"stats": stats, "severity": sev, "confusion": conf, "feature_diff": feat}
    log_info(f"Evaluation complete | anom_rate={stats['anom_rate']:.4f}")
    return result


if __name__ == "__main__":
    evaluate()

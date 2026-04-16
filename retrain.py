"""
retrain.py — Versioned model retraining with backup and automatic rollback.

Workflow:
  1. Back up current model + vectorizer with a timestamp suffix.
  2. Fetch fresh logs.
  3. Re-run feature engineering (fits new vectorizer).
  4. Train new Isolation Forest.
  5. Run predict on the new data and validate the anomaly rate is sane.
  6. If validation fails, restore the backed-up model automatically.
  7. Prune old backups, keeping only the last N versions.
"""

import os
import shutil
import time
import numpy as np
import pandas as pd

from config import (
    MODEL_PATH, VECTORIZER_PATH, FEATURES_PATH,
    ISOLATION_FOREST_CONTAMINATION,
    LOGS_DATASET_CSV, ANOMALY_RESULTS_CSV,
)
from logger import log_info, log_error, log_warning

# ── Config ─────────────────────────────────────────────────────────────────────
BACKUP_DIR     = "model_backups"
MAX_BACKUPS    = 5          # keep only the last N backup sets
MIN_ANOM_RATE  = 0.005      # sanity floor  (0.5%)
MAX_ANOM_RATE  = 0.15       # sanity ceiling (15%)


# ── Backup helpers ─────────────────────────────────────────────────────────────

def _backup_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _backup_current() -> str | None:
    """Copy current model + vectorizer to BACKUP_DIR/<tag>. Returns tag."""
    if not (os.path.isfile(MODEL_PATH) and os.path.isfile(VECTORIZER_PATH)):
        log_warning("No existing model/vectorizer to back up — skipping backup")
        return None

    os.makedirs(BACKUP_DIR, exist_ok=True)
    tag = _backup_tag()
    dst = os.path.join(BACKUP_DIR, tag)
    os.makedirs(dst, exist_ok=True)

    for src in [MODEL_PATH, VECTORIZER_PATH]:
        shutil.copy2(src, os.path.join(dst, os.path.basename(src)))

    log_info(f"Model backed up → {dst}")
    return tag


def _restore_backup(tag: str) -> bool:
    """Restore model + vectorizer from a backup tag."""
    src_dir = os.path.join(BACKUP_DIR, tag)
    if not os.path.isdir(src_dir):
        log_error(f"Backup {tag} not found — cannot restore")
        return False

    for fname in [os.path.basename(MODEL_PATH), os.path.basename(VECTORIZER_PATH)]:
        src = os.path.join(src_dir, fname)
        dst = MODEL_PATH if fname == os.path.basename(MODEL_PATH) else VECTORIZER_PATH
        if os.path.isfile(src):
            shutil.copy2(src, dst)

    log_info(f"Model restored from backup {tag}")
    return True


def _prune_backups():
    """Keep only the last MAX_BACKUPS backup sets."""
    if not os.path.isdir(BACKUP_DIR):
        return
    entries = sorted(os.listdir(BACKUP_DIR))
    to_delete = entries[:-MAX_BACKUPS] if len(entries) > MAX_BACKUPS else []
    for name in to_delete:
        path = os.path.join(BACKUP_DIR, name)
        if os.path.isdir(path):
            shutil.rmtree(path)
            log_info(f"Pruned old backup: {name}")


# ── Validation ─────────────────────────────────────────────────────────────────

def _validate_model() -> tuple[bool, str]:
    """
    Run predict on the freshly trained model and check the anomaly rate.
    Returns (ok, reason_string).
    """
    try:
        from predict import predict_anomalies
        df = predict_anomalies()

        total = len(df)
        if total == 0:
            return False, "Prediction returned 0 rows"

        anom_count = (df["anomaly"] == -1).sum()
        rate = anom_count / total

        if rate < MIN_ANOM_RATE:
            return False, (
                f"Anomaly rate too low: {rate:.4f} < {MIN_ANOM_RATE} "
                "(model may be undertrained or contamination too low)"
            )
        if rate > MAX_ANOM_RATE:
            return False, (
                f"Anomaly rate too high: {rate:.4f} > {MAX_ANOM_RATE} "
                "(contamination parameter may need tuning)"
            )

        log_info(f"Validation passed | anomaly_rate={rate:.4f} | total={total}")
        return True, f"anomaly_rate={rate:.4f} ({anom_count}/{total})"

    except Exception as exc:
        return False, str(exc)


# ── Main retrain pipeline ──────────────────────────────────────────────────────

def retrain(skip_fetch: bool = False) -> bool:
    """
    Full retrain pipeline.

    Args:
        skip_fetch: If True, reuse existing logs_dataset.csv (faster for testing).

    Returns:
        True on success, False on failure (model rolled back automatically).
    """
    log_info("═" * 50)
    log_info("Retrain pipeline started")
    t0 = time.time()

    # 1. Backup
    tag = _backup_current()

    try:
        # 2. Fetch fresh data
        if not skip_fetch:
            from fetch_logs import fetch_logs
            df_raw = fetch_logs()
            if df_raw.empty:
                log_warning("No new logs fetched — aborting retrain")
                return False
        else:
            log_info("Skipping fetch (using existing logs_dataset.csv)")

        # 3. Feature engineering (fit=True → new vectorizer)
        from feature_engineering import engineer_features
        X, _ = engineer_features(fit_vectorizer=True)
        log_info(f"Features shape: {X.shape}")

        # 4. Train new model
        from train_model import train_model
        train_model(X=X)

        # 5. Validate
        ok, reason = _validate_model()
        if not ok:
            log_error(f"Validation failed: {reason}")
            if tag:
                log_info("Rolling back to previous model...")
                _restore_backup(tag)
                log_info("Rollback complete")
            return False

        log_info(f"Retrain complete | {reason} | elapsed={time.time()-t0:.1f}s")

    except Exception as exc:
        log_error(f"Retrain error: {exc}")
        if tag:
            log_info("Restoring backup due to error...")
            _restore_backup(tag)
        return False

    # 6. Prune old backups
    _prune_backups()

    elapsed = time.time() - t0
    print(f"\nRetrain complete in {elapsed:.1f}s")
    print(f"Backups retained: {MAX_BACKUPS} max → {BACKUP_DIR}/")
    return True


def list_backups():
    """Print available model backups."""
    if not os.path.isdir(BACKUP_DIR):
        print("No backups found.")
        return
    entries = sorted(os.listdir(BACKUP_DIR), reverse=True)
    print(f"\nAvailable backups ({len(entries)}):")
    for i, name in enumerate(entries):
        path = os.path.join(BACKUP_DIR, name)
        files = os.listdir(path) if os.path.isdir(path) else []
        size  = sum(os.path.getsize(os.path.join(path, f)) for f in files) // 1024
        print(f"  {'[latest]' if i==0 else '        '} {name}  ({size} KB)")
    print()


def rollback(tag: str | None = None):
    """
    Roll back to a specific backup tag, or to the most recent one.

    Args:
        tag: backup tag string (YYYYMMDD_HHMMSS). None → most recent.
    """
    if not os.path.isdir(BACKUP_DIR):
        log_error("No backup directory found")
        return False

    entries = sorted(os.listdir(BACKUP_DIR), reverse=True)
    if not entries:
        log_error("No backups available")
        return False

    chosen = tag if tag else entries[0]
    ok = _restore_backup(chosen)
    if ok:
        print(f"Rolled back to: {chosen}")
    return ok


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Log Monitor — Model Retraining")
    sub = parser.add_subparsers(dest="cmd")

    train_cmd = sub.add_parser("train", help="Retrain model (default)")
    train_cmd.add_argument("--skip-fetch", action="store_true",
                           help="Reuse existing logs_dataset.csv")

    sub.add_parser("list", help="List available backups")

    roll_cmd = sub.add_parser("rollback", help="Roll back to a previous model")
    roll_cmd.add_argument("--tag", default=None,
                          help="Backup tag (YYYYMMDD_HHMMSS). Omit for latest.")

    args = parser.parse_args()

    if args.cmd == "list":
        list_backups()
    elif args.cmd == "rollback":
        ok = rollback(args.tag)
        sys.exit(0 if ok else 1)
    else:
        ok = retrain(skip_fetch=getattr(args, "skip_fetch", False))
        sys.exit(0 if ok else 1)

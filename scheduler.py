"""
scheduler.py — Loop-based pipeline scheduler.
Runs the full pipeline (fetch → feature engineering → predict → upload)
at a configurable interval. No cron dependency. Low-memory optimised.
"""

import time
import gc
import traceback

from config import SCHEDULE_INTERVAL_SECONDS
from logger import log_info, log_error, log_warning


def _run_pipeline() -> bool:
    """
    Execute one full pipeline cycle.
    Returns True on success, False on failure.
    """
    log_info("═" * 50)
    log_info("Pipeline cycle started")

    try:
        # Step 1 — Fetch logs
        from fetch_logs import fetch_logs
        df = fetch_logs()

        if df is None or df.empty:
            log_info("No new logs — skipping pipeline cycle")
            return True

        # Step 2 — Feature engineering (inference mode: load existing vectorizer)
        from feature_engineering import engineer_features
        X, df_feat = engineer_features(df=df, fit_vectorizer=False)

        # Step 3 — Predict anomalies
        from predict import predict_anomalies
        df_result = predict_anomalies(df=df_feat)

        # Step 4 — Upload to Elasticsearch
        from upload import upload_anomalies
        uploaded, skipped = upload_anomalies(df=df_result)

        log_info(f"Pipeline cycle complete | uploaded={uploaded} skipped={skipped}")
        return True

    except FileNotFoundError as exc:
        log_warning(
            f"Model or vectorizer not found — run train_model.py first. Detail: {exc}"
        )
        return False

    except Exception as exc:
        log_error(f"Pipeline error: {exc}")
        log_error(traceback.format_exc())
        return False

    finally:
        # Free memory on low-resource systems
        gc.collect()


def run_scheduler(interval: int = SCHEDULE_INTERVAL_SECONDS) -> None:
    """
    Infinite loop: run pipeline, sleep, repeat.

    Args:
        interval: seconds between pipeline runs.
    """
    log_info(f"Scheduler started | interval={interval}s")
    print(f"\n⏱  Scheduler running — pipeline every {interval}s  (Ctrl+C to stop)\n")

    consecutive_failures = 0

    while True:
        success = _run_pipeline()

        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= 5:
                log_error(
                    f"5 consecutive pipeline failures — check logs. "
                    "Continuing with backoff."
                )
                # Back off to avoid hammering a broken system
                time.sleep(interval * 3)
                consecutive_failures = 0
                continue

        log_info(f"Next run in {interval}s")
        time.sleep(interval)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AI Log Monitor Scheduler")
    parser.add_argument(
        "--interval",
        type=int,
        default=SCHEDULE_INTERVAL_SECONDS,
        help="Pipeline interval in seconds (default: from .env)",
    )
    args = parser.parse_args()
    run_scheduler(args.interval)

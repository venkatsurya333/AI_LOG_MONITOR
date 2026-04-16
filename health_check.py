"""
health_check.py — Pre-flight system health check.
Verifies: Elasticsearch connectivity, index existence, model artifacts,
disk space, Python package availability, and Telegram reachability.
Prints a clean report and exits non-zero if any critical check fails.
"""

import os
import sys
import importlib
import shutil
import time
import requests

from config import (
    ES_HOST, ES_USER, ES_PASS,
    ES_SOURCE_INDEX, ES_ANOMALY_INDEX,
    MODEL_PATH, VECTORIZER_PATH, FEATURES_PATH,
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID,
    GEMINI_API_KEY,
    LOGS_DATASET_CSV, ANOMALY_RESULTS_CSV,
)
from logger import log_info, log_error


# ── Result store ───────────────────────────────────────────────────────────────

class CheckResult:
    def __init__(self):
        self.checks: list[dict] = []

    def add(self, name: str, ok: bool, detail: str = "", critical: bool = True):
        self.checks.append({"name": name, "ok": ok, "detail": detail, "critical": critical})

    @property
    def all_ok(self) -> bool:
        return all(c["ok"] for c in self.checks if c["critical"])

    def print_report(self):
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║         AI Log Monitor — Health Check                ║")
        print("╚══════════════════════════════════════════════════════╝\n")

        ok_sym  = "\033[92m✔\033[0m"
        err_sym = "\033[91m✘\033[0m"
        warn_sym= "\033[93m!\033[0m"

        for c in self.checks:
            sym    = ok_sym if c["ok"] else (err_sym if c["critical"] else warn_sym)
            status = "OK   " if c["ok"] else ("FAIL " if c["critical"] else "WARN ")
            detail = f"  ← {c['detail']}" if c["detail"] else ""
            print(f"  {sym}  [{status}] {c['name']}{detail}")

        print()
        if self.all_ok:
            print("  \033[92mAll critical checks passed. System is ready.\033[0m")
        else:
            failed = [c for c in self.checks if not c["ok"] and c["critical"]]
            print(f"  \033[91m{len(failed)} critical check(s) failed. Fix before running pipeline.\033[0m")
        print()


# ── Individual checks ──────────────────────────────────────────────────────────

def _check_elasticsearch(res: CheckResult):
    try:
        from elasticsearch import Elasticsearch
        es = Elasticsearch(
            ES_HOST,
            basic_auth=(ES_USER, ES_PASS),
            verify_certs=False,
            request_timeout=10,
        )
        if not es.ping():
            res.add("Elasticsearch connection", False, "ping failed", critical=True)
            return
        res.add("Elasticsearch connection", True, ES_HOST)

        # Check source index
        src_exists = es.indices.exists_alias(name=ES_SOURCE_INDEX) or \
                     bool(es.indices.get(index=ES_SOURCE_INDEX, ignore_unavailable=True))
        res.add("Source index reachable", src_exists,
                f"{ES_SOURCE_INDEX}", critical=True)

        # Check anomaly index (non-critical on first run)
        try:
            anom_count = es.count(index=ES_ANOMALY_INDEX)["count"]
            res.add("Anomaly index", True,
                    f"{ES_ANOMALY_INDEX} | {anom_count} docs", critical=False)
        except Exception:
            res.add("Anomaly index", False,
                    f"{ES_ANOMALY_INDEX} not yet created (ok on first run)", critical=False)

    except Exception as exc:
        res.add("Elasticsearch connection", False, str(exc), critical=True)


def _check_model_artifacts(res: CheckResult):
    for path, label, critical in [
        (MODEL_PATH,      "Isolation Forest model",  True),
        (VECTORIZER_PATH, "TF-IDF vectorizer",        True),
        (FEATURES_PATH,   "Feature matrix (.npy)",    False),
    ]:
        exists = os.path.isfile(path)
        size   = f"{os.path.getsize(path)/1024:.1f} KB" if exists else "missing"
        res.add(label, exists, f"{path} ({size})", critical=critical)


def _check_data_files(res: CheckResult):
    for path, label in [
        (LOGS_DATASET_CSV,    "Logs dataset CSV"),
        (ANOMALY_RESULTS_CSV, "Anomaly results CSV"),
    ]:
        exists = os.path.isfile(path)
        detail = f"{os.path.getsize(path)//1024} KB" if exists else "not yet generated"
        res.add(label, exists, f"{path} ({detail})", critical=False)


def _check_disk_space(res: CheckResult):
    usage = shutil.disk_usage(".")
    free_mb = usage.free / (1024 ** 2)
    pct_used = usage.used / usage.total * 100
    ok = free_mb > 200
    res.add(
        "Disk space",
        ok,
        f"{free_mb:.0f} MB free ({pct_used:.1f}% used)",
        critical=False,
    )


def _check_python_packages(res: CheckResult):
    required = [
        "elasticsearch", "pandas", "numpy", "sklearn",
        "joblib", "dotenv", "requests",
    ]
    for pkg in required:
        name = "sklearn" if pkg == "sklearn" else pkg
        try:
            importlib.import_module(name)
            res.add(f"Package: {pkg}", True, critical=False)
        except ImportError:
            res.add(f"Package: {pkg}", False,
                    "run: pip install -r requirements.txt", critical=True)


def _check_gemini(res: CheckResult):
    if not GEMINI_API_KEY:
        res.add("Gemini API key", False, "GEMINI_API_KEY not set in .env", critical=False)
        return
    res.add("Gemini API key", True, f"{'*'*8}{GEMINI_API_KEY[-4:]}", critical=False)


def _check_telegram(res: CheckResult):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        res.add("Telegram credentials", False,
                "TELEGRAM_TOKEN or CHAT_ID missing", critical=False)
        return
    try:
        url  = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/getMe"
        resp = requests.get(url, timeout=8)
        ok   = resp.status_code == 200 and resp.json().get("ok", False)
        detail = resp.json().get("result", {}).get("username", "?") if ok else resp.text[:60]
        res.add("Telegram bot reachable", ok, f"@{detail}", critical=False)
    except Exception as exc:
        res.add("Telegram bot reachable", False, str(exc), critical=False)


def _check_env_file(res: CheckResult):
    exists = os.path.isfile(".env")
    res.add(".env file present", exists,
            "copy .env.example → .env and fill credentials" if not exists else "",
            critical=True)


# ── Main ───────────────────────────────────────────────────────────────────────

def run_health_check() -> bool:
    res = CheckResult()
    t0  = time.time()

    _check_env_file(res)
    _check_python_packages(res)
    _check_elasticsearch(res)
    _check_model_artifacts(res)
    _check_data_files(res)
    _check_disk_space(res)
    _check_gemini(res)
    _check_telegram(res)

    elapsed = time.time() - t0
    res.print_report()
    print(f"  Completed in {elapsed:.2f}s\n")

    if res.all_ok:
        log_info("Health check passed")
    else:
        log_error("Health check failed — see output above")

    return res.all_ok


if __name__ == "__main__":
    ok = run_health_check()
    sys.exit(0 if ok else 1)

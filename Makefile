# Makefile — AI Log Monitor
# Usage: make <target>
# Requires: python3, pip, bash

PYTHON := python3
VENV   := .venv
PIP    := pip

.PHONY: help setup install health fetch features train predict upload \
        run assistant retrain retrain-skip evaluate report \
        trend clusters clean clean-data clean-model clean-all \
        backup-list rollback scheduler

# ── Default ────────────────────────────────────────────────────────────────────

help:
	@echo ""
	@echo "╔══════════════════════════════════════════════════════╗"
	@echo "║       AI Log Monitor — Makefile Commands             ║"
	@echo "╠══════════════════════════════════════════════════════╣"
	@echo "║  SETUP                                               ║"
	@echo "║    make setup        First-time setup (bash script)  ║"
	@echo "║    make install      pip install requirements        ║"
	@echo "║    make health       Run health check                ║"
	@echo "║                                                      ║"
	@echo "║  PIPELINE (manual steps)                             ║"
	@echo "║    make fetch        Fetch logs from Elasticsearch   ║"
	@echo "║    make features     Run feature engineering         ║"
	@echo "║    make train        Train Isolation Forest          ║"
	@echo "║    make predict      Run anomaly prediction          ║"
	@echo "║    make upload       Upload anomalies to ES          ║"
	@echo "║                                                      ║"
	@echo "║  AUTOMATION                                          ║"
	@echo "║    make scheduler    Start loop-based scheduler      ║"
	@echo "║    make assistant    Start AI (Gemini) assistant     ║"
	@echo "║    make run          Scheduler + assistant (tmux)    ║"
	@echo "║                                                      ║"
	@echo "║  MODEL MANAGEMENT                                    ║"
	@echo "║    make retrain      Full retrain with validation    ║"
	@echo "║    make retrain-skip Skip fetch, retrain only        ║"
	@echo "║    make evaluate     Model evaluation report         ║"
	@echo "║    make backup-list  List model backups              ║"
	@echo "║    make rollback     Rollback to latest backup       ║"
	@echo "║                                                      ║"
	@echo "║  ANALYSIS                                            ║"
	@echo "║    make trend        Print trend analysis            ║"
	@echo "║    make clusters     Print root cause clusters       ║"
	@echo "║    make report       Generate HTML anomaly report    ║"
	@echo "║                                                      ║"
	@echo "║  CLEANUP                                             ║"
	@echo "║    make clean        Remove logs and temp files      ║"
	@echo "║    make clean-data   Remove CSV datasets             ║"
	@echo "║    make clean-model  Remove model artifacts          ║"
	@echo "║    make clean-all    Full reset                      ║"
	@echo "╚══════════════════════════════════════════════════════╝"
	@echo ""

# ── Setup ──────────────────────────────────────────────────────────────────────

setup:
	@bash setup.sh

install:
	$(PIP) install -r requirements.txt

health:
	$(PYTHON) health_check.py

# ── Pipeline ───────────────────────────────────────────────────────────────────

fetch:
	$(PYTHON) fetch_logs.py

features:
	$(PYTHON) feature_engineering.py

train:
	$(PYTHON) train_model.py

predict:
	$(PYTHON) predict.py

upload:
	$(PYTHON) upload.py

pipeline: fetch predict upload
	@echo "Manual pipeline complete."

# ── Automation ─────────────────────────────────────────────────────────────────

scheduler:
	$(PYTHON) scheduler.py

assistant:
	$(PYTHON) ai_assistant.py

run:
	@bash run.sh --ai

# ── Model management ───────────────────────────────────────────────────────────

retrain:
	$(PYTHON) retrain.py train

retrain-skip:
	$(PYTHON) retrain.py train --skip-fetch

evaluate:
	$(PYTHON) evaluate.py

backup-list:
	$(PYTHON) retrain.py list

rollback:
	$(PYTHON) retrain.py rollback

# ── Analysis ───────────────────────────────────────────────────────────────────

trend:
	$(PYTHON) trend_analysis.py

clusters:
	$(PYTHON) clustering.py

detect:
	$(PYTHON) detect_anomalies.py

report:
	$(PYTHON) report.py

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	@rm -f system.log last_run.txt
	@echo "Cleaned: system.log, last_run.txt"

clean-data:
	@rm -f logs_dataset.csv anomaly_results.csv detected_anomalies.csv
	@echo "Cleaned: CSV datasets"

clean-model:
	@rm -f model.pkl vectorizer.pkl features.npy
	@echo "Cleaned: model artifacts"

clean-all: clean clean-data clean-model
	@rm -rf model_backups/ report_*.html __pycache__/ .pytest_cache/
	@echo "Full reset complete."

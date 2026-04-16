#!/usr/bin/env bash
# setup.sh — First-time setup for AI Log Monitor
# Run once before starting the system.

set -euo pipefail

YELLOW='\033[1;33m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

info()  { echo -e "${GREEN}[✔]${NC} $*"; }
warn()  { echo -e "${YELLOW}[!]${NC} $*"; }
error() { echo -e "${RED}[✘]${NC} $*"; exit 1; }

echo ""
echo "╔══════════════════════════════════════╗"
echo "║   AI Log Monitor — Setup Script      ║"
echo "╚══════════════════════════════════════╝"
echo ""

# ── Python check ──────────────────────────────────────────────────────────────
python3 --version &>/dev/null || error "Python 3 not found. Install python3."
info "Python 3 found: $(python3 --version)"

# ── pip install ───────────────────────────────────────────────────────────────
info "Installing Python dependencies..."
pip install --quiet -r requirements.txt || error "pip install failed"
info "Dependencies installed"

# ── .env check ────────────────────────────────────────────────────────────────
if [ ! -f ".env" ]; then
    warn ".env not found — copying from .env.example"
    cp .env.example .env
    warn "Edit .env with your credentials before continuing."
    echo ""
    echo "  Required keys:"
    echo "    ES_HOST, ES_USER, ES_PASS"
    echo "    GEMINI_API_KEY"
    echo "    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID"
    echo ""
    read -rp "Press Enter after editing .env to continue, or Ctrl+C to exit..."
fi
info ".env file present"

# ── Initial data fetch ────────────────────────────────────────────────────────
info "Fetching initial logs from Elasticsearch..."
python3 fetch_logs.py || error "fetch_logs.py failed. Check ES credentials in .env."

# ── Feature engineering (fit vectorizer) ─────────────────────────────────────
info "Running feature engineering (fits vectorizer)..."
python3 feature_engineering.py || error "feature_engineering.py failed"

# ── Model training ────────────────────────────────────────────────────────────
info "Training Isolation Forest model..."
python3 train_model.py || error "train_model.py failed"

# ── First prediction run ──────────────────────────────────────────────────────
info "Running first prediction..."
python3 predict.py || error "predict.py failed"

# ── Upload results ────────────────────────────────────────────────────────────
info "Uploading anomalies to Elasticsearch..."
python3 upload.py || warn "upload.py had issues — check system.log"

echo ""
echo "╔══════════════════════════════════════╗"
echo "║         Setup Complete  ✔            ║"
echo "╚══════════════════════════════════════╝"
echo ""
echo "  Start the pipeline scheduler:"
echo "    python3 scheduler.py"
echo ""
echo "  Start the AI assistant:"
echo "    python3 ai_assistant.py"
echo ""
echo "  View logs:"
echo "    tail -f system.log"
echo ""

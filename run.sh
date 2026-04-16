#!/usr/bin/env bash
# run.sh — Launch scheduler + AI assistant in parallel (tmux or background).
# Usage:
#   ./run.sh            — scheduler only
#   ./run.sh --ai       — scheduler + AI assistant (requires tmux)
#   ./run.sh --retrain  — retrain model then start scheduler

set -euo pipefail

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info() { echo -e "${GREEN}[✔]${NC} $*"; }
warn() { echo -e "${YELLOW}[!]${NC} $*"; }

MODE="scheduler"
RETRAIN=false

for arg in "$@"; do
    case $arg in
        --ai)      MODE="both"   ;;
        --retrain) RETRAIN=true  ;;
    esac
done

# ── Optional retrain ──────────────────────────────────────────────────────────
if [ "$RETRAIN" = true ]; then
    info "Retraining model..."
    python3 fetch_logs.py
    python3 feature_engineering.py
    python3 train_model.py
    info "Retrain complete"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
if [ "$MODE" = "both" ]; then
    if command -v tmux &>/dev/null; then
        info "Starting tmux session: ai-log-monitor"
        tmux new-session -d -s ai-log-monitor -x 220 -y 50 2>/dev/null || true
        tmux send-keys -t ai-log-monitor "python3 scheduler.py" Enter
        tmux split-window -h -t ai-log-monitor
        tmux send-keys -t ai-log-monitor "sleep 3 && python3 ai_assistant.py" Enter
        tmux attach-session -t ai-log-monitor
    else
        warn "tmux not found — starting scheduler in background, AI assistant in foreground"
        python3 scheduler.py &
        SCHED_PID=$!
        info "Scheduler PID: $SCHED_PID"
        python3 ai_assistant.py
        kill "$SCHED_PID" 2>/dev/null || true
    fi
else
    info "Starting pipeline scheduler..."
    python3 scheduler.py
fi

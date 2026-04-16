"""
alerts.py — Smart Telegram alert system with rate-limiting.
Prevents alert flooding by enforcing a per-severity cooldown window.
"""

import time
import requests
from logger import log_info, log_error, log_warning
from config import TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, ALERT_COOLDOWN_SECONDS

# ── In-memory last-sent tracker (severity → epoch) ────────────────────────────
_last_sent: dict[str, float] = {}

SEVERITY_EMOJI = {
    "CRITICAL": "🔴",
    "HIGH":     "🟠",
    "MEDIUM":   "🟡",
    "LOW":      "🟢",
    "INFO":     "⚪",
}


def _is_rate_limited(severity: str) -> bool:
    """Return True if this severity is still within the cooldown window."""
    last = _last_sent.get(severity, 0)
    return (time.time() - last) < ALERT_COOLDOWN_SECONDS


def send_alert(
    message: str,
    severity: str = "INFO",
    score: float = 0.0,
    timestamp: str = "",
) -> bool:
    """
    Send a Telegram alert.

    Args:
        message:   Log message / description.
        severity:  CRITICAL | HIGH | MEDIUM | LOW | INFO
        score:     Normalized anomaly score (0–1).
        timestamp: ISO timestamp of the log event.

    Returns:
        True if sent, False if rate-limited or failed.
    """
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        log_warning("Telegram credentials not configured — alert skipped")
        return False

    if _is_rate_limited(severity):
        log_debug_safe(f"Alert rate-limited for severity={severity}")
        return False

    emoji = SEVERITY_EMOJI.get(severity, "⚪")

    text = (
        f"{emoji} *AI Log Monitor Alert*\n"
        f"━━━━━━━━━━━━━━━━━━━━\n"
        f"*Severity:*  `{severity}`\n"
        f"*Score:*     `{score:.4f}`\n"
        f"*Time:*      `{timestamp or 'N/A'}`\n"
        f"*Message:*\n```\n{message[:400]}\n```"
    )

    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {
        "chat_id":    TELEGRAM_CHAT_ID,
        "text":       text,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, data=payload, timeout=10)
        resp.raise_for_status()
        _last_sent[severity] = time.time()
        log_info(f"Alert sent | severity={severity} | score={score:.4f}")
        return True
    except requests.RequestException as exc:
        log_error(f"Telegram alert failed: {exc}")
        return False


def log_debug_safe(msg: str) -> None:
    """Avoid circular import with logger."""
    try:
        from logger import log_debug
        log_debug(msg)
    except Exception:
        pass

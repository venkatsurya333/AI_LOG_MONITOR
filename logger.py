"""
logger.py — Centralized logger for AI Log Monitor.
"""

import logging
import sys

_logger = logging.getLogger("ai_log_monitor")

if not _logger.handlers:
    _logger.setLevel(logging.DEBUG)

    # File handler
    fh = logging.FileHandler("system.log", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(module)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))

    # Console handler (INFO and above only)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    ))

    _logger.addHandler(fh)
    _logger.addHandler(ch)


def log_info(msg: str) -> None:
    _logger.info(msg)


def log_error(msg: str) -> None:
    _logger.error(msg)


def log_warning(msg: str) -> None:
    _logger.warning(msg)


def log_debug(msg: str) -> None:
    _logger.debug(msg)

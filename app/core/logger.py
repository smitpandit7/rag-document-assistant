import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from app.core.config import settings

# Avoid adding duplicate handlers if get_logger() is called multiple times
_configured_loggers: set[str] = set()


def get_logger(name: str) -> logging.Logger:
    """
    Get a named logger with console + rotating file output.

    Usage (in any service or route file):
        from app.core.logger import get_logger
        logger = get_logger(__name__)

    Args:
        name: Usually __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)

    # Only configure once per logger name
    if name in _configured_loggers:
        return logger

    logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ── Console handler ──────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # ── Rotating file handler ─────────────────────────────────────────────
    # Max 5MB per file, keep last 3 files → max 15MB of logs total
    log_path = Path(settings.LOG_DIR) / "app.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = RotatingFileHandler(
        filename=str(log_path),
        maxBytes=5 * 1024 * 1024,   # 5 MB
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # Prevent log messages from bubbling up to the root logger
    logger.propagate = False

    _configured_loggers.add(name)

    return logger
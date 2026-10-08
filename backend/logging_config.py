"""
Centralized logging configuration for QuantFunc WebUI.

Supports:
- Silent, persistent file logging to logs/quantfunc.log (all levels including HTTP access).
- Clean, uncluttered console output by default (suppresses polling GET requests).
- Full detailed console output when --verbose / -v flag is passed.
"""
from __future__ import annotations

import copy
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

from uvicorn.config import LOGGING_CONFIG

# Default project logs directory
DEFAULT_LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FORMAT = "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(
    verbose: bool = False,
    log_dir: Optional[Path] = None,
) -> tuple[Path, dict]:
    """Configure Python logging and build Uvicorn log configuration.

    Args:
        verbose: If True, prints detailed debug and HTTP access logs to console.
                 If False, keeps console quiet and writes details to log file.
        log_dir: Directory where log files are stored. Defaults to <project_root>/logs.

    Returns:
        (log_file_path, uvicorn_log_config_dict)
    """
    resolved_dir = Path(log_dir) if log_dir else DEFAULT_LOG_DIR
    resolved_dir.mkdir(parents=True, exist_ok=True)
    log_file = resolved_dir / "quantfunc.log"

    # ---------- 1. Configure standard Python root logger ----------
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplication on re-init
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    # File handler: receives ALL logs (DEBUG and higher) silently in background
    file_handler = logging.handlers.RotatingFileHandler(
        str(log_file),
        maxBytes=15 * 1024 * 1024,  # 15 MB
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    root_logger.addHandler(file_handler)

    # Console handler: filtered depending on verbose flag
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if verbose else logging.INFO)
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # ---------- 2. Build Uvicorn log config ----------
    uvicorn_config = copy.deepcopy(LOGGING_CONFIG)

    # Add custom file formatter & handler
    uvicorn_config["formatters"]["file_format"] = {
        "format": LOG_FORMAT,
        "datefmt": DATE_FORMAT,
    }
    uvicorn_config["handlers"]["file_handler"] = {
        "class": "logging.handlers.RotatingFileHandler",
        "filename": str(log_file),
        "maxBytes": 15 * 1024 * 1024,
        "backupCount": 5,
        "encoding": "utf-8",
        "formatter": "file_format",
        "level": "DEBUG",
    }

    # Attach file handler to uvicorn loggers so all requests are preserved in file
    uvicorn_config["loggers"]["uvicorn"].setdefault("handlers", ["default"])
    if "file_handler" not in uvicorn_config["loggers"]["uvicorn"]["handlers"]:
        uvicorn_config["loggers"]["uvicorn"]["handlers"].append("file_handler")

    uvicorn_config["loggers"]["uvicorn.error"].setdefault("handlers", ["default"])
    if "file_handler" not in uvicorn_config["loggers"]["uvicorn.error"]["handlers"]:
        uvicorn_config["loggers"]["uvicorn.error"]["handlers"].append("file_handler")

    # uvicorn.access logger:
    # If not verbose: ONLY write to file (NEVER clutter console with polling GET /api/jobs/...)
    # If verbose: write to both console and file
    if verbose:
        uvicorn_config["loggers"]["uvicorn.access"]["handlers"] = ["access", "file_handler"]
        uvicorn_config["loggers"]["uvicorn.access"]["level"] = "INFO"
    else:
        uvicorn_config["loggers"]["uvicorn.access"]["handlers"] = ["file_handler"]
        uvicorn_config["loggers"]["uvicorn.access"]["level"] = "INFO"

    # Disable propagation to root logger to prevent duplicate console prints
    uvicorn_config["loggers"]["uvicorn"]["propagate"] = False
    uvicorn_config["loggers"]["uvicorn.error"]["propagate"] = False
    uvicorn_config["loggers"]["uvicorn.access"]["propagate"] = False

    # Quiet external noisy libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("requests").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

    return log_file, uvicorn_config


def get_logger(name: str = "quantfunc") -> logging.Logger:
    """Get a named logger with proper hierarchy."""
    return logging.getLogger(name)

"""
Logging setup.

Call setup_logging() once at the start of each script.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str | Path] = None,
    fmt: str = "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
) -> None:
    """
    Configure root logger.

    Args:
        level: Logging level string ("DEBUG", "INFO", "WARNING", "ERROR").
        log_file: Optional path to write logs to in addition to stdout.
        fmt: Log format string.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a"))

    logging.basicConfig(
        level=numeric_level,
        format=fmt,
        handlers=handlers,
        force=True,  # override any existing root handlers
    )
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("accelerate").setLevel(logging.WARNING)

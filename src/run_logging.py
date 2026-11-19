from __future__ import annotations

import logging
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger("download_llm")
logger.addHandler(logging.NullHandler())
LOG_PATH = Path("logs/download-llm.log")

_URL_CREDENTIALS = re.compile(r"(https?://)[^/\s@]+@", re.IGNORECASE)
_URL_QUERY = re.compile(r"(https?://[^\s'\"<>?]+)\?[^\s'\"<>]+", re.IGNORECASE)
_AUTHORIZATION = re.compile(r"(\b(?:Bearer|Basic)\s+)\S+", re.IGNORECASE)
_SERVICE_TOKEN = re.compile(r"\b(?:hf_|dckr_(?:pat|oat)_)[a-zA-Z0-9_-]{12,}\b")


class LogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        value = super().format(record)
        for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
            secret = os.environ.get(name, "").strip()
            if secret:
                value = value.replace(secret, "[redacted]")
        value = _SERVICE_TOKEN.sub("[redacted]", value)
        value = _AUTHORIZATION.sub(r"\1[redacted]", value)
        value = _URL_CREDENTIALS.sub(r"\1[redacted]@", value)
        return _URL_QUERY.sub(r"\1?[redacted]", value)


def log_file(level: int, message: str, *args: object, exc_info: bool = False) -> None:
    """Record debugging details already displayed by the terminal or subprocess."""
    logger.log(level, message, *args, exc_info=exc_info, extra={"file_only": True})


@contextmanager
def run_logging() -> Iterator[None]:
    run_id = uuid4().hex[:12]
    formatter = LogFormatter(f"%(asctime)s %(levelname)s [run={run_id}] %(message)s")
    console = logging.StreamHandler()
    handlers: list[logging.Handler] = [console]
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console.setFormatter(formatter)
    console.addFilter(lambda record: not getattr(record, "file_only", False))
    logger.addHandler(console)
    try:
        try:
            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
        except OSError as exc:
            logger.warning(
                "Cannot open %s; logging to console only (%s)", LOG_PATH, exc
            )
        else:
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            handlers.append(file_handler)
        yield
    finally:
        for handler in handlers:
            logger.removeHandler(handler)
            handler.close()
        logger.setLevel(previous_level)
        logger.propagate = previous_propagate

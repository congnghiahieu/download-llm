from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger("download_llm")
logger.addHandler(logging.NullHandler())
LOG_PATH = Path("logs/download-llm.log")


@contextmanager
def run_logging() -> Iterator[None]:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    console = logging.StreamHandler()
    handlers: list[logging.Handler] = [console]
    previous_level, previous_propagate = logger.level, logger.propagate
    logger.setLevel(logging.INFO)
    logger.propagate = False
    console.setFormatter(formatter)
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

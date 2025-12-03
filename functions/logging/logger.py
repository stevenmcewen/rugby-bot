from __future__ import annotations

import logging


def get_logger(name: str) -> logging.Logger:
    """
    Return a logger configured for use in Azure Functions.

    If the host has already configured handlers, we avoid adding duplicates.
    """

    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Default to INFO; the host can override via environment if desired.
    if logger.level == logging.NOTSET:
        logger.setLevel(logging.INFO)

    return logger


def log_function_start(logger: logging.Logger, function_name: str, **context) -> None:
    logger.info("Starting %s", function_name, extra={"context": context or {}})


def log_function_error(
    logger: logging.Logger, function_name: str, exc: BaseException
) -> None:
    logger.exception("Error in %s: %s", function_name, exc)



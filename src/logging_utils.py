import logging
import os
import sys
from typing import Optional

DEFAULT_LOG_PATH = os.path.join("logs", "app.log")
_EXC_HOOK_INSTALLED = False


def _install_excepthook(logger: logging.Logger):
    global _EXC_HOOK_INSTALLED
    if _EXC_HOOK_INSTALLED:
        return

    def _handle_exception(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logger.exception("Unhandled exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _handle_exception
    _EXC_HOOK_INSTALLED = True


def setup_logging(
    log_path: Optional[str] = None,
    logger_name: Optional[str] = None,
    level: int = logging.INFO,
    install_excepthook: bool = True,
):
    path = log_path or os.environ.get("AI_PLAY_LOG_PATH", DEFAULT_LOG_PATH)
    os.makedirs(os.path.dirname(path), exist_ok=True)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    abs_path = os.path.abspath(path)

    has_file = any(
        isinstance(handler, logging.FileHandler) and os.path.abspath(handler.baseFilename) == abs_path
        for handler in logger.handlers
    )
    if not has_file:
        file_handler = logging.FileHandler(path)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    has_stream = any(
        isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler)
        for handler in logger.handlers
    )
    if not has_stream:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if install_excepthook:
        _install_excepthook(logger)

    return logger

import os
import logging
from rich.logging import RichHandler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
LOG_DIR = os.getenv("MG2RAG_LOG_DIR", os.path.join(PROJECT_ROOT, "logs"))
LOG_TO_FILE = os.getenv("MG2RAG_LOG_TO_FILE", "false").lower() in {"1", "true", "yes", "on"}
if LOG_TO_FILE:
    os.makedirs(LOG_DIR, exist_ok=True)

LOG_LEVEL = getattr(logging, os.getenv("LOG_LEVEL", "debug1").upper(), logging.INFO)






def get_logger(name: str, log_file: str = None, level: int = LOG_LEVEL) -> logging.Logger:
    """
    Get a logger with a specific name and optional file logging.
    强制统一风格，无视之前的配置。
    """
    logger = logging.getLogger(name)

    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    log_file = log_file or f"{name.replace('.', '->')}.log"
    log_path = os.path.join(LOG_DIR, log_file)

    if LOG_TO_FILE:
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))
        logger.addHandler(file_handler)

    console_handler = RichHandler(show_time=True,omit_repeated_times=False)
    console_handler.setLevel(level)

    logger.setLevel(level)
    logger.addHandler(console_handler)

    return logger

import logging
import sys
from pathlib import Path


def setup_logging(
    log_level: str = "INFO",
    log_file: str | None = None,
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """Configure logging for the package.

    Args:
        log_level: The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to a log file. If None, logs will only go to stdout
        log_format: The format string for log messages
    """
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {log_level}")

    formatter = logging.Formatter(log_format)

    root_logger = logging.getLogger()
    root_logger.setLevel(numeric_level)

    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log_file is specified
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Set specific logger levels for third-party libraries
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.backends.backend_pdf").disabled = True
    logger_names = logging.getLogger().manager.loggerDict.keys()
    for logger_name in logger_names:
        if logger_name.startswith("matplotlib"):
            logging.getLogger(logger_name).disabled = True


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.

    Args:
        name: The name of the module requesting the logger

    Returns:
        A configured logger instance
    """
    return logging.getLogger(name)

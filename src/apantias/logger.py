import logging
import sys


"""
levels:
logging.debug()
logging.info()
logging.warning()
logging.error()
logging.critical()
"""


class CustomFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    def __init__(self, logger_name: str, level: str = "info"):
        # Create a logger
        self.logger = logging.getLogger(logger_name)
        levels = [
            logging.DEBUG,
            logging.INFO,
            logging.WARNING,
            logging.ERROR,
            logging.CRITICAL,
        ]
        if level not in ["debug", "info", "warning", "error", "critical"]:
            raise ValueError("Invalid level")
        level = levels[["debug", "info", "warning", "error", "critical"].index(level)]
        self.logger.setLevel(level)

        # Create a console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        # Add the formatter to the handler
        ch.setFormatter(CustomFormatter())
        # Add the handler to the logger
        self.logger.addHandler(ch)

    def get_logger(self) -> logging.Logger:
        return self.logger


# Create a global logger instance to avoid multiple handlers
global_logger = Logger("apantias", level="info").get_logger()
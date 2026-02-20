"""Handles logging.
levels:
logging.debug()
logging.info()
logging.warning()
logging.error()
logging.critical()

"""

import logging
import sys


class CustomFormatter(logging.Formatter):
    """Formatting Class"""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_string = "%(asctime)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_string + reset,
        logging.INFO: grey + format_string + reset,
        logging.WARNING: yellow + format_string + reset,
        logging.ERROR: red + format_string + reset,
        logging.CRITICAL: bold_red + format_string + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


class Logger:
    """Global Logger, used in the standard and bin_to_h5 modules"""

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
        level = levels[["debug", "info", "warning", "error", "critical"].index(level)]  # type: ignore
        self.logger.setLevel(level)

        # Create a console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        # Add the formatter to the handler
        ch.setFormatter(CustomFormatter())
        # Add the handler to the logger
        self.logger.addHandler(ch)

    def get_logger(self) -> logging.Logger:
        """Description"""
        return self.logger


class FileLogger:
    def __init__(self, log_file="apantias.log", level=logging.INFO):
        self._logger = logging.getLogger(f"apantias.file.{log_file}")
        self._logger.setLevel(level)
        self._logger.propagate = False  # Prevent logs from printing to console
        # Prevent duplicate handlers
        if not any(isinstance(h, logging.FileHandler) and h.baseFilename == log_file for h in self._logger.handlers):
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(funcName)s - %(message)s"))
            self._logger.addHandler(handler)

    def info(self, msg, *args, **kwargs):
        self._logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        self._logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        self._logger.error(msg, *args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        self._logger.debug(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        self._logger.critical(msg, *args, **kwargs)

    def get_logger(self):
        return self._logger


class NullLogger:
    def info(self, *args, **kwargs):
        pass

    def warning(self, *args, **kwargs):
        pass

    def error(self, *args, **kwargs):
        pass

    def debug(self, *args, **kwargs):
        pass

    def critical(self, *args, **kwargs):
        pass

    def get_logger(self):
        return None


# Create a global logger instance to avoid multiple handlers
global_logger = Logger("apantias", level="info").get_logger()
# use the NullLogger to suppress log output.
global_file_logger = NullLogger()
# global_file_logger = FileLogger("my_logfile.log")

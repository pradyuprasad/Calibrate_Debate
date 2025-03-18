import logging
import sys
from typing import Optional

class LoggerFactory:
    _instance = None
    _loggers = {}
    _log_file = "tournament.log"
    _initialized = False

    @classmethod
    def get_logger(cls, name: str = "logger", level: Optional[int] = None, write_to_file: bool = True) -> logging.Logger:
        """
        Get or create a logger with the given name.

        Args:
            name: The name of the logger
            level: Optional logging level to set
            write_to_file: Whether to write logs to file in addition to console

        Returns:
            A configured logger instance
        """
        # Create the singleton instance if it doesn't exist
        if cls._instance is None:
            cls._instance = cls()
            cls._initialized = True

        # Return existing logger if we've already created one with this name
        if name in cls._loggers:
            logger = cls._loggers[name]
            if level is not None:
                logger.setLevel(level)
            return logger

        # Create a new logger
        logger = logging.getLogger(name)

        # Set level if provided, otherwise INFO
        if level is not None:
            logger.setLevel(level)
        else:
            logger.setLevel(logging.INFO)

        # Prevent propagation to root logger to avoid duplicate logs
        logger.propagate = False

        # Only add handlers if the logger doesn't have any
        if not logger.handlers:
            # Add console handler
            console_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

            # Add file handler if requested
            if write_to_file:
                file_handler = logging.FileHandler(cls._log_file)
                file_handler.setFormatter(formatter)
                logger.addHandler(file_handler)

        # Cache and return the logger
        cls._loggers[name] = logger
        return logger

    @classmethod
    def set_log_file(cls, file_path: str):
        """
        Set a custom log file path

        Args:
            file_path: Path to the log file
        """
        cls._log_file = file_path

        # Update existing loggers if needed
        for logger in cls._loggers.values():
            # Remove any existing file handlers
            for handler in logger.handlers[:]:
                if isinstance(handler, logging.FileHandler):
                    logger.removeHandler(handler)

            # Add new file handler
            file_handler = logging.FileHandler(cls._log_file)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

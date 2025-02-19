import os
import sys
import logging
from logging.handlers import TimedRotatingFileHandler

def setup_logger(logger_name, log_file, level=logging.INFO):
    """Function to setup as many loggers as you want"""
    
    PROJECT_PATH = os.environ.get("PROJECT_PATH", '/app')
    LOG_DIR = os.path.join(PROJECT_PATH, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    LOG_FILE = os.path.join(LOG_DIR, log_file)

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Create a timed rotating file handler
    file_handler = TimedRotatingFileHandler(
        LOG_FILE, when="midnight", interval=1, backupCount=30
    )
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)

    # Add a stream handler for console output
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(stream_handler)

    return logger

def get_console_logger() -> logging.Logger:
    """Create a simple console logger for when no logger is provided."""
    console_logger = logging.getLogger("console_logger")
    if not console_logger.handlers:  # Only add handler if none exists
        console_logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
        console_logger.addHandler(handler)
    return console_logger
import logging
import sys
from typing import Optional

from config import settings

def setup_logging(name: Optional[str] = None) -> logging.Logger:
    """
    Set up and return a configured logger.
    """
    logger = logging.getLogger(name or "")
    logger.propagate = False

    if not logger.handlers:
        logger.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
        formatter = logging.Formatter(settings.log_format)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    """
    return setup_logging(name)

# Loggers for common modules
agent_logger = get_logger("agent")
nodes_logger = get_logger("nodes")
services_logger = get_logger("services")
ocr_logger = get_logger("services.ocr")

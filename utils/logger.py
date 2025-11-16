
"""
Logging configuration
"""

import sys
from pathlib import Path
from loguru import logger


def setup_logger(
    log_file: str = "logs/agent.log",
    level: str = "INFO",
    rotation: str = "10 MB"
):
    """Configure logger."""
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        level=level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>"
    )
    
    # Add file handler
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.add(
        log_file,
        level=level,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function} - {message}",
        rotation=rotation,
        retention="7 days",
        compression="zip"
    )
    
    logger.info("Logger configured")
    return logger

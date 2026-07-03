"""Logging configuration for VeritasRAG."""
import sys
from pathlib import Path
from loguru import logger


def setup_logging(log_dir: str = "logs", json_format: bool = False):
    """Configure loguru to write to console and a rotating file."""
    # Remove default handler
    logger.remove()

    # Ensure log directory exists
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    log_file = Path(log_dir) / "veritas.log"

    # Add console handler
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level="INFO",
    )

    # Add file handler
    logger.add(
        log_file,
        rotation="10 MB",
        retention="1 month",
        level="DEBUG",
        serialize=json_format,
    )

    return logger

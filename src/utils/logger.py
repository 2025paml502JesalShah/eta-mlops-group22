from pathlib import Path
from loguru import logger

LOG_DIR = Path("reports")
LOG_DIR.mkdir(exist_ok=True)

logger.add(
    LOG_DIR / "pipeline.log",
    level="INFO",
    rotation="1 MB",
    format="{time} | {level} | {message}",
)
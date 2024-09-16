from loguru import logger
import sys
from app.utils.config import settings

# Remove default logger
logger.remove()

# Add stdout handler
logger.add(sys.stdout, level=settings.log_level)

# Add file handler
logger.add("logs/ml_service.log", rotation="1 MB", level=settings.log_level)

from src.constants.constants import *
import logging
import sys
import os

os.makedirs(LOG_DIR, exist_ok=True)

log_file = os.path.join(LOG_DIR, LOG_FILE)

logging.basicConfig(
    level=logging.INFO,
    format=LOGGING_FORMAT,
    datefmt=DATEFMT,
    handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(log_file)],
)

logger = logging.getLogger(__name__)

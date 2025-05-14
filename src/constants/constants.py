# Application metadata
PROJECT_NAME = "Dockerize Model"

# Paths and file locations
LOG_DIR = "logs"
LOG_FILE = "logging.log"


# Logging configuration
LOGGING_FORMAT = (
    "[%(asctime)s] [%(levelname)s] [%(name)s:%(funcName)s:%(lineno)d] - %(message)s"
)
DATEFMT = "%Y-%m-%d %H:%M:%S"


# DATA SCRIPT
FRUIT_CATEGORIES = [
    "fresh_peaches",
    "fresh_pomegranates",
    "fresh_strawberries",
    "rotten_peaches",
    "rotten_pomegranates",
    "rotten_strawberries",
]

# Processing constant
TEST_SIZE = 0.2
R_S = 0

# Visualization constant
FIG_SIZE = (8, 8)

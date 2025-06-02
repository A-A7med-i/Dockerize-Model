# Paths
MODEL_PATH = "/media/ahmed/Data/Dockerize-Model/models/checkpoints/model.keras"
PROCESSED_DATA = "/media/ahmed/Data/Dockerize-Model/data/processed"
RAW_DATA = "/media/ahmed/Data/Dockerize-Model/data/raw/images"
LOG_FILE = "logging.log"
LOG_DIR = "logs"


# Application metadata
PROJECT_NAME = "Dockerize Model"


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

FRUIT_MAP = {
    0: "rotten_peaches",
    1: "fresh_pomegranates",
    2: "rotten_strawberries",
    3: "rotten_pomegranates",
    4: "fresh_strawberries",
    5: "fresh_peaches",
}

FRUIT_RENAMED = {
    "fresh_peaches": "Fresh Peaches",
    "fresh_pomegranates": "Fresh Pomegranates",
    "fresh_strawberries": "Fresh Strawberries",
    "rotten_peaches": "Rotten Peaches",
    "rotten_pomegranates": "Rotten Pomegranates",
    "rotten_strawberries": "Rotten Strawberries",
}

# Processing constant
TARGET_WIDTH = 300
TEST_SIZE = 0.2
R_S = 0

# Visualization constant
FIG_SIZE = (8, 8)


# Model constant
INPUT_SHAPE = (300, 300, 3)
NUM_CLASSES = 6
LEARNING_RATE = 0.001
BASE_MODEL_TRAINABLE = False
BASE_MODEL_WEIGHTS = "imagenet"
DENSE_UNITS = 128
DENSE_ACTIVATION = "relu"
OUTPUT_ACTIVATION = "softmax"
LOSS = "sparse_categorical_crossentropy"
METRICS = ["accuracy"]
EPOCHS = (15,)
BATCH_SIZE = (32,)
VALIDATION_SPLIT = 0.1

# API constant
HOST = "0.0.0.0"
PORT = 5000
RESIZE = (300, 300)

# Project structure constant
DIRECTORIES = [
    "models/checkpoints",
    "data/raw/",
    "data/processed/",
    "src/constants",
    "src/utils",
    "src/data",
    "src/processing",
    "src/visualization",
    "src/models",
    "src/api",
    "notebooks",
]

FILES = [
    "src/__init__.py",
    "src/constants/constants.py",
    "src/constants/__init__.py",
    "src/utils/__init__.py",
    "src/utils/helper.py",
    "src/data/__init__.py",
    "src/data/implement_data.py",
    "src/processing/__init__.py",
    "src/processing/processor.py",
    "src/visualization/__init__.py",
    "src/visualization/plot.py",
    "src/models/__init__.py",
    "src/models/model.py",
    "src/api/__init__.py",
    "src/api/endpoints.py",
    "src/api/main.py",
    "src/api/schemas.py",
    "src/api/process.py",
    "models/checkpoints/.gitkeep",
    "requirements.txt",
    "notebooks/EDA.ipynb",
    "notebooks/experiments.ipynb",
    "notebooks/exploratory.ipynb",
    "DockerFile",
    ".gitignore",
    "README.md",
    "LICENSE",
    "setup.py",
]

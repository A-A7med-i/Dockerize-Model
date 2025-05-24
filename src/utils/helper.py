from src.constants.constants import LOG_DIR, LOG_FILE
from typing import Optional
import tensorflow as tf
from src import logger

import os


def log_separator(
    message: str = "NEW RUN STARTED", width: int = 100, separator_char: str = "-"
) -> None:
    """
    Prints a separator line to the log file to clearly mark the start of a new run.

    Args:
        message (str): The message to display in the center of the separator.
        width (int): The total width of the separator line.
        separator_char (str): The character to use for the separator line.

    Returns:
        None
    """
    separator_line = message.center(width, separator_char)
    log_file_path = os.path.join(LOG_DIR, LOG_FILE)

    with open(log_file_path, "a") as file:
        file.write(f"{separator_line}\n")


def load_model(path: str) -> Optional[tf.keras.Model]:
    """
    Load a TensorFlow Keras model from the specified path.

    Args:
        path (str): The file path to the saved model.

    Returns:
        Optional[tf.keras.Model]: The loaded model if successful, None otherwise.

    Raises:
        FileNotFoundError: If the model file does not exist.
    """

    if not os.path.exists(path):

        logger.error(f"Model path does not exist: {path}")
        raise FileNotFoundError(f"Model file not found at: {path}")

    try:
        model = tf.keras.models.load_model(path)
        logger.info(f"Model successfully loaded from: {path}")
        return model

    except (IOError, ImportError) as e:
        logger.error(f"Failed to load model from {path}: {str(e)}")
        return None

    except Exception as e:
        logger.error(f"Unexpected error loading model from {path}: {str(e)}")
        return None

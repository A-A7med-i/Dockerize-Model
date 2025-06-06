from src.constants.constants import LOG_DIR, LOG_FILE
from typing import Any, Dict, List, Union, Optional
import tensorflow as tf
from src import logger
import json
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


def save_json(data: List, file_path: str) -> None:
    """Save data to a JSON file with proper formatting.

    Args:
        data: The data to save (dict or list)
        file_path: Path where to save the JSON file
    """
    try:
        with open(file_path, "w", encoding="utf-8") as file:
            json.dump(data, file, ensure_ascii=False)
    except (IOError, TypeError) as e:
        raise Exception(f"Failed to save JSON to {file_path}: {e}")


def load_json(file_path: str) -> List:
    """Load data from a JSON file.

    Args:
        file_path: Path to the JSON file to load

    Returns:
        The loaded JSON data (dict or list)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return json.load(file)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format in {file_path}: {e}")
    except IOError as e:
        raise Exception(f"Failed to read JSON from {file_path}: {e}")
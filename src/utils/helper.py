from src.constants.constants import *
from typing import Any, Dict, Union
from pathlib import Path
import yaml
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


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read and parse YAML configuration file.

    Args:
                    path: Path to YAML config file (string or Path object)

    Returns:
                    Dict containing configuration parameters

    Raises:
                    FileNotFoundError: If config file doesn't exist
                    yaml.YAMLError: If the file contains invalid YAML
    """
    path = Path(path)

    try:
        with path.open("r") as file:
            return yaml.safe_load()

    except FileNotFoundError:
        raise FileNotFoundError(f"Config file not found at: {path}")

from src.constants.constants import *
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

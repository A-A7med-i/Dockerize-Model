from src.constants.constants import DIRECTORIES, FILES
from dataclasses import dataclass
from typing import List


@dataclass
class ProjectStructure:
    """
    Defines the project directory and file structure.

    Attributes:
        directories (List[str]): List of directory paths to create
        files (List[str]): List of files to create
    """

    directories: List[str] = None
    files: List[str] = None

    def __init__(self):
        """Initialize default project structure."""
        self.directories = DIRECTORIES
        self.files = FILES

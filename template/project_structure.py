from src.constants.constants import *
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
        self.directories = ["data/raw/", "data/processed/", "src/constants", "config"]

        self.files = [
            "setup.py",
            "src/__init__.py",
            "src/constants/constants.py",
            "src/constants/__init__.py",
            "config/config.yml",
            ".gitignore",
            "README.md",
            "LICENSE",
            "requirements.txt",
        ]

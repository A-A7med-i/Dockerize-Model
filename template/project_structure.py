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
        self.directories = [
            "data/raw/",
            "data/processed/",
            "src/constants",
            "src/utils",
            "src/data",
            "src/processing",
            "src/visualization",
            "config",
        ]

        self.files = [
            "config/config.yml",
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
            "requirements.txt",
            ".gitignore",
            "README.md",
            "LICENSE",
            "setup.py",
        ]

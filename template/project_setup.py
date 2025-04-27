from .project_structure import ProjectStructure
from pathlib import Path
import logging
import sys
import os


class ProjectSetup:
    """
    Handles project setup and structure creation.

    Attributes:
        project_name (str): Name of the project
        log_dir (Path): Directory for storing logs
        logger (logging.Logger): Logger instance
        structure (ProjectStructure): Project structure configuration
    """

    def __init__(self, project_name: str, log_dir: str = "logs") -> None:
        """
        Initialize ProjectSetup with project name and log directory.

        Args:
            project_name (str): Name of the project
            log_dir (str): Directory path for logs
        """
        self.project_name = project_name
        self.log_dir = Path(log_dir)
        self.setup_logging()
        self.logger = logging.getLogger(__name__)
        self.structure = ProjectStructure()

    def setup_logging(self) -> None:
        """Configure logging with file and console handlers."""
        self.log_dir.mkdir(exist_ok=True)
        log_file = self.log_dir / "logging.log"

        logging.basicConfig(
            level=logging.INFO,
            format="[%(asctime)s]: %(message)s",
            datefmt="%Y-%m-%d %H-%M-%S",
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout),
            ],
        )

    def create_directory(self, path: Path) -> None:
        """
        Create a directory if it doesn't exist.

        Args:
            path (Path): Directory path to create

        Raises:
            OSError: If directory creation fails
        """
        try:
            if path.exists():
                self.logger.info(f"Directory already exists: {path}")
            else:
                os.makedirs(path, exist_ok=True)
                self.logger.info(f"Created directory: {path}")

        except OSError as e:
            self.logger.error(f"Error creating directory {path}: {e}")
            raise

    def create_file(self, path: Path) -> None:
        """
        Create a file if it doesn't exist.

        Args:
            path (Path): File path to create

        Raises:
            OSError: If file creation fails
        """
        try:
            if path.exists():
                self.logger.info(f"File already exists: {path}")
            else:
                path.touch(exist_ok=True)
                self.logger.info(f"Created file: {path}")

        except OSError as e:
            self.logger.error(f"Error creating file {path}: {e}")
            raise

    def create_structure(self) -> None:
        """
        Create the project directory structure and files.

        Args:
            project_name (str): Name of the project

        Raises:
            ValueError: If project name is empty
            OSError: If directory or file creation fails
        """
        if not self.project_name:
            self.logger.error("Project name cannot be empty.")
            raise ValueError("Project name cannot be empty.")

        for dir_path_str in self.structure.directories:
            self.create_directory(Path(dir_path_str))

        for file_path_str in self.structure.files:
            file_path = Path(file_path_str)
            file_path.parent.mkdir(parents=True, exist_ok=True)
            self.create_file(file_path)

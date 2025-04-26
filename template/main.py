from template.project_setup import ProjectSetup
from src.constants.constants import *
import sys


def main():
    try:
        setup = ProjectSetup(PROJECT_NAME)
        setup.create_structure()
        setup.logger.info(
            f"Project structure for '{PROJECT_NAME}' created successfully."
        )

    except (ValueError, OSError) as e:
        setup.logger.error(f"Failed to create project structure: {e}")
        sys.exit(1)
    except Exception as e:
        setup.logger.error(f"An unexpected error occurred: {e}")
        sys.exit(1)
    finally:
        setup.logger.info("Project setup process completed")
        setup.logger.info(f"{'-' * 50}")


if __name__ == "__main__":
    main()

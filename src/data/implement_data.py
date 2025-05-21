from src.constants.constants import FRUIT_CATEGORIES
from typing import List, Tuple, Optional
from src import logger
import numpy as np
import glob
import cv2
import os


class FruitImageDataset:
    """
    A dataset class for handling fruit images with fresh/rotten classification.

    This class provides functionality to load and process fruit images from a directory
    structure, where images are organized by fruit type and condition (fresh/rotten).
    """

    def __init__(self, dataset_root_dir: str) -> None:
        """
        Initialize the fruit image dataset.

        Parameters
        ----------
        dataset_root_dir : str
            Root directory containing the fruit image dataset
        """
        self.dataset_root_dir = dataset_root_dir
        self._image_paths_with_labels: List[Tuple[str, Optional[str]]] = []
        self.fruit_categories = FRUIT_CATEGORIES
        logger.info(
            f"Initialized FruitImageDataset with base directory: {dataset_root_dir}"
        )

    def get_all_image_paths_with_labels(
        self, refresh: bool = False
    ) -> List[Tuple[str, Optional[str]]]:
        """
        Get all image paths with their corresponding fruit category labels.

        Parameters
        ----------
        refresh : bool, optional
            If True, reload all paths even if they were already loaded, by default False

        Returns
        -------
        List[Tuple[str, Optional[str]]]
            List of tuples containing (image_path, detected_fruit_category) pairs
        """
        if self._image_paths_with_labels and not refresh:
            logger.debug("Using cached image paths with labels")
            return self._image_paths_with_labels

        self._image_paths_with_labels = []

        try:
            all_image_paths = glob.glob(os.path.join(self.dataset_root_dir, "*/*"))
            logger.info(f"Found {len(all_image_paths)} images")

            for image_path in all_image_paths:
                parent_dir_name = os.path.basename(os.path.dirname(image_path))

                detected_fruit_category = None
                for fruit_category in self.fruit_categories:
                    if fruit_category.lower() in parent_dir_name.lower():
                        detected_fruit_category = fruit_category
                        break
                self._image_paths_with_labels.append(
                    (image_path, detected_fruit_category)
                )
                logger.debug(f"Processed {len(self._image_paths_with_labels)} images")

            return self._image_paths_with_labels

        except Exception as e:
            logger.error(f"Failed to get image paths: {str(e)}")
            return []

    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """
        Load a single image from the given path.

        Parameters
        ----------
        image_path : str
            Path to the image file

        Returns
        -------
        Optional[np.ndarray]
            Numpy array containing the RGB image, or None if loading fails
        """
        try:
            image_bgr = cv2.imread(image_path)
            if image_bgr is None:
                logger.error(f"Failed to load image: {image_path}")
                return None

            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            logger.debug(f"Successfully loaded image: {image_path}")
            return image_rgb

        except Exception as e:
            logger.error(f"Error loading image {image_path}: {str(e)}")
            return None

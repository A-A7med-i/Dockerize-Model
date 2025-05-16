from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
from src.constants.constants import *
from src import logger
import numpy as np
import cv2


class ImageProcessor:
    """
    A class for processing images, including resizing, normalization, and splitting into training and testing sets.

    This processor handles common image preprocessing tasks needed for machine learning models,
    maintaining aspect ratio during resizing and properly normalizing pixel values.

    Attributes:
        target_width (int): The target width to resize images to while maintaining aspect ratio.
    """

    def __init__(self, target_width: int = 300) -> None:
        """
        Initialize the ImageProcessor with a target width for resizing.

        Parameters
        ----------
        target_width : int, default=300
            The desired width in pixels for resizing images. Height will be calculated
            to maintain the original aspect ratio.
        """
        self.target_width = target_width
        logger.info(f"Initialized ImageProcessor with target width: {target_width}")

    def resize_and_normalize(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Resize an image to the target width while maintaining aspect ratio and normalize pixel values.

        The function performs two operations:
        1. Resize the image to the target width while preserving aspect ratio
        2. Normalize pixel values to the range [0.0, 1.0]

        Parameters
        ----------
        image : np.ndarray
            The input image as a NumPy array with shape (height, width, channels)

        Returns
        -------
        Optional[np.ndarray]
            The processed image as a NumPy array with float32 data type and pixel values
            between 0 and 1, or None if processing fails

        Raises
        ------
        Exception
            If image processing fails (caught and logged)
        """
        try:
            target_height = int(self.target_width * image.shape[0] / image.shape[1])
            resized_image = cv2.resize(image, (self.target_width, target_height))
            normalized_image = resized_image.astype(np.float32) / 255.0
            return normalized_image

        except Exception as e:
            logger.error(f"Error during image resizing and normalization: {str(e)}")
            return None

    def split_data(
        self,
        data: List[Tuple[str, str]],
    ) -> Tuple[List[str], List[str], List[str], List[str]]:
        """
        Split a dataset of image paths and labels into training and testing sets.

        This method uses stratified sampling to ensure that the class distribution
        is preserved in both training and testing sets. The split ratio and random seed
        are taken from the constants module.

        Parameters
        ----------
        data : List[Tuple[str, str]]
            A list where each element is a tuple containing (image_path, label)

        Returns
        -------
        Tuple[List[str], List[str], List[str], List[str]]
            A tuple containing:
            - train_image_paths: List of image paths for training
            - test_image_paths: List of image paths for testing
            - train_labels: List of labels corresponding to training images
            - test_labels: List of labels corresponding to testing images

            Returns empty lists for all components if splitting fails

        Notes
        -----
        Uses TEST_SIZE and R_S (random state) from the constants module for
        controlling the split ratio and reproducibility.
        """
        try:
            paths, labels = zip(*data)

            X_train, X_test, y_train, y_test = train_test_split(
                paths,
                list(labels),
                test_size=TEST_SIZE,
                random_state=R_S,
                shuffle=True,
                stratify=labels,
            )
            return X_train, X_test, y_train, y_test

        except Exception as e:
            logger.error(f"Error during dataset splitting: {str(e)}")
            return [], [], [], []

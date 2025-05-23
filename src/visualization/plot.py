from src.constants.constants import FIG_SIZE, FRUIT_RENAMED
import matplotlib.pyplot as plt
from typing import List, Tuple
import seaborn as sns
import pandas as pd
import random


def plot_random_image(
    image_data_pairs: List[Tuple[str, str]],
    dataset_loader,
    image_processor,
    figsize: Tuple[int, int] = FIG_SIZE,
) -> None:
    """
    Plot a random image from the provided dataset with its label as the title.

    Parameters
    ----------
    image_data_pairs : List[Tuple[str, str]]
        List of tuples containing (image_path, label) pairs
    dataset_loader : object
        Object with a load_image method to load images from paths
    image_processor : object
        Object with a resize_and_normalize method to process images
    figsize : Tuple[int, int], default=(8, 8)
        Figure size as (width, height) in inches

    Returns
    -------
    None
        This function displays a plot but doesn't return any value

    Notes
    -----
    If image loading or processing fails, an error message is logged
    and no image is displayed.
    """

    random_index = random.randint(0, len(image_data_pairs) - 1)
    image_path, image_label = image_data_pairs[random_index]

    image_label = FRUIT_RENAMED[image_label]

    raw_image = dataset_loader.load_image(image_path)

    processed_image = image_processor.resize_and_normalize(raw_image)

    plt.figure(figsize=figsize)
    plt.imshow(processed_image)

    plt.title(f"Label: {image_label}")

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def plot_label_counts(
    image_data_pairs: List[Tuple[str, str]], fig_size=FIG_SIZE
) -> None:
    """
    Plot a count of labels in the dataset using seaborn's countplot.

    Parameters
    ----------
    image_data_pairs : List[Tuple[str, str]]
        List of tuples containing (image_path, label) pairs
    """
    try:
        _, labels = zip(*image_data_pairs)

        labels_series = pd.Series(labels, name="Label")

        plt.figure(figsize=fig_size)
        ax = sns.countplot(x=labels_series)

        for i in ax.containers:
            ax.bar_label(i)

        plt.title("Distribution of Image Labels")
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting label distribution: {str(e)}")

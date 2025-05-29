from src.constants.constants import RESIZE
from PIL import Image
import numpy as np
import io


def process_image(image_content: bytes) -> np.ndarray:
    """
    Process image bytes into the format required by the model.

    Args:
        image_content: Raw bytes of the image file

    Returns:
        np.ndarray: Processed image ready for model prediction

    Raises:
        UnidentifiedImageError: If the image cannot be processed
    """
    image = Image.open(io.BytesIO(image_content))

    if image.mode != "RGB":
        image = image.convert("RGB")

    image = image.resize(RESIZE)

    return np.expand_dims(np.array(image) / 255.0, axis=0)

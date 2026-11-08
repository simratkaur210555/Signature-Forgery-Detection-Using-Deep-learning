import cv2
import numpy as np

from .config import IMG_HEIGHT, IMG_WIDTH


def preprocess_signature(image_path):
    """
    Preprocess a CEDAR signature image.

    Steps:
    1. Read image as grayscale
    2. Remove unnecessary white borders
    3. Resize while maintaining aspect ratio
    4. Place signature in the center of a fixed-size canvas
    5. Normalize pixel values to 0-1
    6. Add CNN channel dimension

    Returns:
        numpy array of shape (155, 220, 1)
    """

    # ------------------------------------------
    # 1. Read image
    # ------------------------------------------

    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # ------------------------------------------
    # 2. Remove unnecessary white borders
    # ------------------------------------------

    # Convert signature to foreground
    _, threshold = cv2.threshold(
        image,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Find bounding box around signature
    coordinates = cv2.findNonZero(threshold)

    if coordinates is not None:

        x, y, width, height = cv2.boundingRect(
            coordinates
        )

        image = image[
            y:y + height,
            x:x + width
        ]

    # ------------------------------------------
    # 3. Resize while preserving aspect ratio
    # ------------------------------------------

    height, width = image.shape

    scale = min(
        IMG_WIDTH / width,
        IMG_HEIGHT / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    image = cv2.resize(
        image,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # ------------------------------------------
    # 4. Create white canvas
    # ------------------------------------------

    canvas = np.ones(
        (IMG_HEIGHT, IMG_WIDTH),
        dtype=np.uint8
    ) * 255

    # ------------------------------------------
    # 5. Center signature
    # ------------------------------------------

    x_offset = (
        IMG_WIDTH - new_width
    ) // 2

    y_offset = (
        IMG_HEIGHT - new_height
    ) // 2

    canvas[
        y_offset:y_offset + new_height,
        x_offset:x_offset + new_width
    ] = image

    # ------------------------------------------
    # 6. Normalize pixels
    # ------------------------------------------

    canvas = (
        canvas.astype(np.float32) / 255.0
    )

    # ------------------------------------------
    # 7. Add channel dimension
    # ------------------------------------------

    canvas = np.expand_dims(
        canvas,
        axis=-1
    )

    return canvas
"""Prediction and dataset visualization functionality.

Provides tools to draw detected bounding boxes on images, crop detected vehicle
regions for downstream color analysis, and inspect dataset color categories
in a visual matplotlib grid.
"""

from itertools import islice
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

try:
    from loader import save_file
except ImportError:
    from src.loader import save_file


def draw_bounding_boxes(
    image_path: str | Path,
    detections: list[dict[str, Any]],
) -> Path:
    """Draw bounding boxes around detected objects and crop individual vehicles.

    Crops each detected vehicle from an unannotated clone of the image first to
    ensure annotation lines do not affect downstream color classification.
    Then draws labeled bounding boxes on the original image and saves both
    the crops and the annotated image to disk.

    Args:
        image_path: Path to the source image file.
        detections: List of detection dictionaries containing 'bbox',
                    'class_name', and 'confidence'.

    Returns:
        Path: The file path to the saved annotated image.
    """
    image = Image.open(str(image_path))
    draw = ImageDraw.Draw(image)
    crop_image = image.copy()  # Clone unannotated image for clean vehicle crops

    for i, detection in enumerate(detections):
        bbox = detection["bbox"]
        class_name = detection["class_name"]
        confidence = detection["confidence"]

        # 1. Crop vehicle region from clean image copy
        cropped_img = crop_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))
        save_file(image_path, cropped_img, name=f"cropped_{class_name}_{i}.png")

        # 2. Draw rectangle and confidence label on display image
        draw.rectangle(bbox, outline="red", width=2)
        draw.text(
            (bbox[0], bbox[1]),
            f"{class_name} ({confidence:.2f})",
            fill="red",
        )

    # Save and return the annotated final image
    output_path = save_file(image_path, image, name="output_image.png")
    return output_path


def visualize_color_dataset(color: str, samples_count: int = 25) -> None:
    """Display a grid of sample vehicle images from a specific color category.

    Args:
        color: The color category folder name (e.g. 'red', 'black', 'blue').
        samples_count: Number of images to render in the grid (default: 25).

    Raises:
        FileNotFoundError: If the specified color directory does not exist.
    """
    folder_path = Path("dataset/Car_colors/train") / color
    if not folder_path.exists():
        raise FileNotFoundError(f"Color folder not found: {folder_path}")

    # Display images in a 5x5 grid using matplotlib
    plt.figure(figsize=(10, 10))
    valid_extensions = {".jpg", ".jpeg", ".png"}

    image_files = [
        f for f in islice(folder_path.iterdir(), samples_count * 2)
        if f.is_file() and f.suffix.lower() in valid_extensions
    ][:samples_count]

    for i, image_file in enumerate(image_files):
        img = Image.open(str(image_file))
        plt.subplot(5, 5, i + 1)
        plt.title(image_file.name, fontsize=8)
        plt.imshow(img)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


# Backward-compatible aliases for existing callers
draw_rects = draw_bounding_boxes
test_color_data = visualize_color_dataset


if __name__ == "__main__":
    # Example standalone usage
    visualize_color_dataset("red")
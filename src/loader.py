"""File and path loading utilities for the vehicle analyzer system.

Provides helper routines to resolve project-relative paths regardless
of the current working directory, and to save image outputs safely.
"""

from pathlib import Path
from typing import Any


def get_test_image_path(path: str | Path = "test/test_image.png") -> Path:
    """Resolve an image path relative to the project root.

    Args:
        path: A relative or absolute path string or Path object.
              Defaults to "test/test_image.png".

    Returns:
        Path: An absolute Path object pointing to the target image file.
    """
    # Navigate up from src/ to the project repository root
    project_root = Path(__file__).resolve().parents[1]
    candidate = Path(path)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    return candidate.resolve()


def save_file(image_path: str | Path, image: Any, name: str = "output_image.png") -> Path:
    """Save an image to disk in the same directory as the source image.

    Args:
        image_path: The reference source image path (directory is used).
        image: A PIL Image or compatible object with a .save() method.
        name: The filename for the saved output. Defaults to "output_image.png".

    Returns:
        Path: The absolute path where the image was saved.
    """
    output_path = Path(image_path).parent / name
    image.save(output_path)
    print(f"Output image saved to: {output_path}")
    return output_path

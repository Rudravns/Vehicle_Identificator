"""Application entry point for vehicle detection and color recognition.

Orchestrates the two-stage pipeline:
1. Object detection & vehicle localization via YOLOv11.
2. Vehicle cropping and color classification via a custom TensorFlow CNN.
"""

import os
from pathlib import Path
from typing import Any

# Support both direct execution (`python src/main.py`) and package imports (`python -m src.main`)
try:
    import analyzer
    import detector
    import loader
    import visualizer
except ImportError:
    from src import analyzer, detector, loader, visualizer


class VehicleAnalyzerApp:
    """High-level application orchestrator for vehicle detection and color analysis."""

    def __init__(self, clear_screen: bool = True):
        """Initialize application state.

        Args:
            clear_screen: Whether to clear the terminal window upon startup.
        """
        if clear_screen:
            os.system("cls" if os.name == "nt" else "clear")

    def identify_objects(self, image_path: str | Path = "test/test_image.png") -> list[dict[str, Any]]:
        """Detect vehicles in an image, draw bounding boxes, and save cropped vehicle patches.

        Args:
            image_path: Path to the target image file.

        Returns:
            list[dict[str, Any]]: Bounding box detection records with confidence scores.
        """
        detections = detector.detect_vehicles(image_path)
        visualizer.draw_bounding_boxes(image_path, detections)
        return detections

    def test_color_data(self, color: str = "red") -> None:
        """Display a visual 5x5 grid of training images for a given color category.

        Args:
            color: The color name folder to preview (e.g. 'red', 'black', 'blue').
        """
        visualizer.visualize_color_dataset(color)

    def predict_color(
        self,
        image_path: str | Path,
        model_path: str | Path = "car_color_model.keras",
    ) -> str:
        """Classify the color of a vehicle from an input image.

        Args:
            image_path: Path to the cropped vehicle image to classify.
            model_path: Path to the trained Keras model file.

        Returns:
            str: The predicted color name.
        """
        color = analyzer.predict_vehicle_color(model_path, image_path)
        print(f"Predicted color for {image_path}: {color}")
        return color


# Backward-compatible alias for existing callers
Main = VehicleAnalyzerApp


if __name__ == "__main__":
    app = Main()
    app.predict_color("test/cropped_car_7.png")
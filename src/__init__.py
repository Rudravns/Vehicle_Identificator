"""Vehicle detection and color classification package.

A two-stage computer vision and deep learning system that detects vehicles
using YOLOv11 and classifies their colors using a custom TensorFlow CNN.
"""

__version__ = "1.0.0"

try:
    from src.analyzer import apply_model_to_image, predict_vehicle_color
    from src.detector import detect_vehicles, test
    from src.loader import get_test_image_path, save_file
    from src.main import Main, VehicleAnalyzerApp
    from src.train import check_available_processor, train_color_classifier
    from src.visualizer import (
        draw_bounding_boxes,
        draw_rects,
        test_color_data,
        visualize_color_dataset,
    )
except ImportError:
    from analyzer import apply_model_to_image, predict_vehicle_color
    from detector import detect_vehicles, test
    from loader import get_test_image_path, save_file
    from main import Main, VehicleAnalyzerApp
    from train import check_available_processor, train_color_classifier
    from visualizer import (
        draw_bounding_boxes,
        draw_rects,
        test_color_data,
        visualize_color_dataset,
    )

__all__ = [
    "VehicleAnalyzerApp",
    "Main",
    "detect_vehicles",
    "predict_vehicle_color",
    "apply_model_to_image",
    "draw_bounding_boxes",
    "draw_rects",
    "visualize_color_dataset",
    "test_color_data",
    "get_test_image_path",
    "save_file",
    "train_color_classifier",
    "check_available_processor",
    "test",
]
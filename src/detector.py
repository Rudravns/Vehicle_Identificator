"""Vehicle object detection functionality using Ultralytics YOLOv11.

Loads the YOLOv11 Nano detector weights and performs object detection
on input scenes, extracting bounding boxes, confidence scores, and class labels.
"""

from pathlib import Path
from typing import Any

from ultralytics import YOLO

try:
    from loader import get_test_image_path
except ImportError:
    from src.loader import get_test_image_path


def detect_vehicles(
    image_path: str | Path | None = None,
    model_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Run object detection on an image and return detected objects.

    Args:
        image_path: Path to the target image file. If None, defaults to
                    the sample test image via `get_test_image_path()`.
        model_path: Path to the YOLO weights file. If None, resolves to
                    `yolo11n.pt` at the project repository root.

    Returns:
        list[dict[str, Any]]: A list of dictionaries for each detected object:
            - class_id (int): Numerical class index.
            - class_name (str): Label name (e.g. 'car', 'truck', 'bus').
            - confidence (float): Detection confidence score (0.0 to 1.0).
            - bbox (list[float]): Bounding box coordinates [x1, y1, x2, y2].

    Raises:
        FileNotFoundError: If the target image file does not exist.
    """
    # Resolve the model path (default: project root / yolo11n.pt)
    project_root = Path(__file__).resolve().parents[1]
    if model_path is None:
        target_model = project_root / "yolo11n.pt"
    else:
        target_model = Path(model_path)

    # Resolve target image path
    if image_path is None:
        target_image = get_test_image_path()
    else:
        target_image = Path(image_path)
        if not target_image.is_absolute():
            target_image = project_root / target_image

    if not target_image.exists():
        raise FileNotFoundError(f"Target image not found: {target_image}")

    # Load YOLO model and execute inference
    model = YOLO(str(target_model))
    results = model(str(target_image))
    detections: list[dict[str, Any]] = []

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = result.names.get(class_id, str(class_id))
            detections.append({
                "class_id": class_id,
                "class_name": class_name,
                "confidence": confidence,
                "bbox": box.xyxy[0].tolist(),
            })

    return detections


# Backward-compatible alias for existing tests and callers
test = detect_vehicles
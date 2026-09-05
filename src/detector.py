"""Object detection functionality."""

from pathlib import Path

from ultralytics import YOLO
from loader import get_test_image_path

def test():
    """Run object detection on the sample image and return detected objects."""
    model = YOLO(str(Path(__file__).resolve().parents[1] / "yolo11n.pt"))
    image_path = get_test_image_path()

    if not image_path.exists():
        raise FileNotFoundError(f"Test image not found: {image_path}")

    results = model(str(image_path))
    detections = []

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
"""Prediction visualization functionality."""


from PIL.Image import Image
from PIL import Image, ImageDraw

from get_path import save_file

def draw_rects(image_path, detections):
        """Draw rectangles around detected objects on the image."""

        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)

        for detection in detections:
            bbox = detection["bbox"]
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1]), f"{detection['class_name']} ({detection['confidence']:.2f})", fill="red")

        output_path = save_file(image_path, image)
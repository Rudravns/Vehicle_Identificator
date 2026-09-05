"""Prediction visualization functionality."""


from itertools import islice
from PIL.Image import Image
from PIL import Image, ImageDraw
from pathlib import Path
import matplotlib.pyplot as plt
from loader import save_file

def draw_rects(image_path, detections):
        """Draw rectangles around detected objects on the image."""

        image = Image.open(image_path)
        draw = ImageDraw.Draw(image)
        crop_image = image.copy()  # Create a copy of the original image for cropping

        for i, detection in enumerate(detections):
            bbox = detection["bbox"]

            #first crop the detected object from the cloned img so red rects cant effect the AI
            cropped_img = crop_image.crop((bbox[0], bbox[1], bbox[2], bbox[3]))  # Crop the detected object
            save_file(image_path, cropped_img, name=f"cropped_{detection['class_name']}_{i}.png")  # Save the cropped image

            # Draw the rectangle and label on the original image
            draw.rectangle(bbox, outline="red", width=2)
            draw.text((bbox[0], bbox[1]), f"{detection['class_name']} ({detection['confidence']:.2f})", fill="red")
         


        output_path = save_file(image_path, image)




def test_color_data(color: str):
    """load the color data."""
    folder_path = Path("dataset/Car_colors/train") / color
    if not folder_path.exists():
        raise FileNotFoundError(f"Color folder not found: {folder_path}")

    #using matplotlib to display the images in the folder
    plt.figure(figsize=(10, 10))
    for i, image_file in enumerate(islice(folder_path.iterdir(), 25)):
         
        if image_file.is_file() and image_file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
            img = Image.open(image_file)
            plt.subplot(5, 5, i + 1)  # Adjust the number of rows and columns as needed
            plt.title(image_file.name)
            plt.imshow(img)
            plt.axis("off")

    plt.show()

if __name__ == "__main__":
    # Example usage
    test_color_data("red")  # Replace "red" with the desired color folder name
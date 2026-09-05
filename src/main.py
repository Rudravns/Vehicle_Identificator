"""Application entry point."""


import detector, visualizer, analyzer, loader
import os


class Main:
    def __init__(self):
        #reset  
        os.system('cls' if os.name == 'nt' else 'clear')
        #print("Cleared")

    def identify_objects(self, image_path: str):
        """Identify objects in the given image."""
        detections = detector.test()
        visualizer.draw_rects(image_path, detections)
        return detections

    def test_color_data(self, color: str):
        """Test color data visualization."""
        visualizer.test_color_data(color)

    def predict_color(self, image_path: str):
        """Predict the color of the car in the given image."""
        # Implement your color prediction logic here
        color =  analyzer.apply_model_to_image("car_color_model.keras", image_path)
        print(f"Predicted color for {image_path}: {color}")

if __name__ == "__main__":
    app = Main()
    app.predict_color("test/cropped_car_7.png")  # Replace with the path to your test image

# use this to run:  .venv\Scripts\python.exe src\main.py
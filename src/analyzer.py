"""Vehicle color classification analysis using a trained TensorFlow CNN model.

Preprocesses input vehicle crop images to 224x224 RGB tensors, applies the
trained Keras color classification model, and maps prediction indices to
human-readable color class labels.
"""

from pathlib import Path
import tensorflow as tf

# Standard 15 vehicle color classes in alphabetical order
DEFAULT_COLOR_CLASSES = [
    "beige", "black", "blue", "brown", "gold",
    "green", "grey", "orange", "pink", "purple",
    "red", "silver", "tan", "white", "yellow",
]


def predict_vehicle_color(model_path: str | Path, image_path: str | Path) -> str:
    """Classify the color of a vehicle from a cropped image.

    Args:
        model_path: Path to the trained Keras model file (.keras).
        image_path: Path to the vehicle image (cropped patch or full image).

    Returns:
        str: The predicted color name (e.g. 'black', 'red', 'white').

    Raises:
        FileNotFoundError: If the model file or image file is not found.
    """
    model_file = Path(model_path)
    # Check current directory, then fallback to models/ folder if not found
    if not model_file.exists():
        fallback = Path(__file__).resolve().parents[1] / "models" / model_file.name
        if fallback.exists():
            model_file = fallback
        else:
            raise FileNotFoundError(f"Trained model not found at: {model_path}")

    target_img = Path(image_path)
    if not target_img.exists():
        raise FileNotFoundError(f"Input image not found at: {image_path}")

    # Load the trained Keras CNN model
    model = tf.keras.models.load_model(str(model_file))

    # Determine class names: inspect train folder structure or use default class list
    train_dir = Path("dataset/Car_colors/train")
    if train_dir.exists():
        discovered_classes = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])
        class_names = discovered_classes if discovered_classes else DEFAULT_COLOR_CLASSES
    else:
        class_names = DEFAULT_COLOR_CLASSES

    # Preprocess image: load at target resolution (224, 224) and expand batch dimension
    img = tf.keras.utils.load_img(str(target_img), target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    # Perform inference
    predictions = model.predict(img_array)
    predicted_class_index = int(tf.argmax(predictions[0]).numpy())

    predicted_color = class_names[predicted_class_index]
    print(f"Predicted class index: {predicted_class_index}, Color: {predicted_color}")

    return predicted_color


# Backward-compatible alias for existing callers
apply_model_to_image = predict_vehicle_color
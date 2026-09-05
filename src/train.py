"""TensorFlow CNN training pipeline for vehicle color classification.

Builds and trains a Convolutional Neural Network (CNN) using Keras to classify
vehicle images into 15 color categories, saving the trained model artifact.
"""

from pathlib import Path
from typing import Any
import tensorflow as tf


def train_color_classifier(
    dataset_dir: str | Path = "dataset/Car_colors",
    image_size: tuple[int, int] = (224, 224),
    batch_size: int = 32,
    epochs: int = 10,
    output_model_path: str | Path = "car_color_model.keras",
) -> tuple[tf.keras.Model, Any]:
    """Train a CNN model to classify vehicle colors using TensorFlow Keras.

    Args:
        dataset_dir: Path to the root directory containing 'train' and 'val' subdirectories.
        image_size: Target image dimensions (height, width). Defaults to (224, 224).
        batch_size: Batch size for training. Defaults to 32.
        epochs: Number of complete training epochs. Defaults to 10.
        output_model_path: Filepath where the trained .keras model will be saved.

    Returns:
        tuple[tf.keras.Model, Any]: The trained Keras Model and its training History object.

    Raises:
        FileNotFoundError: If the train or val directories do not exist.
    """
    dataset_path = Path(dataset_dir)
    train_dir = dataset_path / "train"
    val_dir = dataset_path / "val"

    if not train_dir.exists() or not val_dir.exists():
        raise FileNotFoundError(
            f"Dataset train/val directories not found in {dataset_path}."
        )

    # 1. Load dataset splits from directories
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        val_dir,
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical",
    )

    # 2. Build Convolutional Neural Network (CNN) architecture
    num_classes = len(train_ds.class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1.0 / 255, input_shape=(*image_size, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax"),
    ])

    # 3. Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 4. Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
    )

    # 5. Save the trained model artifact
    output_path = Path(output_model_path)
    model.save(str(output_path))
    print(f"Model training complete and saved as {output_path}")

    return model, history


def check_available_processor() -> None:
    """Check which compute hardware (GPU or CPU) is available for TensorFlow."""
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU is available for TensorFlow: {len(gpus)} device(s) found.")
    else:
        print("GPU is not available. Using CPU for TensorFlow.")


# Backward-compatible aliases for existing callers
train_colors_using_tensorflow = train_color_classifier
test_which_prossor_is_available = check_available_processor


if __name__ == "__main__":
    train_color_classifier()
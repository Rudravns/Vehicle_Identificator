"""Project configuration."""
import tensorflow as tf
from pathlib import Path

def train_colors_using_tensorflow():
    """Train a model to classify colors using TensorFlow."""
    # 1. Define paths and parameters
    dataset_dir = Path("dataset/Car_colors")
    image_size = (224, 224)
    batch_size = 32
    epochs = 10

    # 2. Load dataset splits
    train_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "train",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    val_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir / "val",
        image_size=image_size,
        batch_size=batch_size,
        label_mode="categorical"
    )

    # 3. Create a Convolutional Neural Network (CNN)
    num_classes = len(train_ds.class_names)

    model = tf.keras.Sequential([
        tf.keras.layers.Rescaling(1./255, input_shape=(224, 224, 3)),
        tf.keras.layers.Conv2D(32, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, 3, activation="relu"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(num_classes, activation="softmax")
    ])

    # 4. Compile the model
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # 5. Train and save the trained model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs
    )

    model.save("car_color_model.keras")
    print("Model training complete and saved as car_color_model.keras")


def test_which_prossor_is_available():
    """Check which processor is available for TensorFlow."""
    if tf.config.list_physical_devices("GPU"):
        print("GPU is available for TensorFlow.")
    else:
        print("GPU is not available. Using CPU for TensorFlow.")

if __name__ == "__main__":
    # Example usage
    train_colors_using_tensorflow()
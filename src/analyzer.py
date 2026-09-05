from pathlib import Path
import tensorflow as tf

def apply_model_to_image(model_path: str, image_path: str):
    """Apply a trained model to an image and map index to directory folder names."""
    model = tf.keras.models.load_model(model_path)

    # Automatically build class list from folder structure
    train_dir = Path("dataset/Car_colors/train")
    class_names = sorted([f.name for f in train_dir.iterdir() if f.is_dir()])

    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class_index = tf.argmax(predictions[0]).numpy()
    
    predicted_color = class_names[predicted_class_index]
    print(f"Predicted class index: {predicted_class_index}, Color: {predicted_color}")
    
    return predicted_color
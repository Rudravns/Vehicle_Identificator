# Models Directory

This directory houses the machine learning models utilized by the vehicle detection and color classification pipeline.

## Models Overview

### 1. `yolo11n.pt` (Vehicle Object Detector)
- **Framework**: Ultralytics YOLOv11 Nano
- **Format**: PyTorch checkpoint (`.pt`)
- **Size**: ~5.6 MB
- **Function**: Detects vehicles (cars, trucks, buses, motorcycles) in arbitrary scenes and outputs bounding box coordinates `[x1, y1, x2, y2]`, confidence scores, and class labels.
- **Used by**: `src/detector.py`

### 2. `car_color_model.keras` (Vehicle Color Classifier)
- **Framework**: TensorFlow 2.x / Keras
- **Format**: Keras Model (`.keras` zipped archive)
- **Input Dimensions**: `(224, 224, 3)` RGB images (rescaled `1./255`)
- **Architecture**:
  - Rescaling layer `(1./255)`
  - Conv2D (32 filters, 3x3 kernel, ReLU) + MaxPooling2D
  - Conv2D (64 filters, 3x3 kernel, ReLU) + MaxPooling2D
  - Flatten + Dense (128 units, ReLU)
  - Dense output layer (15 classes, Softmax)
- **Number of Classes**: 15 distinct vehicle colors
  - `beige`, `black`, `blue`, `brown`, `gold`, `green`, `grey`, `orange`, `pink`, `purple`, `red`, `silver`, `tan`, `white`, `yellow`
- **Trained by**: `src/train.py`
- **Used by**: `src/analyzer.py` via `apply_model_to_image()`

## Retraining the Color Classifier
To retrain or fine-tune the color model with updated dataset samples:
```bash
python src/train.py
```
This script will read dataset splits from `dataset/Car_colors/train` and `dataset/Car_colors/val`, train the CNN for 10 epochs, and update the saved `.keras` model.


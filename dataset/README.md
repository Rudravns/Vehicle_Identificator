# Dataset Directory

This directory contains the dataset structure and samples for vehicle detection and color recognition.

## Source Dataset

The vehicle color recognition dataset originates from Kaggle:
- **URL**: [Kaggle Vehicle Color Recognition Dataset](https://www.kaggle.com/datasets/seebicb/vehicle-color-recognition/data)

## Directory Structure

```text
dataset/
├── Car_colors/
│   ├── train/        # Training images organized by color class
│   ├── val/          # Validation images organized by color class
│   └── test/         # Testing images organized by color class
├── data.yaml         # YOLO dataset configuration
├── images/
│   ├── train/        # YOLO vehicle detection training images
│   └── val/          # YOLO vehicle detection validation images
└── labels/
    ├── train/        # YOLO annotation txt files
    ├── val/          # YOLO annotation txt files
    └── test/         # YOLO annotation txt files
```

## Color Classes (15 Classes)

The color recognition model classifies vehicles into 15 categories:
1. `beige`
2. `black`
3. `blue`
4. `brown`
5. `gold`
6. `green`
7. `grey`
8. `orange`
9. `pink`
10. `purple`
11. `red`
12. `silver`
13. `tan`
14. `white`
15. `yellow`

## Usage in Code

- `src/train.py`: Loads `dataset/Car_colors/train` and `dataset/Car_colors/val` to train `car_color_model.keras`.
- `src/analyzer.py`: Dynamically inspects `dataset/Car_colors/train` to map predicted class indices to color names.
- `src/visualizer.py`: Provides `test_color_data(color)` to inspect 25 random samples from any color category using Matplotlib.


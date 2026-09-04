# TEPEE Hackathon Project

This project uses a YOLO-based object detection pipeline to detect vehicles in images and visualize the results.

## Project goal

The app loads a sample test image, runs object detection, and draws bounding boxes around detected vehicles.

## Requirements

- Python 3.10+
- A virtual environment is recommended
- Install dependencies from `requirements.txt`

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Dataset

You need to get the dataset from here:
https://www.kaggle.com/datasets/seebicb/vehicle-color-recognition/data

Place the downloaded dataset in the project so that the folder structure matches:

```text
dataset/
  data.yaml
  images/
    test/
    train/
    val/
  labels/
    test/
    train/
    val/
```

The sample test image used by the app is expected at:

```text
dataset/images/test/test_image.png
```

## Run the app

From the project root:

```bash
.venv\Scripts\python.exe src\main.py
```

This will run the detector and save an annotated output image in the same folder as the source image.

## Project structure

```text
src/
  detector.py
  get_path.py
  main.py
  visualizer.py

dataset/
  images/
  labels/

results/
```

## Notes

- The YOLO model is loaded from the project root as `yolo11n.pt`.
- The app currently uses the provided sample test image for validation and demonstration.
- If you want to run on a different image, update the path in the helper or detector logic.

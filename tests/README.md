# Tests Directory

This directory contains automated unit and integration tests for the project.

## Running Tests

Ensure you have activated your virtual environment with test dependencies:

```bash
# Using pytest
.vehicle_analyze\Scripts\pytest.exe -v

# Or if activated:
pytest -v
```

## Test Files

- `test_detector.py`: Tests the vehicle detection pipeline:
  - `test_get_test_image_path()`: Confirms the sample test image path resolves correctly regardless of current working directory.
  - `test_detector_runs_on_test_image()`: Runs YOLOv11 detection on `test/test_image.png` and ensures detected bounding boxes, classes, and confidence scores are returned as a list.

## Adding Tests

When adding new tests:
- Prefix test file names with `test_` (e.g., `test_analyzer.py`).
- Imports from `src/` are automatically resolved via `conftest.py` in the workspace root.


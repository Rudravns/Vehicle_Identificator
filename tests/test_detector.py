"""Tests for the detector module."""

from src import detector


def test_get_test_image_path():
    image_path = detector.get_test_image_path()
    assert image_path.exists()
    assert image_path.name == "test_image.png"


def test_detector_runs_on_test_image():
    detections = detector.test()
    assert isinstance(detections, list)

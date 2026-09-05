from pathlib import Path


def get_test_image_path(path: str = "test/test_image.png") -> Path:
    """Return the project test image path regardless of the current working directory."""
    project_root = Path(__file__).resolve().parents[1]
    candidate = Path(path)

    if not candidate.is_absolute():
        candidate = project_root / candidate

    return candidate.resolve()

def save_file(image_path: str | Path, image, name: str = "output_image.png") -> None:
    output_path = Path(image_path).parent / name
    image.save(output_path)
    print(f"Output image saved to: {output_path}")


"""Utilities for validating images before TensorFlow dataset decoding."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, UnidentifiedImageError


_TENSORFLOW_IMAGE_FORMATS = {"BMP", "GIF", "JPEG", "PNG"}


def tensorflow_compatible_path(path: Path, cache_root: Path) -> Path:
    """Validate an image and convert non-TensorFlow formats into a PNG cache."""
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError(f"Cannot decode training image '{path}': {exc}") from exc

    if image_format in _TENSORFLOW_IMAGE_FORMATS:
        return path

    split_name = path.parent.parent.name
    class_name = path.parent.name
    converted_path = cache_root / split_name / class_name / f"{path.name}.png"
    if not converted_path.exists() or converted_path.stat().st_mtime < path.stat().st_mtime:
        converted_path.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(path) as image:
            image.convert("RGB").save(converted_path, format="PNG")
    return converted_path

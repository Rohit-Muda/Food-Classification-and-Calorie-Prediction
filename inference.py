"""Image validation and inference helpers shared by the Streamlit app."""

from __future__ import annotations

import hashlib
import io
from typing import Protocol

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class PredictionModel(Protocol):
    def predict(self, image_array: np.ndarray, verbose: int = 0) -> np.ndarray:
        ...


def file_hash(file_bytes: bytes) -> str:
    """Hash uploaded bytes to detect image changes in the session."""
    return hashlib.sha256(file_bytes).hexdigest()


def decode_uploaded_image(file_bytes: bytes) -> Image.Image:
    """Validate an uploaded image and return an orientation-corrected RGB copy."""
    if not file_bytes:
        raise ValueError("The uploaded file is empty.")

    with Image.open(io.BytesIO(file_bytes)) as image:
        image.verify()

    with Image.open(io.BytesIO(file_bytes)) as image:
        width, height = image.size
        if width <= 0 or height <= 0 or width * height > 40_000_000:
            raise ValueError("The image dimensions are invalid or too large.")
        return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_for_model(
    pil_image: Image.Image, image_size: tuple[int, int]
) -> np.ndarray:
    """Resize as in training, convert to RGB, and normalize pixels to [0, 1]."""
    if len(image_size) != 2 or any(size <= 0 for size in image_size):
        raise ValueError(f"Invalid model image dimensions: {image_size}.")
    image = np.asarray(pil_image.convert("RGB"), dtype=np.float32) / 255.0
    resized = tf.image.resize(image, image_size, method="bilinear")
    return resized.numpy()[np.newaxis, ...]


def run_inference(
    model: PredictionModel, image_array: np.ndarray
) -> tuple[int, np.ndarray]:
    """Return the predicted class index and its probability vector."""
    predictions = model.predict(image_array, verbose=0)
    if predictions.ndim != 2 or predictions.shape[0] != 1:
        raise ValueError(f"Expected one prediction row, got shape {predictions.shape}.")
    probabilities = predictions[0]
    if not np.isfinite(probabilities).all() or np.any(probabilities < 0):
        raise ValueError("The model returned invalid class probabilities.")
    return int(np.argmax(probabilities)), probabilities

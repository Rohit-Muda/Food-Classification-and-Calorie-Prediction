"""Image validation and inference helpers shared by the Streamlit app."""

from __future__ import annotations

import hashlib
import io
import shutil
from typing import Protocol
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


class PredictionModel(Protocol):
    def predict(self, image_array: np.ndarray, verbose: int = 0) -> np.ndarray:
        ...

    @property
    def input_shape(self) -> tuple[None, int, int, int]:
        ...

    @property
    def output_shape(self) -> tuple[None, int]:
        ...


class SavedModelPredictor:
    """Keras-like prediction adapter for a TensorFlow SavedModel signature."""

    def __init__(self, model_path: Path) -> None:
        self._model = tf.saved_model.load(str(model_path))
        try:
            self._signature = self._model.signatures["serving_default"]
        except KeyError as exc:
            raise ValueError(
                f"SavedModel at '{model_path}' has no serving_default signature."
            ) from exc

        _, keyword_specs = self._signature.structured_input_signature
        if len(keyword_specs) != 1:
            raise ValueError("SavedModel must have exactly one image input.")
        self._input_name, input_spec = next(iter(keyword_specs.items()))
        if input_spec.shape.rank != 4 or input_spec.shape[-1] != 3:
            raise ValueError(f"SavedModel input must be RGB images, got {input_spec.shape}.")

        output_specs = self._signature.structured_outputs
        if len(output_specs) != 1:
            raise ValueError("SavedModel must have exactly one classification output.")
        self._output_name = next(iter(output_specs))
        output_spec = output_specs[self._output_name]
        if output_spec.shape.rank != 2:
            raise ValueError(f"SavedModel output must be class scores, got {output_spec.shape}.")

        self.input_shape = tuple(input_spec.shape.as_list())
        self.output_shape = tuple(output_spec.shape.as_list())

    def predict(self, image_array: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        output = self._signature(
            **{self._input_name: tf.convert_to_tensor(image_array)}
        )
        return output[self._output_name].numpy()


def export_saved_model(model: tf.keras.Model, export_path: Path) -> None:
    """Export a classifier with a stable TensorFlow serving signature."""
    staging_path = export_path.with_name(f"{export_path.name}.staging")
    if staging_path.exists():
        shutil.rmtree(staging_path)

    input_shape = [None, *model.input_shape[1:]]

    @tf.function(
        input_signature=[
            tf.TensorSpec(input_shape, dtype=tf.float32, name="images")
        ]
    )
    def serve(images: tf.Tensor) -> dict[str, tf.Tensor]:
        return {"probabilities": model(images, training=False)}

    tf.saved_model.save(
        model,
        str(staging_path),
        signatures={"serving_default": serve},
    )
    if export_path.exists():
        shutil.rmtree(export_path)
    staging_path.replace(export_path)


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

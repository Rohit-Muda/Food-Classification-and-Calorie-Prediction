import io
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image, UnidentifiedImageError

from dataset_utils import tensorflow_compatible_path
from inference import (
    decode_uploaded_image,
    file_hash,
    preprocess_for_model,
    run_inference,
)


class FixedModel:
    def __init__(self, predictions: np.ndarray) -> None:
        self.predictions = predictions

    def predict(self, image_array: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.input_shape = image_array.shape
        self.verbose = verbose
        return self.predictions


class InferenceTests(unittest.TestCase):
    def setUp(self) -> None:
        image = Image.new("RGB", (12, 8), color=(255, 128, 0))
        buffer = io.BytesIO()
        image.save(buffer, format="PNG")
        self.image_bytes = buffer.getvalue()

    def test_decode_uploaded_image_returns_rgb_image(self) -> None:
        image = decode_uploaded_image(self.image_bytes)

        self.assertEqual(image.mode, "RGB")
        self.assertEqual(image.size, (12, 8))

    def test_decode_rejects_corrupt_bytes(self) -> None:
        with self.assertRaises((UnidentifiedImageError, OSError)):
            decode_uploaded_image(b"not an image")

    def test_preprocess_matches_model_input_contract(self) -> None:
        image = decode_uploaded_image(self.image_bytes)
        result = preprocess_for_model(image, (16, 20))

        self.assertEqual(result.shape, (1, 16, 20, 3))
        self.assertEqual(result.dtype, np.float32)
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)

    def test_preprocess_rejects_invalid_dimensions(self) -> None:
        with self.assertRaises(ValueError):
            preprocess_for_model(Image.new("RGB", (1, 1)), (0, 10))

    def test_inference_returns_highest_probability_index(self) -> None:
        model = FixedModel(np.array([[0.1, 0.7, 0.2]], dtype=np.float32))

        class_index, probabilities = run_inference(model, np.zeros((1, 2, 2, 3)))

        self.assertEqual(class_index, 1)
        np.testing.assert_allclose(probabilities, [0.1, 0.7, 0.2])

    def test_inference_rejects_invalid_probabilities(self) -> None:
        model = FixedModel(np.array([[0.1, np.nan]], dtype=np.float32))

        with self.assertRaises(ValueError):
            run_inference(model, np.zeros((1, 2, 2, 3)))

    def test_file_hash_is_stable_and_content_sensitive(self) -> None:
        self.assertEqual(file_hash(b"image"), file_hash(b"image"))
        self.assertNotEqual(file_hash(b"image"), file_hash(b"other"))

    def test_webp_image_is_converted_for_tensorflow_without_changing_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "Train" / "Hot Dog" / "dish.jpg"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (8, 8), color="red").save(source, format="WEBP")

            converted = tensorflow_compatible_path(
                source, Path(temp_dir) / "converted_images"
            )

            self.assertTrue(converted.is_file())
            self.assertNotEqual(converted, source)
            with Image.open(converted) as image:
                self.assertEqual(image.format, "PNG")
            with Image.open(source) as image:
                self.assertEqual(image.format, "WEBP")


if __name__ == "__main__":
    unittest.main()

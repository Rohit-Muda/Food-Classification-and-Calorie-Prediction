from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "food_classifier.keras"
CLASS_PATH = ROOT / "class_names.json"
CALORIES_PATH = ROOT / "calories.csv"


def _safe_read_model() -> tuple[tf.keras.Model, list[str]]:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")
    if not CLASS_PATH.exists():
        raise FileNotFoundError(f"Class mapping file not found: {CLASS_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    class_names = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    if not isinstance(class_names, list) or len(class_names) != model.output_shape[-1]:
        raise ValueError("The saved class mapping does not match the model output.")
    return model, class_names


@st.cache_resource
def load_model_and_classes() -> tuple[tf.keras.Model, list[str]]:
    return _safe_read_model()


@st.cache_data
def load_calories() -> pd.DataFrame:
    if not CALORIES_PATH.exists():
        return pd.DataFrame(columns=["food_class", "reference_calories_kcal", "reference_basis", "calorie_category", "source"])
    return pd.read_csv(CALORIES_PATH)


def preprocess_image(uploaded_file) -> np.ndarray:
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.asarray(image, dtype=np.float32)[None, ...]
    resized = tf.image.resize(image_array, (128, 128), method="bilinear")
    return resized / 255.0


st.set_page_config(page_title="Food Classification App", page_icon="🍽️")
st.title("Food Image Classification and Calorie Category Prediction")
st.write(
    "This app identifies the food class using a small CNN trained on five food categories and then shows a reference calorie estimate from a nutrition lookup table."
)

uploaded_file = st.file_uploader("Upload a food image (JPG, JPEG, or PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded image", use_container_width=True)
    except UnidentifiedImageError:
        st.error("The uploaded file could not be read as an image. Please upload a valid JPG, JPEG, or PNG file.")
        st.stop()

    if st.button("Predict Food Class"):
        try:
            model, class_names = load_model_and_classes()
            image_array = preprocess_image(uploaded_file)
            prediction = model.predict(image_array, verbose=0)[0]
            class_index = int(np.argmax(prediction))
            predicted_class = class_names[class_index]
            confidence = float(prediction[class_index]) * 100.0

            st.success(f"Predicted food class: {predicted_class}")
            st.metric("Prediction confidence", f"{confidence:.1f}%")
            if confidence < 50:
                st.warning("The model is uncertain about this prediction; check the other possible classes or try a clearer image.")

            calories_df = load_calories()
            match = calories_df[calories_df["food_class"] == predicted_class]
            if match.empty:
                st.warning(f"No nutrition reference is available for '{predicted_class}' in the lookup table.")
            else:
                row = match.iloc[0]
                reference_calories = float(row["reference_calories_kcal"])
                category = str(row["calorie_category"])
                source_text = str(row["source"])
                basis = str(row["reference_basis"])
                st.write(f"Reference calories: {reference_calories:.2f} kcal")
                st.write(f"Calorie category: {category}")
                st.caption(f"Reference basis: {basis}")
                st.caption(f"Source: {source_text}")

            st.caption(
                "Disclaimer: The calorie value is an approximate reference estimate, not a direct measurement from the image. It depends on serving size, ingredients, and preparation method."
            )

        except Exception as exc:
            st.error(f"Prediction failed: {exc}")
else:
    st.info("Upload a food image to begin prediction.")

st.caption("This interface is intentionally simple and suitable for a classroom demonstration.")

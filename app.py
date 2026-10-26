"""
app.py — FoodSense AI: Food Image Classification and Calorie Reference
=======================================================================
Streamlit application that accepts a food image, classifies it using the
trained model, and displays nutritional reference information.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, UnidentifiedImageError

from inference import (
    PredictionModel,
    SavedModelPredictor,
    decode_uploaded_image,
    file_hash,
    preprocess_for_model,
    run_inference,
)

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
try:
    from config import (
        ROOT, MODEL_PATH, SAVED_MODEL_PATH, CLASS_PATH, CALORIES_PATH,
        CONFIG_PATH, ALL_CLASSES,
        CONFIDENCE_THRESHOLD as CONFIDENCE_WARN,
        SUPPORTED_FILE_TYPES as SUPPORTED_TYPES,
        DEFAULT_IMAGE_SIZE as _DEFAULT_IMAGE_SIZE,
        CALORIE_LOW_MAX, CALORIE_HIGH_MIN
    )
except ImportError:
    ROOT             = Path(__file__).resolve().parent
    MODEL_PATH       = ROOT / "food_classifier.keras"
    SAVED_MODEL_PATH = ROOT / "food_classifier_saved_model"
    CLASS_PATH       = ROOT / "class_names.json"
    CALORIES_PATH    = ROOT / "calories.csv"
    CONFIG_PATH      = ROOT / "model_config.json"
    ALL_CLASSES      = [
        "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries",
        "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito",
    ]
    CONFIDENCE_WARN  = 0.55              # show warning below this threshold
    SUPPORTED_TYPES  = ["jpg", "jpeg", "png"]
    _DEFAULT_IMAGE_SIZE = (160, 160)
    CALORIE_LOW_MAX  = 300
    CALORIE_HIGH_MIN = 400


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  CACHED RESOURCE LOADING
# ─────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model() -> tuple[PredictionModel, list[str], tuple[int, int]]:
    """Load the portable model and class map; return model input image size."""
    if not SAVED_MODEL_PATH.is_dir() and not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No model found at '{SAVED_MODEL_PATH}' or '{MODEL_PATH}'. "
            "Run the model export step before starting the app."
        )
    if not CLASS_PATH.exists():
        raise FileNotFoundError(
            f"Class mapping not found at '{CLASS_PATH}'. "
            "Run train_model.py to regenerate class_names.json."
        )
    if SAVED_MODEL_PATH.is_dir():
        model = SavedModelPredictor(SAVED_MODEL_PATH)
    else:
        model = tf.keras.models.load_model(
            str(MODEL_PATH), compile=False, safe_mode=True
        )
    classes = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    if (
        not isinstance(classes, list)
        or not all(isinstance(name, str) for name in classes)
        or len(set(classes)) != len(classes)
        or classes != ALL_CLASSES
    ):
        raise ValueError(
            "class_names.json does not match the configured class order. "
            "Regenerate the mapping with the training pipeline."
        )
    if (
        not isinstance(model.input_shape, tuple)
        or len(model.input_shape) != 4
        or model.input_shape[-1] != 3
        or not isinstance(model.output_shape, tuple)
        or len(model.output_shape) != 2
        or model.output_shape[-1] != len(classes)
    ):
        raise ValueError(
            "The model must accept RGB images and output one score for each "
            f"configured class; got input {model.input_shape} and output "
            f"{model.output_shape}."
        )
    # Derive image size from model's actual input shape (H, W from shape[1], shape[2])
    in_shape = model.input_shape  # e.g. (None, 160, 160, 3)
    img_h = int(in_shape[1]) if in_shape[1] is not None else _DEFAULT_IMAGE_SIZE[0]
    img_w = int(in_shape[2]) if in_shape[2] is not None else _DEFAULT_IMAGE_SIZE[1]
    if img_h <= 0 or img_w <= 0:
        raise ValueError(f"Invalid model image dimensions: {(img_h, img_w)}.")
    return model, classes, (img_h, img_w)


def load_confidence_threshold() -> float:
    """Load the optional runtime threshold metadata, validating its range."""
    if not CONFIG_PATH.exists():
        return CONFIDENCE_WARN
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    threshold = config.get("confidence_warn_threshold", CONFIDENCE_WARN)
    if (
        isinstance(threshold, bool)
        or not isinstance(threshold, (int, float))
        or not 0.0 <= threshold <= 1.0
    ):
        raise ValueError("model_config.json has an invalid confidence warning threshold.")
    return float(threshold)


@st.cache_data(show_spinner=False)
def load_nutrition() -> pd.DataFrame:
    """Load and normalise the nutrition lookup CSV."""
    if not CALORIES_PATH.exists():
        return pd.DataFrame()
    df = pd.read_csv(CALORIES_PATH)
    df["food_class_lower"] = df["food_class"].str.strip().str.lower()
    return df


def lookup_nutrition(food_class: str, df: pd.DataFrame) -> pd.Series | None:
    """Return the matching nutrition row or None."""
    if df.empty:
        return None
    match = df[df["food_class_lower"] == food_class.strip().lower()]
    if match.empty:
        return None
    return match.iloc[0]


def calorie_category(kcal: float) -> tuple[str, str]:
    """Return (category_label, badge_colour) based on project thresholds."""
    if kcal < CALORIE_LOW_MAX:
        return "Low", "#27ae60"
    if kcal < CALORIE_HIGH_MIN:
        return "Medium", "#e67e22"
    return "High", "#e74c3c"


# ─────────────────────────────────────────────
#  PAGE CONFIG & GLOBAL CSS
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="FoodSense AI",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page background ── */
.stApp {
    background: #0f1117;
    color: #e8eaf0;
}

/* ── Section cards ── */
.card {
    background: #1a1d2e;
    border: 1px solid #2a2d40;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}

/* ── Header area ── */
.hero-title {
    font-size: 2.6rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -0.5px;
    line-height: 1.15;
    margin: 0;
}
.hero-sub {
    font-size: 1.05rem;
    color: #8b8fa8;
    margin-top: 8px;
    line-height: 1.6;
}
.hero-divider {
    height: 3px;
    background: linear-gradient(90deg, #6c63ff, #5ba4ff, #6c63ff);
    border-radius: 2px;
    margin: 18px 0 28px 0;
}

/* ── Prediction result headline ── */
.prediction-label {
    font-size: 1.9rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0;
}
.prediction-sub {
    font-size: 0.9rem;
    color: #8b8fa8;
    margin-top: 4px;
}

/* ── Confidence bar ── */
.conf-bar-container {
    background: #2a2d40;
    border-radius: 8px;
    height: 10px;
    margin: 10px 0 4px 0;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 8px;
    transition: width 0.5s ease;
}

/* ── Calorie badge ── */
.cal-badge {
    display: inline-block;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
    text-transform: uppercase;
}

/* ── Stat grid ── */
.stat-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 12px;
    margin-top: 12px;
}
.stat-cell {
    background: #0f1117;
    border: 1px solid #2a2d40;
    border-radius: 10px;
    padding: 12px 14px;
}
.stat-label {
    font-size: 0.75rem;
    color: #6b6f88;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-bottom: 4px;
}
.stat-value {
    font-size: 1.2rem;
    font-weight: 600;
    color: #e8eaf0;
}

/* ── Warning / info boxes ── */
.warn-box {
    background: rgba(230, 126, 34, 0.12);
    border: 1px solid rgba(230, 126, 34, 0.35);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #e67e22;
    margin-top: 12px;
}
.info-box {
    background: rgba(91, 164, 255, 0.08);
    border: 1px solid rgba(91, 164, 255, 0.25);
    border-radius: 10px;
    padding: 12px 16px;
    font-size: 0.88rem;
    color: #8b8fa8;
    margin-top: 12px;
}
.disclaimer {
    font-size: 0.78rem;
    color: #565972;
    margin-top: 14px;
    line-height: 1.55;
    border-top: 1px solid #2a2d40;
    padding-top: 12px;
}

/* ── Upload zone styling ── */
section[data-testid="stFileUploadDropzone"] {
    background: #1a1d2e !important;
    border: 2px dashed #3a3d52 !important;
    border-radius: 14px !important;
}
section[data-testid="stFileUploadDropzone"]:hover {
    border-color: #6c63ff !important;
}

/* ── Streamlit button override ── */
.stButton > button {
    background: linear-gradient(135deg, #6c63ff, #5ba4ff) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.55rem 1.8rem !important;
    font-size: 0.97rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 14px rgba(108, 99, 255, 0.4) !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
.stButton > button:active { transform: scale(0.98) !important; }

/* ── Metric widget ── */
[data-testid="stMetric"] {
    background: #1a1d2e;
    border-radius: 12px;
    padding: 10px 14px;
    border: 1px solid #2a2d40;
}
[data-testid="stMetricLabel"] { color: #8b8fa8 !important; }
[data-testid="stMetricValue"] { color: #e8eaf0 !important; }

/* ── Image captions ── */
.image-meta {
    font-size: 0.8rem;
    color: #6b6f88;
    margin-top: 6px;
}

/* ── Other class probabilities ── */
.other-class-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 5px 0;
    border-bottom: 1px solid #2a2d40;
    font-size: 0.88rem;
    color: #8b8fa8;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  HERO HEADER
# ─────────────────────────────────────────────

st.markdown("""
<div style="padding: 10px 0 0 0;">
  <p class="hero-title">🍽️ FoodSense AI</p>
  <p class="hero-sub">
    Upload a food photo to identify the food type and get a nutritional
    reference estimate. Supports 10 fast-food categories.
  </p>
  <div class="hero-divider"></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  LOAD ASSETS (model + nutrition)
# ─────────────────────────────────────────────

model_ok       = True
model_obj      = None
class_names    = []
image_size     = _DEFAULT_IMAGE_SIZE
nutrition_df   = pd.DataFrame()
model_load_err = ""

try:
    model_obj, class_names, image_size = load_model()
    CONFIDENCE_WARN = load_confidence_threshold()
    nutrition_df = load_nutrition()
except FileNotFoundError as exc:
    model_ok       = False
    model_load_err = str(exc)
    logger.error("Model load error: %s", exc)
except Exception as exc:
    model_ok       = False
    model_load_err = f"Unexpected error loading model: {exc}"
    logger.error("Unexpected model load error: %s", exc, exc_info=True)

if not model_ok:
    st.error(f"⚠️ **Model not available**\n\n{model_load_err}")
    st.info("Run `python train_model.py` in the project directory to train the model, then restart this app.")
    st.stop()


# ─────────────────────────────────────────────
#  LAYOUT: TWO COLUMNS
# ─────────────────────────────────────────────

left_col, right_col = st.columns([1, 1], gap="large")

# ── LEFT: upload + preview ──
with left_col:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("#### 📤 Upload Food Image")
    uploaded = st.file_uploader(
        "Drag & drop or click to browse",
        type=SUPPORTED_TYPES,
        label_visibility="collapsed",
    )

    if uploaded is not None:
        file_bytes = uploaded.read()
        if len(file_bytes) > 20 * 1024 * 1024:
            st.error("Image is too large. Please upload a file smaller than 20 MB.")
            st.stop()
        img_hash   = file_hash(file_bytes)

        # Validate image
        try:
            pil_img = decode_uploaded_image(file_bytes)
        except (
            UnidentifiedImageError,
            OSError,
            ValueError,
            Image.DecompressionBombError,
        ) as exc:
            st.error(f"Could not read the uploaded file as an image: {exc}")
            st.markdown('</div>', unsafe_allow_html=True)
            st.stop()

        # Display preview
        st.image(pil_img, use_container_width=True)
        w, h = pil_img.size
        st.markdown(
            f'<p class="image-meta">📄 {uploaded.name} &nbsp;·&nbsp; {w}×{h} px</p>',
            unsafe_allow_html=True,
        )

        # ── PREDICT button ──
        st.markdown("<br>", unsafe_allow_html=True)
        predict_clicked = st.button("🔍 Analyse Food", key="predict_btn")

        # Clear cached result if image changed
        if "last_hash" in st.session_state and st.session_state["last_hash"] != img_hash:
            for key in ["prediction_result", "last_hash"]:
                st.session_state.pop(key, None)

        if predict_clicked:
            with st.spinner("Analysing image …"):
                try:
                    arr       = preprocess_for_model(pil_img, image_size)
                    cls_idx, probs = run_inference(model_obj, arr)
                    st.session_state["prediction_result"] = {
                        "class_idx":  cls_idx,
                        "class_name": class_names[cls_idx],
                        "confidence": float(probs[cls_idx]),
                        "all_probs":  probs.tolist(),
                        "img_hash":   img_hash,
                    }
                    st.session_state["last_hash"] = img_hash
                except Exception as exc:
                    logger.error("Inference failed: %s", exc, exc_info=True)
                    st.error(f"Prediction failed: {exc}")

    else:
        for key in ("prediction_result", "last_hash"):
            st.session_state.pop(key, None)
        # Placeholder when no image is uploaded
        st.markdown("""
        <div style="text-align:center; padding: 40px 20px; color: #565972;">
            <div style="font-size: 3rem; margin-bottom: 12px;">🥗</div>
            <div style="font-size: 0.95rem;">
                Supported formats: JPG, JPEG, PNG<br>
                <span style="font-size:0.82rem; color:#3a3d52;">
                    Recognised categories: Baked Potato · Burger · Crispy Chicken · Donut ·
                    Fries · Hot Dog · Pizza · Sandwich · Taco · Taquito
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ── RIGHT: results ──
with right_col:
    result = st.session_state.get("prediction_result")

    if result is None:
        # Empty state
        st.markdown("""
        <div class="card" style="text-align:center; padding:60px 24px; color:#3a3d52;">
            <div style="font-size:3.5rem; margin-bottom:16px;">✨</div>
            <div style="font-size:1.0rem; color:#565972; line-height:1.6;">
                Upload a food image and click <strong style="color:#6c63ff;">Analyse Food</strong>
                to see the prediction and nutritional information here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        cls_name   = result["class_name"]
        confidence = result["confidence"]
        all_probs  = result["all_probs"]
        conf_pct   = confidence * 100

        # ── Confidence colour ──
        if confidence >= 0.80:
            conf_colour = "#27ae60"
        elif confidence >= CONFIDENCE_WARN:
            conf_colour = "#e67e22"
        else:
            conf_colour = "#e74c3c"

        # ── Prediction Card ──
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown(f"""
        <div>
            <div style="font-size:0.8rem; color:#6b6f88; text-transform:uppercase;
                        letter-spacing:0.6px; margin-bottom:6px;">Predicted Food</div>
            <div class="prediction-label">{cls_name}</div>
        </div>
        <div style="margin-top:16px;">
            <div style="display:flex; justify-content:space-between; align-items:center;
                        margin-bottom:4px;">
                <span style="font-size:0.82rem; color:#8b8fa8;">Model Confidence</span>
                <span style="font-size:0.95rem; font-weight:600; color:{conf_colour};">{conf_pct:.1f}%</span>
            </div>
            <div class="conf-bar-container">
                <div class="conf-bar-fill"
                     style="width:{conf_pct:.1f}%; background:{conf_colour};"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        if confidence < CONFIDENCE_WARN:
            st.markdown(
                '<div class="warn-box">⚠️ Low confidence — the model is uncertain about this image. '
                'Try a clearer, better-lit photo showing the food prominently.</div>',
                unsafe_allow_html=True,
            )

        st.markdown('</div>', unsafe_allow_html=True)

        # ── Nutritional Reference Card ──
        nutrition_row = lookup_nutrition(cls_name, nutrition_df)
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.markdown("#### 🥦 Nutritional Reference")

        if nutrition_row is not None:
            kcal         = float(nutrition_row["reference_calories_kcal"])
            basis        = str(nutrition_row.get("reference_basis", "—"))
            source       = str(nutrition_row.get("source", "—"))
            cat_label, cat_color = calorie_category(kcal)

            st.markdown(f"""
            <div style="display:flex; align-items:baseline; gap:10px; margin-bottom:4px;">
                <span style="font-size:2.2rem; font-weight:700; color:#e8eaf0;">{kcal:.0f}</span>
                <span style="font-size:1.0rem; color:#8b8fa8;">kcal&nbsp;/&nbsp;serve</span>
                <span class="cal-badge" style="background:{cat_color}22; color:{cat_color};
                      border:1px solid {cat_color}55;">{cat_label} Calorie</span>
            </div>
            <div class="stat-grid">
                <div class="stat-cell">
                    <div class="stat-label">Category</div>
                    <div class="stat-value">{cat_label}</div>
                </div>
                <div class="stat-cell">
                    <div class="stat-label">Calories</div>
                    <div class="stat-value">{kcal:.0f} kcal</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            with st.expander("Reference details", expanded=False):
                st.markdown(f"**Basis:** {basis}")
                st.markdown(f"**Source:** {source}")
                st.markdown(
                    "**Thresholds:** Low < 300 kcal · Medium 300–399 kcal · High ≥ 400 kcal "
                    "*(per serve, approximate)*"
                )

        else:
            st.markdown(
                f'<div class="warn-box">No nutrition record found for '
                f'<strong>{cls_name}</strong> in the lookup table.</div>',
                unsafe_allow_html=True,
            )

        st.markdown("""
        <div class="disclaimer">
            <strong>Important:</strong> Calorie values are approximate reference estimates based on
            typical fast-food servings. The AI classifies the food type only — it cannot measure
            portion size, specific ingredients, or preparation method. Do not use this for medical
            or dietary decisions.
        </div>
        """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Top Predictions Card ──
        with st.expander("📊 All class probabilities", expanded=False):
            sorted_pairs = sorted(
                zip(class_names, all_probs), key=lambda x: x[1], reverse=True
            )
            for cls, prob in sorted_pairs:
                pct    = prob * 100
                bar_w  = f"{pct:.1f}%"
                bold   = "font-weight:600; color:#e8eaf0;" if cls == cls_name else ""
                st.markdown(f"""
                <div style="margin-bottom:8px;">
                    <div style="display:flex; justify-content:space-between;
                                font-size:0.85rem; {bold} margin-bottom:3px;">
                        <span>{cls}</span><span>{pct:.1f}%</span>
                    </div>
                    <div style="background:#2a2d40; border-radius:4px; height:6px; overflow:hidden;">
                        <div style="width:{bar_w}; height:6px;
                                    background:{'#6c63ff' if cls == cls_name else '#3a3d52'};
                                    border-radius:4px;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding:24px 0 8px 0;
            color:#3a3d52; font-size:0.78rem; border-top:1px solid #1a1d2e;">
    FoodSense AI &nbsp;·&nbsp; Food classification model trained on the Fast Food Classification V2 dataset
    &nbsp;·&nbsp; Reference calories from Nutrition_Value_Dataset.csv and USDA FoodData Central
</div>
""", unsafe_allow_html=True)

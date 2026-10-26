"""
config.py — Centralized Configuration for FoodSense AI
======================================================
Centralized configuration values for model paths, data paths,
image dimensions, confidence thresholds, and nutritional categories.
"""

from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Fast Food Classification V2"
MODEL_PATH = ROOT / "food_classifier.keras"
CLASS_PATH = ROOT / "class_names.json"
CALORIES_PATH = ROOT / "calories.csv"
OUTPUTS_DIR = ROOT / "outputs"
CONFIG_PATH = ROOT / "model_config.json"

# All 10 Supported Classes (Alphabetical, matches tf.data directory ordering)
ALL_CLASSES = [
    "Baked Potato",
    "Burger",
    "Crispy Chicken",
    "Donut",
    "Fries",
    "Hot Dog",
    "Pizza",
    "Sandwich",
    "Taco",
    "Taquito",
]
NUM_CLASSES = len(ALL_CLASSES)

# Model & Preprocessing settings
DEFAULT_IMAGE_SIZE = (160, 160)
CONFIDENCE_THRESHOLD = 0.55  # Below this, display low-confidence caution

# Nutrition categorization (kcal / serving)
CALORIE_LOW_MAX = 300
CALORIE_HIGH_MIN = 400

# UI Settings
SUPPORTED_FILE_TYPES = ["jpg", "jpeg", "png"]

"""
fast_train.py — High-Speed Feature Extraction & Transfer Benchmark
==================================================================
Extracts ImageNet features once, trains classifiers in seconds,
and exports the best end-to-end model to food_classifier.keras.
"""

from __future__ import annotations
import json
import random
import sys
import time
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from dataset_utils import tensorflow_compatible_path

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Fast Food Classification V2"
OUTPUTS_DIR = ROOT / "outputs"
MODEL_PATH = ROOT / "food_classifier.keras"
CLASS_PATH = ROOT / "class_names.json"
OUTPUTS_DIR.mkdir(exist_ok=True)

ALL_CLASSES = [
    "Baked Potato", "Burger", "Crispy Chicken", "Donut", "Fries",
    "Hot Dog", "Pizza", "Sandwich", "Taco", "Taquito"
]
NUM_CLASSES = len(ALL_CLASSES)
IMAGE_SIZE = (160, 160)
BATCH_SIZE = 64
TRAIN_CAP: int | None = None  # Use every image in the training split.

print(f"Loading dataset from: {DATASET_ROOT}")
print(f"Classes ({NUM_CLASSES}): {ALL_CLASSES}")

def get_paths_and_labels(split_name: str, max_per_class: int | None = None):
    paths, labels = [], []
    rng = random.Random(SEED)
    for idx, cls_name in enumerate(ALL_CLASSES):
        cls_dir = DATASET_ROOT / split_name / cls_name
        cls_paths = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if split_name == "Train" and max_per_class and len(cls_paths) > max_per_class:
            cls_paths = sorted(rng.sample(cls_paths, max_per_class))
        paths.extend(
            str(tensorflow_compatible_path(p, OUTPUTS_DIR / "converted_images"))
            for p in cls_paths
        )
        labels.extend([idx] * len(cls_paths))
    return paths, np.array(labels, dtype=np.int32)

def build_ds(paths, labels):
    h, w = IMAGE_SIZE
    def _load(path_t, label_t):
        raw = tf.io.read_file(path_t)
        img = tf.io.decode_image(raw, channels=3, expand_animations=False)
        img.set_shape([None, None, 3])
        img = tf.image.resize(img, (h, w))
        img = tf.cast(img, tf.float32) / 255.0
        return img, label_t

    ds = tf.data.Dataset.from_tensor_slices((np.asarray(paths), labels))
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

train_paths, y_train = get_paths_and_labels("Train", max_per_class=TRAIN_CAP)
val_paths, y_val = get_paths_and_labels("Valid")
test_paths, y_test = get_paths_and_labels("Test")

print(
    f"Dataset sizes -> Train: {len(train_paths)}, Valid: {len(val_paths)}, "
    f"Test: {len(test_paths)}"
)

train_ds = build_ds(train_paths, y_train)
val_ds = build_ds(val_paths, y_val)
test_ds = build_ds(test_paths, y_test)

# Build feature extractor base
print("\nInitializing MobileNetV2 feature extractor...")
base_mobilenet = tf.keras.applications.MobileNetV2(
    input_shape=(*IMAGE_SIZE, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base_mobilenet.trainable = False

# Rescaling: [0, 1] -> [-1, 1]
fe_input = tf.keras.Input(shape=(*IMAGE_SIZE, 3))
x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0)(fe_input)
x = base_mobilenet(x, training=False)
fe_model = tf.keras.Model(fe_input, x)

print("Extracting features (one-time pass)...")
t0 = time.time()
feat_train = fe_model.predict(train_ds, verbose=1)
feat_val = fe_model.predict(val_ds, verbose=1)
ext_time = time.time() - t0
print(f"Feature extraction done in {ext_time:.1f}s!")
print(f"Feature shapes -> Train: {feat_train.shape}, Val: {feat_val.shape}")

# Model 1: Deep MLP Head
print("\n--- Training Deep MLP Head on Features ---")
mlp_head = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1280,)),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(512, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(256, activation="relu"),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
])
mlp_head.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

cbs = [
    tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=8, restore_best_weights=True, mode="max"),
    tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5)
]
t1 = time.time()
mlp_head.fit(
    feat_train, y_train,
    validation_data=(feat_val, y_val),
    epochs=50,
    batch_size=64,
    callbacks=cbs,
    verbose=1
)
mlp_time = time.time() - t1
mlp_val_pred = np.argmax(mlp_head.predict(feat_val, verbose=0), axis=1)
mlp_val_acc = float(np.mean(mlp_val_pred == y_val))
print(f"MLP Head -> Val Acc: {mlp_val_acc:.4f} (trained in {mlp_time:.1f}s)")

# Model 2: Logistic Regression (L2 regularized) on Features
print("\n--- Training Logistic Regression on Features ---")
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(feat_train)
X_val_sc = scaler.transform(feat_val)

lr_clf = LogisticRegression(C=0.1, max_iter=500, random_state=SEED)
t2 = time.time()
lr_clf.fit(X_tr_sc, y_train)
lr_time = time.time() - t2
lr_val_acc = lr_clf.score(X_val_sc, y_val)
print(f"Logistic Regression -> Val Acc: {lr_val_acc:.4f} (trained in {lr_time:.1f}s)")

# Compare and select using validation accuracy only. The test set remains held out.
if mlp_val_acc >= lr_val_acc:
    best_name = "MobileNetV2 + Deep MLP Head"
    selected_head = mlp_head
else:
    best_name = "MobileNetV2 + Linear Head"
    # Convert LogisticRegression to a single Dense layer in Keras for end-to-end inference
    w = lr_clf.coef_ / scaler.scale_
    b = lr_clf.intercept_ - np.dot(lr_clf.coef_, scaler.mean_ / scaler.scale_)
    dense_layer = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")
    dense_layer.build((None, 1280))
    dense_layer.set_weights([w.T, b])
    selected_head = tf.keras.Sequential([dense_layer])

print(f"\nSelected architecture by validation accuracy: {best_name}")

# Construct Full End-to-End Model
print("\nConstructing end-to-end Keras model...")
full_inputs = tf.keras.Input(shape=(*IMAGE_SIZE, 3), name="input_image")
scaled = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_rescale")(full_inputs)
features = base_mobilenet(scaled, training=False)
full_outputs = selected_head(features)
full_model = tf.keras.Model(full_inputs, full_outputs, name="FoodSense_MobileNetV2")

full_model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
test_probabilities = full_model.predict(test_ds, verbose=1)
best_y_pred = np.argmax(test_probabilities, axis=1)
best_test_acc = float(np.mean(best_y_pred == y_test))
full_model.save(str(MODEL_PATH))
CLASS_PATH.write_text(json.dumps(ALL_CLASSES, indent=2), encoding="utf-8")
(ROOT / "model_config.json").write_text(
    json.dumps(
        {
            "image_size": list(IMAGE_SIZE),
            "confidence_warn_threshold": 0.55,
            "model_name": best_name,
            "validation_accuracy": max(mlp_val_acc, lr_val_acc),
            "test_accuracy": best_test_acc,
        },
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(f"[OK] Full model saved: {MODEL_PATH}")
print(f"[OK] Class names saved: {CLASS_PATH}")
print(f"Final held-out test accuracy: {best_test_acc:.4f}")

# Evaluation report and confusion matrix
report_str = classification_report(y_test, best_y_pred, target_names=ALL_CLASSES, digits=3, zero_division=0)
(OUTPUTS_DIR / "classification_report.txt").write_text(report_str, encoding="utf-8")
print("\nClassification Report (Test Set):")
print(report_str)

# Confusion Matrix
cm = confusion_matrix(y_test, best_y_pred, labels=np.arange(NUM_CLASSES))
fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cm, cmap="Blues")
ax.set_title(f"{best_name} — Confusion Matrix (Test Set: {best_test_acc*100:.1f}%)")
ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
ax.set_xticklabels(ALL_CLASSES, rotation=45, ha="right", fontsize=9)
ax.set_yticklabels(ALL_CLASSES, fontsize=9)
ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
fig.colorbar(im, ax=ax)
thresh = cm.max() / 2.0
for r in range(cm.shape[0]):
    for c in range(cm.shape[1]):
        ax.text(c, r, str(cm[r, c]), ha="center", va="center",
                color="white" if cm[r, c] > thresh else "black", fontsize=8)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
plt.close(fig)

# Save Comparison Table
comp_lines = [
    "MODEL COMPARISON\n",
    f"{'Model':<35} {'Val Acc':>10} {'Selected Test Acc':>18} {'Train Time':>12}\n",
    "-" * 70 + "\n",
    f"{'MobileNetV2 + Deep MLP Head':<35} {f'{mlp_val_acc*100:.1f}%':>10} "
    f"{f'{best_test_acc*100:.1f}%' if selected_head is mlp_head else '—':>18} "
    f"{f'{mlp_time:.1f}s':>12}\n",
    f"{'MobileNetV2 + Linear Head':<35} {f'{lr_val_acc*100:.1f}%':>10} "
    f"{f'{best_test_acc*100:.1f}%' if selected_head is not mlp_head else '—':>18} "
    f"{f'{lr_time:.1f}s':>12}\n",
    "\n",
    f"Selected Model : {best_name}\n",
    f"Final test accuracy (selected model only): {best_test_acc*100:.2f}%\n"
]
(OUTPUTS_DIR / "model_comparison.txt").write_text("".join(comp_lines), encoding="utf-8")
print("\n" + "".join(comp_lines))

# Sanity Check Reload
print("\nSanity Check: reloading saved model...")
test_reload = tf.keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=True)
print(f"Reload successful! Output shape: {test_reload.output_shape}")
for images, labels in test_ds.take(1):
    preds = test_reload.predict(images[:5], verbose=0)
    for i in range(5):
        act = ALL_CLASSES[labels[i]]
        pred_cls = ALL_CLASSES[np.argmax(preds[i])]
        conf = np.max(preds[i]) * 100
        mark = "[CORRECT]" if act == pred_cls else "[WRONG]"
        print(f"  {mark} Actual: {act:<15} Predicted: {pred_cls:<15} ({conf:.1f}%)")
print("\nALL TASKS COMPLETED SUCCESSFULLY!")

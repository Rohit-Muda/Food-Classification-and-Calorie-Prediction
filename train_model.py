"""
train_model.py — Food Image Classification Training Pipeline
=============================================================
CPU-optimised training pipeline that trains and compares:
  A. Improved Custom CNN (96×96, all 10 classes, full dataset)
  B. MobileNetV2 — head-only training + fine-tune top 40 layers
  C. EfficientNetB0 — head-only training + fine-tune top 30 layers

Hardware strategy (CPU):
  - Streams decoded images without caching full resized splits in RAM
  - Converts uncommon source formats to a TensorFlow-readable PNG cache
  - Smaller batch = more frequent gradient updates with lower RAM use
  - Early stopping with low patience avoids wasted epochs

Run:
    python train_model.py
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
from inference import export_saved_model

# ─────────────────────────────────────────────
#  REPRODUCIBILITY
# ─────────────────────────────────────────────
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Fast Food Classification V2"
MODEL_PATH   = ROOT / "food_classifier.keras"
SAVED_MODEL_PATH = ROOT / "food_classifier_saved_model"
CLASS_PATH   = ROOT / "class_names.json"
OUTPUTS_DIR  = ROOT / "outputs"
IMAGE_EXTS   = {".jpg", ".jpeg", ".png"}

# ─────────────────────────────────────────────
#  CLASSES  (alphabetical = tf.data directory order)
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
#  CPU-OPTIMISED SETTINGS
# ─────────────────────────────────────────────
CNN_IMAGE_SIZE      = (96, 96)    # smaller than 128 → faster, still informative
TRANSFER_IMAGE_SIZE = (160, 160)  # smaller than 224 → acceptable for MobileNetV2
BATCH_SIZE          = 32
# Use the complete training split in normal runs. Set a positive cap only for
# deliberate smoke runs.
TRAIN_CAP: int | None = None

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────

def _collect_split(
    split_name: str,
    image_size: tuple[int, int],
    max_per_class: int | None = None,
) -> tf.data.Dataset:
    """
    Build a tf.data.Dataset for a given split.
    Images are normalised to [0, 1] float32.
    If max_per_class is set, randomly sample that many images per class
    (reproducible with SEED).
    """
    paths, labels = [], []
    rng = random.Random(SEED)

    for idx, cls_name in enumerate(ALL_CLASSES):
        cls_dir = DATASET_ROOT / split_name / cls_name
        if not cls_dir.is_dir():
            raise FileNotFoundError(f"Missing: {cls_dir}")

        cls_paths = sorted(
            p for p in cls_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
        if not cls_paths:
            raise FileNotFoundError(f"No images in {cls_dir}")

        if split_name == "Train" and max_per_class and len(cls_paths) > max_per_class:
            cls_paths = sorted(rng.sample(cls_paths, max_per_class))

        paths.extend(
            str(tensorflow_compatible_path(p, OUTPUTS_DIR / "converted_images"))
            for p in cls_paths
        )
        labels.extend([idx] * len(cls_paths))
        print(f"  {split_name}/{cls_name}: {len(cls_paths)} images (idx={idx})")

    h, w = image_size

    def _load(path_tensor: tf.Tensor, label: tf.Tensor):
        raw   = tf.io.read_file(path_tensor)
        image = tf.io.decode_image(raw, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, (h, w))
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    ds = tf.data.Dataset.from_tensor_slices(
        (np.asarray(paths), np.asarray(labels, dtype=np.int32))
    )
    if split_name == "Train":
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def add_augmentation(ds: tf.data.Dataset) -> tf.data.Dataset:
    """Apply food-safe augmentation on batched dataset."""
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.10),
        tf.keras.layers.RandomZoom(0.10),
        tf.keras.layers.RandomBrightness(0.10),
        tf.keras.layers.RandomContrast(0.10),
    ])
    return ds.map(
        lambda imgs, lbls: (aug(imgs, training=True), lbls),
        num_parallel_calls=tf.data.AUTOTUNE,
    )


# ─────────────────────────────────────────────
#  MODEL BUILDERS
# ─────────────────────────────────────────────

def build_cnn(num_classes: int) -> tf.keras.Model:
    """Improved CNN baseline (96×96 input)."""
    inputs = tf.keras.Input(shape=(*CNN_IMAGE_SIZE, 3))
    x = inputs
    for filters in (32, 64, 128, 256):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Activation("relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.4)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="ImprovedCNN")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_mobilenet(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    MobileNetV2 transfer learning.
    Input: (160, 160, 3) float32 in [0,1].
    Model rescales to [0,255] then applies preprocess_input internally.
    """
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*TRANSFER_IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*TRANSFER_IMAGE_SIZE, 3))
    # Native Rescaling maps [0, 1] to [-1, 1]: (2.0 * x) - 1.0 (exact MobileNetV2 format)
    x = tf.keras.layers.Rescaling(scale=2.0, offset=-1.0, name="mobilenet_rescale")(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="MobileNetV2")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def build_efficientnet(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """
    EfficientNetB0 transfer learning.
    EfficientNet has built-in preprocessing; accepts [0, 255].
    Input: (160, 160, 3) float32 in [0,1] → rescaled to [0,255] inside model.
    """
    base = tf.keras.applications.EfficientNetB0(
        input_shape=(*TRANSFER_IMAGE_SIZE, 3),
        include_top=False,
        weights="imagenet",
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(*TRANSFER_IMAGE_SIZE, 3))
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs, name="EfficientNetB0")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


# ─────────────────────────────────────────────
#  CALLBACKS
# ─────────────────────────────────────────────

def make_callbacks(ckpt_path: str, patience_es: int = 4, patience_lr: int = 2) -> list:
    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path,
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=0,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=patience_es,
            restore_best_weights=True,
            mode="max",
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-6,
            verbose=0,
        ),
    ]


# ─────────────────────────────────────────────
#  EVALUATION HELPERS
# ─────────────────────────────────────────────

def predict_all(model: tf.keras.Model, ds: tf.data.Dataset) -> tuple[np.ndarray, np.ndarray]:
    y_true_list, y_pred_list = [], []
    for images, labels in ds:
        preds = model.predict(images, verbose=0)
        y_true_list.append(labels.numpy())
        y_pred_list.append(np.argmax(preds, axis=1))
    return np.concatenate(y_true_list), np.concatenate(y_pred_list)


def save_history_plot(history_dict: dict, title: str, save_path: Path) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history_dict.get("accuracy", []),     label="Train Acc")
    axes[0].plot(history_dict.get("val_accuracy", []), label="Val Acc")
    axes[0].set_title(f"{title} — Accuracy"); axes[0].set_xlabel("Epoch"); axes[0].legend()
    axes[1].plot(history_dict.get("loss", []),     label="Train Loss")
    axes[1].plot(history_dict.get("val_loss", []), label="Val Loss")
    axes[1].set_title(f"{title} — Loss"); axes[1].set_xlabel("Epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: Path) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(NUM_CLASSES))
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(NUM_CLASSES)); ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(ALL_CLASSES, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ALL_CLASSES, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    thresh = matrix.max() / 2.0
    for r in range(matrix.shape[0]):
        for c in range(matrix.shape[1]):
            ax.text(c, r, str(matrix[r, c]), ha="center", va="center",
                    color="white" if matrix[r, c] > thresh else "black", fontsize=8)
    fig.tight_layout(); fig.savefig(save_path, dpi=150); plt.close(fig)


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> dict:
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    loss, accuracy = model.evaluate(dataset, verbose=0)
    y_true, y_pred = predict_all(model, dataset)
    report = classification_report(
        y_true, y_pred,
        labels=np.arange(NUM_CLASSES),
        target_names=ALL_CLASSES,
        digits=3, zero_division=0, output_dict=True,
    )
    return {
        "loss": loss,
        "accuracy": accuracy,
        "macro_f1": report["macro avg"]["f1-score"],
        "y_true":        y_true,
        "y_pred":        y_pred,
        "report":        report,
    }


# ─────────────────────────────────────────────
#  EXPERIMENT RUNNERS
# ─────────────────────────────────────────────

def run_cnn(train_ds, val_ds) -> dict:
    print("\n" + "=" * 60)
    print("MODEL A — Improved Custom CNN (96×96)")
    print("=" * 60)
    ckpt_path = OUTPUTS_DIR / "cnn_best.keras"
    OUTPUTS_DIR.mkdir(exist_ok=True)
    model = build_cnn(NUM_CLASSES)
    cbs = make_callbacks(str(ckpt_path), patience_es=5, patience_lr=2)
    t0 = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=30, callbacks=cbs, verbose=1)
    elapsed = time.time() - t0
    if ckpt_path.exists():
        model = tf.keras.models.load_model(str(ckpt_path), compile=False)
    save_history_plot(history.history, "Improved CNN", OUTPUTS_DIR / "cnn_history.png")
    metrics = evaluate_model(model, val_ds)
    metrics.update({"training_time": elapsed, "model": model, "name": "Improved CNN"})
    print(f"  Validation Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Training time : {elapsed/60:.1f} min")
    return metrics


def run_mobilenet(train_ds, val_ds) -> dict:
    print("\n" + "=" * 60)
    print("MODEL B — MobileNetV2 Transfer Learning (160×160)")
    print("=" * 60)
    model, base = build_mobilenet(NUM_CLASSES)
    ckpt_path   = str(OUTPUTS_DIR / "mobilenet_best.keras")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Phase 1: head only
    print("  Phase 1: head training (base frozen) …")
    t0      = time.time()
    cbs_p1  = make_callbacks(ckpt_path, patience_es=3, patience_lr=2)
    hist_p1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cbs_p1, verbose=1)

    if Path(ckpt_path).exists():
        model = tf.keras.models.load_model(ckpt_path, compile=False, safe_mode=True)
        # Re-get base for fine-tuning
        try:
            base = model.get_layer("mobilenetv2_1.00_160")
        except ValueError:
            # Fallback to standard name if size differs
            for l in model.layers:
                if "mobilenetv2" in l.name.lower():
                    base = l
                    break

    # Phase 2: fine-tune top 40 layers
    print("  Phase 2: fine-tuning top 40 layers …")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs_p2  = make_callbacks(ckpt_path, patience_es=3, patience_lr=2)
    hist_p2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cbs_p2, verbose=1)
    elapsed = time.time() - t0

    if Path(ckpt_path).exists():
        model = tf.keras.models.load_model(ckpt_path, compile=False, safe_mode=True)

    combined = {k: hist_p1.history.get(k, []) + hist_p2.history.get(k, [])
                for k in ["accuracy", "val_accuracy", "loss", "val_loss"]}
    save_history_plot(combined, "MobileNetV2", OUTPUTS_DIR / "mobilenet_history.png")
    metrics = evaluate_model(model, val_ds)
    metrics.update({"training_time": elapsed, "model": model, "name": "MobileNetV2"})
    print(f"  Validation Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Training time : {elapsed/60:.1f} min")
    return metrics


def run_efficientnet(train_ds, val_ds) -> dict:
    print("\n" + "=" * 60)
    print("MODEL C — EfficientNetB0 Transfer Learning (160×160)")
    print("=" * 60)
    model, base = build_efficientnet(NUM_CLASSES)
    ckpt_path   = str(OUTPUTS_DIR / "efficientnet_best.keras")
    OUTPUTS_DIR.mkdir(exist_ok=True)

    # Phase 1: head only
    print("  Phase 1: head training (base frozen) …")
    t0      = time.time()
    cbs_p1  = make_callbacks(ckpt_path, patience_es=3, patience_lr=2)
    hist_p1 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cbs_p1, verbose=1)

    if Path(ckpt_path).exists():
        model = tf.keras.models.load_model(ckpt_path, compile=False, safe_mode=True)
        try:
            base = model.get_layer("efficientnetb0")
        except ValueError:
            for l in model.layers:
                if "efficientnet" in l.name.lower():
                    base = l
                    break

    # Phase 2: fine-tune top 30 layers
    print("  Phase 2: fine-tuning top 30 layers …")
    base.trainable = True
    for layer in base.layers[:-30]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(5e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    cbs_p2  = make_callbacks(ckpt_path, patience_es=3, patience_lr=2)
    hist_p2 = model.fit(train_ds, validation_data=val_ds, epochs=10, callbacks=cbs_p2, verbose=1)
    elapsed = time.time() - t0

    if Path(ckpt_path).exists():
        model = tf.keras.models.load_model(ckpt_path, compile=False, safe_mode=True)

    combined = {k: hist_p1.history.get(k, []) + hist_p2.history.get(k, [])
                for k in ["accuracy", "val_accuracy", "loss", "val_loss"]}
    save_history_plot(combined, "EfficientNetB0", OUTPUTS_DIR / "efficientnet_history.png")
    metrics = evaluate_model(model, val_ds)
    metrics.update({"training_time": elapsed, "model": model, "name": "EfficientNetB0"})
    print(f"  Validation Accuracy: {metrics['accuracy']:.4f}  |  "
          f"Macro F1: {metrics['macro_f1']:.4f}")
    print(f"  Training time : {elapsed/60:.1f} min")
    return metrics



# ─────────────────────────────────────────────
#  RESULTS REPORTING
# ─────────────────────────────────────────────

def save_final_report(test_metrics: dict, best: dict, all_results: list[dict]) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    report_str = classification_report(
        test_metrics["y_true"], test_metrics["y_pred"],
        labels=np.arange(NUM_CLASSES),
        target_names=ALL_CLASSES,
        digits=3, zero_division=0,
    )
    (OUTPUTS_DIR / "classification_report.txt").write_text(report_str, encoding="utf-8")
    print("\nFinal classification report (test set):")
    print(report_str)
    save_confusion_matrix(test_metrics["y_true"], test_metrics["y_pred"],
                          f"{best['name']} — Confusion Matrix (Test Set)",
                          OUTPUTS_DIR / "confusion_matrix.png")
    lines = ["MODEL COMPARISON\n",
             f"{'Model':<20} {'Val Acc':>10} {'Val Macro F1':>14} {'Time(min)':>10}\n"]
    for r in all_results:
        lines.append(f"{r['name']:<20} {r['accuracy']:>10.4f} "
                     f"{r['macro_f1']:>14.4f} {r['training_time']/60:>10.1f}\n")
    lines.append(f"\nSelected model (validation): {best['name']}\n")
    lines.append(f"Selected validation accuracy: {best['accuracy']:.4f}\n")
    lines.append(f"Final test accuracy         : {test_metrics['accuracy']:.4f}\n")
    lines.append(f"Final test macro F1          : {test_metrics['macro_f1']:.4f}\n")
    (OUTPUTS_DIR / "model_comparison.txt").write_text("".join(lines), encoding="utf-8")


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main() -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print(f"Dataset: {DATASET_ROOT}")
    print(f"Classes: {NUM_CLASSES} ({', '.join(ALL_CLASSES)})")
    print(f"Train cap per class: {TRAIN_CAP or 'none (full training split)'}")
    print(f"CNN image size     : {CNN_IMAGE_SIZE}")
    print(f"Transfer img size  : {TRANSFER_IMAGE_SIZE}")
    print(f"Batch size         : {BATCH_SIZE}")
    print("=" * 60)

    # ── Build datasets ──
    print("\n[Building CNN datasets (96×96)]")
    train_sm = add_augmentation(
        _collect_split("Train", CNN_IMAGE_SIZE, max_per_class=TRAIN_CAP)
    )
    val_sm   = _collect_split("Valid", CNN_IMAGE_SIZE)
    test_sm  = _collect_split("Test",  CNN_IMAGE_SIZE)

    print("\n[Building transfer-learning datasets (160×160)]")
    train_lg = add_augmentation(
        _collect_split("Train", TRANSFER_IMAGE_SIZE, max_per_class=TRAIN_CAP)
    )
    val_lg   = _collect_split("Valid", TRANSFER_IMAGE_SIZE)
    test_lg  = _collect_split("Test",  TRANSFER_IMAGE_SIZE)

    all_results: list[dict] = []

    cnn_m = run_cnn(train_sm, val_sm)
    all_results.append(cnn_m)

    mob_m = run_mobilenet(train_lg, val_lg)
    all_results.append(mob_m)

    eff_m = run_efficientnet(train_lg, val_lg)
    all_results.append(eff_m)

    # Select using validation metrics only; reserve the test split for one final estimate.
    best = max(
        all_results,
        key=lambda result: (
            result["accuracy"],
            result["macro_f1"],
        ),
    )
    final_test_ds = test_sm if best["name"] == "Improved CNN" else test_lg
    final_test = evaluate_model(best["model"], final_test_ds)
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<20} {'Val Acc':>10} {'Val Macro F1':>14} {'Time(min)':>10}")
    print("-" * 55)
    for r in all_results:
        star = " <- SELECTED" if r is best else ""
        print(f"{r['name']:<20} {r['accuracy']:>10.4f} "
              f"{r['macro_f1']:>14.4f} {r['training_time']/60:>10.1f}{star}")
    print(f"Final test accuracy: {final_test['accuracy']:.4f}")
    print("=" * 70)

    # ── Save best model ──
    best["model"].save(str(MODEL_PATH))
    export_saved_model(best["model"], SAVED_MODEL_PATH)
    CLASS_PATH.write_text(json.dumps(ALL_CLASSES, indent=2), encoding="utf-8")
    (ROOT / "model_config.json").write_text(
        json.dumps(
            {
                "image_size": list(best["model"].input_shape[1:3]),
                "confidence_warn_threshold": 0.55,
                "model_name": best["name"],
                "validation_accuracy": best["accuracy"],
                "test_accuracy": final_test["accuracy"],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"\n[OK] Best model saved   : {MODEL_PATH}")
    print(f"[OK] Class names saved  : {CLASS_PATH}")
    save_final_report(final_test, best, all_results)

    # ── Sanity check ──
    print("\nSanity check -- reloading model ...")
    reloaded = tf.keras.models.load_model(str(MODEL_PATH), compile=False, safe_mode=True)
    classes  = json.loads(CLASS_PATH.read_text(encoding="utf-8"))
    assert len(classes) == reloaded.output_shape[-1], "Class count mismatch after reload!"
    print(f"  Output shape : {reloaded.output_shape}")
    print(f"  Num classes  : {len(classes)}")

    for images, labels in final_test_ds.take(1):
        preds = reloaded.predict(images[:5], verbose=0)
        print("  Sample predictions:")
        for i in range(min(5, len(labels))):
            actual    = classes[int(labels[i])]
            predicted = classes[int(np.argmax(preds[i]))]
            conf      = float(preds[i].max()) * 100
            mark      = "[CORRECT]" if actual == predicted else "[WRONG]"
            print(f"    {mark} actual={actual:<16}  predicted={predicted:<16}  conf={conf:.1f}%")

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print(f"  Best model   : {best['name']}")
    print(f"  Test accuracy: {final_test['accuracy']:.4f} "
          f"({final_test['accuracy']*100:.1f}%)")
    print(f"  Test macro F1: {final_test['macro_f1']:.4f}")
    if final_test["accuracy"] >= 0.90:
        print("  [OK] 90% accuracy target achieved!")
    else:
        remaining = 0.90 - final_test["accuracy"]
        print(f"  [--] Below 90% target by {remaining:.4f} ({remaining*100:.1f}%)")
        print("    -> Improve augmentation, training duration, or model/input resolution")
    print("=" * 60)



if __name__ == "__main__":
    main()

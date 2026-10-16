from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

SEED = 42
IMAGE_SIZE = (128, 128)
BATCH_SIZE = 32
TRAIN_IMAGES_PER_CLASS = 500
SELECTED_CLASSES = ["Burger", "Fries", "Pizza", "Donut", "Sandwich"]
ROOT = Path(__file__).resolve().parent
DATASET_ROOT = ROOT / "Fast Food Classification V2"
MODEL_PATH = ROOT / "food_classifier.keras"
CLASS_PATH = ROOT / "class_names.json"
OUTPUTS_DIR = ROOT / "outputs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def set_seed() -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    tf.keras.utils.set_random_seed(SEED)


def get_split_paths(split_name: str) -> tuple[np.ndarray, np.ndarray]:
    paths = []
    labels = []
    for class_index, class_name in enumerate(SELECTED_CLASSES):
        class_dir = DATASET_ROOT / split_name / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")

        class_paths = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not class_paths:
            raise FileNotFoundError(f"No supported images found in {class_dir}")
        if split_name == "Train" and len(class_paths) > TRAIN_IMAGES_PER_CLASS:
            class_rng = random.Random(SEED + class_index)
            class_paths = sorted(class_rng.sample(class_paths, TRAIN_IMAGES_PER_CLASS))

        paths.extend(str(path) for path in class_paths)
        labels.extend([class_index] * len(class_paths))
        print(f"{split_name}/{class_name}: {len(class_paths)} images")

    return np.asarray(paths), np.asarray(labels, dtype=np.int32)


def build_dataset(split_name: str, *, augment: bool = False) -> tf.data.Dataset:
    paths, labels = get_split_paths(split_name)
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))

    def load_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        image_bytes = tf.io.read_file(path)
        image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.resize(image, IMAGE_SIZE)
        image = tf.cast(image, tf.float32) / 255.0
        return image, label

    if split_name == "Train":
        dataset = dataset.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    dataset = dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        augmentation = tf.keras.Sequential(
            [
                tf.keras.layers.RandomFlip("horizontal"),
                tf.keras.layers.RandomRotation(0.08),
            ]
        )
        dataset = dataset.map(
            lambda images, labels: (augmentation(images, training=True), labels),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    return dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


def build_model() -> tf.keras.Model:
    inputs = tf.keras.layers.Input(shape=(*IMAGE_SIZE, 3))
    x = inputs
    for filters in (32, 64, 128):
        x = tf.keras.layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = tf.keras.layers.MaxPooling2D()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.5)(x)
    outputs = tf.keras.layers.Dense(len(SELECTED_CLASSES), activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_history(history: tf.keras.callbacks.History) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(history.history["accuracy"], label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Accuracy")
    axes[0].legend()
    axes[1].plot(history.history["loss"], label="Train")
    axes[1].plot(history.history["val_loss"], label="Validation")
    axes[1].set_title("Loss")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "training_history.png", dpi=150)
    plt.close(fig)


def save_evaluation(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    OUTPUTS_DIR.mkdir(exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(SELECTED_CLASSES)))
    fig, ax = plt.subplots(figsize=(7, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xticks(range(len(SELECTED_CLASSES)))
    ax.set_yticks(range(len(SELECTED_CLASSES)))
    ax.set_xticklabels(SELECTED_CLASSES, rotation=45, ha="right")
    ax.set_yticklabels(SELECTED_CLASSES)
    fig.colorbar(image, ax=ax)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            ax.text(col, row, matrix[row, col], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(len(SELECTED_CLASSES)),
        target_names=SELECTED_CLASSES,
        digits=3,
        zero_division=0,
    )
    (OUTPUTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")
    print(report)


def main() -> None:
    set_seed()
    train_ds = build_dataset("Train", augment=True)
    val_ds = build_dataset("Valid")
    test_ds = build_dataset("Test")

    model = build_model()
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-5,
        ),
    ]
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=10,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(str(MODEL_PATH))
    CLASS_PATH.write_text(json.dumps(SELECTED_CLASSES, indent=2), encoding="utf-8")

    test_loss, test_accuracy = model.evaluate(test_ds, verbose=0)
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    y_true = np.concatenate([labels.numpy() for _, labels in test_ds])
    y_pred = np.argmax(model.predict(test_ds, verbose=0), axis=1)
    save_history(history)
    save_evaluation(y_true, y_pred)

    reloaded_model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)
    sample_images, sample_labels = next(iter(test_ds))
    prediction = reloaded_model.predict(sample_images[:1], verbose=0)[0]
    actual_class = SELECTED_CLASSES[int(sample_labels[0])]
    predicted_class = SELECTED_CLASSES[int(np.argmax(prediction))]
    print(f"Reloaded model sample: actual={actual_class}, predicted={predicted_class}")


if __name__ == "__main__":
    main()

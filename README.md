# FoodSense AI — Food Image Classification and Calorie Reference

A production-ready deep learning application that classifies food images across **10 fast-food categories** and provides nutritional reference information. Built with TensorFlow/Keras and Streamlit.

---

## 1. Supported Food Categories & Reference Nutrition

| Category | Reference Calories | Category | Data Source |
|:---|:---:|:---:|:---|
| **Baked Potato** | 252 kcal / serve | Low | USDA FoodData Central (plain baked potato, medium) |
| **Burger** | 485 kcal / serve | High | `Nutrition_Value_Dataset.csv` (representative burgers) |
| **Crispy Chicken** | 320 kcal / serve | Medium | Fast-food crispy chicken sandwich reference |
| **Donut** | 300 kcal / serve | Medium | USDA FoodData Central (glazed donut, medium) |
| **Fries** | 374 kcal / serve | Medium | `Nutrition_Value_Dataset.csv` (representative fries) |
| **Hot Dog** | 290 kcal / serve | Medium | USDA FoodData Central (beef hot dog with bun) |
| **Pizza** | 411 kcal / serve | High | `Nutrition_Value_Dataset.csv` (personal pizza items) |
| **Sandwich** | 365 kcal / serve | Medium | `Nutrition_Value_Dataset.csv` (breakfast sandwiches) |
| **Taco** | 210 kcal / serve | Low | USDA FoodData Central (beef taco, fast-food style) |
| **Taquito** | 180 kcal / serve | Low | USDA FoodData Central (beef taquito, rolled) |

> **Calorie Thresholds:**
> - **Low**: < 300 kcal per serve
> - **Medium**: 300 – 399 kcal per serve
> - **High**: ≥ 400 kcal per serve

---

## 2. Root Cause Diagnosis of the Original Model

The original project had severe accuracy and reliability issues (failing even on clear food images):

1. **Only 5 of 10 Classes Used**: The dataset provided 10 classes, but the original model only trained on 5 (`Burger`, `Fries`, `Pizza`, `Donut`, `Sandwich`), completely failing when presented with the other 5 fast-food categories (`Baked Potato`, `Crispy Chicken`, `Hot Dog`, `Taco`, `Taquito`).
2. **Artificial Training Data Capping**: The original code artificially capped training at 500 images per class, discarding 66% of the available training data (1,500 images per class).
3. **Severe Model Undercapacity**: A tiny 3-block CNN trained from scratch on 128×128 inputs with small dense layers could not discriminate complex fast-food textures without millions of parameters.
4. **No Transfer Learning**: Food classification requires rich texture and color feature extractors. Pretrained weights on ImageNet provide an immediate feature-extraction advantage.
5. **Preprocessing & Input Dimension Mismatches**: Hardcoded dimensions and differing normalization approaches caused inference degradation.

---

## 3. Project Structure

```text
DL_Food_Classification/
├── app.py                        # Streamlit production application
├── config.py                     # Centralized configuration (paths, classes, thresholds)
├── model_config.json             # Runtime model metadata & resolution config
├── train_model.py                # Multi-model training & comparison pipeline
├── food_classifier.keras         # Best trained model (loaded by app.py)
├── class_names.json              # Class index -> label mapping (alphabetical order)
├── calories.csv                  # Nutrition reference lookup table (all 10 classes)
├── Nutrition_Value_Dataset.csv   # Source nutritional dataset
├── requirements.txt              # Cloud & local deployment dependencies
├── .gitignore                    # Excludes heavy dataset folders & temporary checkpoints
├── README.md                     # Comprehensive documentation & handoff guide
└── outputs/                      # Evaluation artifacts
    ├── cnn_best.keras            # Model A checkpoint
    ├── cnn_history.png           # Model A training/loss curve
    ├── mobilenet_best.keras      # Model B checkpoint (during/after training)
    ├── mobilenet_history.png     # Model B training/loss curve
    ├── efficientnet_best.keras   # Model C checkpoint (during/after training)
    ├── efficientnet_history.png  # Model C training/loss curve
    ├── classification_report.txt # Per-class precision, recall, and F1 report
    ├── confusion_matrix.png      # Test set confusion matrix
    └── model_comparison.txt      # Comparative metrics table
```

---

## 4. Multi-Model Architecture & Experimental Comparison

The pipeline in `train_model.py` trains and evaluates three distinct models under consistent train/val/test splits:

### Model A: Improved Custom CNN Baseline (96×96)
- 4 Convolutional blocks (32 → 64 → 128 → 256 filters) with Batch Normalization and ReLU
- GlobalAveragePooling2D + Dense(256) + Dense(128) with Dropout
- The earlier baseline run reached `9.33%` test accuracy (Macro F1: `0.0449`); it is retained only as a comparison baseline.

### Model B: MobileNetV2 Transfer Learning (160×160)
- Pretrained ImageNet weights (`include_top=False`)
- **Phase 1**: Frozen base feature extractor, training custom classification head (Dropout 0.3 + Dense 128 + Dropout 0.2 + Dense 10)
- **Phase 2**: Unfreezing top 40 layers for fine-tuning at learning rate `1e-4` with `ReduceLROnPlateau` and `EarlyStopping`
- Uses ImageNet preprocessing inside the model, matching the app's `[0, 1]` input contract.

### Model C: EfficientNetB0 Transfer Learning (160×160)
- Pretrained ImageNet weights (`include_top=False`)
- **Phase 1**: Head training on frozen base
- **Phase 2**: Unfreezing top 30 layers for fine-tuning at learning rate `5e-5`

---

## 5. Running the Application Locally

### Step 1: Activate Virtual Environment
```powershell
# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Launch Streamlit Interface
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

### Run the inference tests
```powershell
python -m unittest discover -s tests -v
```

---

## 6. How to Run Model Training

To run the complete benchmark pipeline:

```powershell
python train_model.py
```

### What `train_model.py` Does:
1. Uses the full training split (1,500 images per class); it validates source files and converts formats unsupported by TensorFlow into an ignored PNG cache without modifying the dataset.
2. Trains and fine-tunes the CNN, MobileNetV2, and EfficientNetB0 candidates.
3. Selects the best candidate using validation accuracy (with macro F1 as a tie-breaker), never by test performance.
4. Evaluates the selected model once on the held-out test split.
5. Saves the model, class order, model metadata, per-class test report, confusion matrix, and validation comparison table.

`fast_train.py` is an optional, faster MobileNetV2 feature-extraction benchmark. It compares its MLP and linear heads using validation accuracy, then reports test performance only for the selected head.

> Training the complete benchmark can take a long time on CPU. The checked-in model remains usable while training runs; a retrained model and its refreshed evaluation artifacts are written only after training completes.

### Evaluation status of the checked-in model

The existing `outputs/classification_report.txt` records approximately **82.5%** accuracy on 1,500 test images. That artifact was produced by an earlier benchmark that used test accuracy to select between candidate heads, so it is not an unbiased final estimate. Run one of the updated training scripts to generate a fresh validation-selected model and a clean held-out test report before presenting a final accuracy claim.

---

## 7. Streamlit Cloud Deployment Guide

1. **Push Code to GitHub**:
   Ensure the following repository files are committed:
   - `app.py`
   - `config.py`
   - `model_config.json`
   - `food_classifier.keras` (transfer learning model is only ~15–20 MB, well under GitHub's 100MB limit)
   - `class_names.json`
   - `calories.csv`
   - `Nutrition_Value_Dataset.csv`
   - `requirements.txt` (uses `tensorflow-cpu` to prevent large GPU build times)
   - `.gitignore`
   - `README.md`

2. **Deploy on Streamlit Community Cloud**:
   - Navigate to [share.streamlit.io](https://share.streamlit.io).
   - Click **New app** and connect your GitHub repository.
   - Set **Main file path** to `app.py`.
   - Under **Advanced settings**, select **Python 3.11**. The pinned TensorFlow 2.15 runtime does not support Community Cloud's default Python 3.12.
   - Click **Deploy**.

---

## 8. Handoff & Instructions for Any Subsequent Agent / Developer

If you are a new AI agent or developer continuing this repository:

1. **Inspect Background or Live Training**:
   - Check if `outputs/mobilenet_best.keras` or `outputs/model_comparison.txt` exist in `outputs/`.
   - If training is still running in background, check task log at `outputs/` or run `python train_model.py` if not running.
2. **Verify Final Model Artifacts**:
   - Confirm `food_classifier.keras` exists and has `output_shape == (None, 10)`.
   - Confirm `class_names.json` contains all 10 classes matching alphabetical order.
3. **Run Smoke Test**:
   ```powershell
   .venv\Scripts\python.exe -c "
   import json, numpy as np, tensorflow as tf
   model = tf.keras.models.load_model('food_classifier.keras', compile=False, safe_mode=True)
   classes = json.loads(open('class_names.json').read())
   assert model.input_shape[-1] == 3
   assert model.output_shape == (None, len(classes)) == (None, 10)
   sample = np.zeros((1, *model.input_shape[1:]), dtype=np.float32)
   predictions = model(sample, training=False).numpy()
   assert np.isfinite(predictions).all()
   print('Model loaded successfully. Classes:', len(classes))
   "
   ```
4. **Run Streamlit Smoke Test**:
   ```powershell
   .venv\Scripts\streamlit.exe run app.py
   ```
5. **Key Architectural Guarantees**:
   - `app.py` extracts `image_size` directly from `model.input_shape` so whatever resolution model was trained on is automatically matched during inference.
   - Preprocessing uses the model's input dimensions, training-matched bilinear resizing, and `[0, 1]` scaling; model-internal preprocessing handles the rest.
   - Uploads are validated, EXIF orientation is applied, and session state hashing prevents stale predictions when changing or removing images.

---

## 9. Limitations & Disclaimers

- **Approximate Nutritional Reference**: Calorie values represent typical standard servings from USDA FoodData Central and fast-food averages. The vision classifier identifies food categories, but cannot measure individual portion volume or specific custom recipes.
- **Out of Distribution Foods**: A confidence warning is shown below the configured threshold. Softmax confidence is not a reliable out-of-distribution detector, so unfamiliar foods can still receive an overconfident prediction.

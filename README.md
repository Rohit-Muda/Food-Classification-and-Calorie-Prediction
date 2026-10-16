# Food Image Classification and Calorie Category Prediction

This project trains a compact CNN to classify food images into five classes: Burger, Fries, Pizza, Donut, and Sandwich. The application then shows a reference calorie estimate and a calorie category based on a small nutrition lookup file.

## Project overview

- Dataset: `Fast Food Classification V2` within the main project folder
- Selected classes: `Burger`, `Fries`, `Pizza`, `Donut`, `Sandwich`
- Model: simple CNN built with TensorFlow/Keras
- App: Streamlit single-page interface

## Dataset structure

The project uses the existing split directories already present in the workspace:

- `Fast Food Classification V2/Train`
- `Fast Food Classification V2/Valid`
- `Fast Food Classification V2/Test`

The selected classes are present in all three splits, and the model is trained using the training split only. Validation and test data are kept separate to avoid leakage.

## Nutrition lookup approach

The nutrition reference file is `Nutrition_Value_Dataset.csv`. It contains per-serve calorie values for representative burger, fries, pizza, and sandwich items. For classes that are absent in the supplied nutrition dataset, a clearly labeled fallback reference is used for the app.

For the UI, the calorie categories follow a simple project-defined serving-based rule:

- Low: below 250 kcal
- Medium: 250 to below 400 kcal
- High: 400 kcal or above

This is an approximate classroom rule, not a medical classification.

## Files in this project

- `train_model.py` — dataset verification, model training, evaluation, and saved artifacts
- `app.py` — Streamlit app for prediction and calorie lookup
- `calories.csv` — reference calorie values and categories
- `requirements.txt` — Python dependencies
- `food_classifier.keras` — trained CNN model
- `class_names.json` — mapping between class indices and class labels
- `outputs/` — training history, confusion matrix, and classification report

## Training and evaluation

Run:

```powershell
python train_model.py
```

This script:

1. Verifies the five selected classes exist in each split.
2. Uses a reproducible random sample of up to 500 training images per class and streams validation/test images from the original split folders without copying or modifying the source dataset.
3. Resizes images to 128x128 RGB.
4. Normalizes pixel values to 0–1.
5. Trains on a reproducible sample of up to 500 training images per class with modest augmentation and early stopping.
6. Saves the model and class names.
7. Writes evaluation outputs to `outputs/`.

## Running the Streamlit app

```powershell
streamlit run app.py
```

Then open the local URL shown in the terminal.

## Deployment notes

- This project is kept simple for GitHub and Streamlit Community Cloud hosting.
- Do not commit the original image dataset.
- Commit `food_classifier.keras` and `class_names.json`; the app needs both. The current model is about 51.5 MB, below GitHub's 100 MB per-file limit but above its 50 MB warning threshold.
- The training dataset is not needed to run the deployed app and should remain out of the repository.

## Limitations

- The saved class mapping and training/app preprocessing were checked and agree. The current test accuracy is 54.0% (900 held-out images); the lightweight CNN still confuses some classes.
- Visual spot checks found a likely mislabeled example in `Test/Sandwich` (`Sandwich-Test (1).jpeg` appears to show a donut), so source-label noise may also affect measured accuracy. The original dataset is not modified.
- The CNN predicts the food class only.
- The calorie value is a reference estimate from a nutrition lookup table, not a direct prediction from the image.
- The output is approximate and depends on serving size, ingredients, and preparation method.

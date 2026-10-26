# Explainable AI Healthcare Risk Prediction - Frontend

A Streamlit dashboard for the project. Lets a user pick a condition (Diabetes
or Heart Disease), enter health parameters, get a risk prediction, and see a
SHAP-based explanation of *why* the model predicted that risk.

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. (Recommended) Add the real datasets

The app runs out of the box on **synthetic demo data** for Diabetes, Heart
Disease, Liver Disease, and Kidney Disease (Breast Cancer always uses real
data automatically), so you can test the UI immediately. For your actual
submission, download the real datasets:

- **Diabetes**: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- **Heart Disease**: https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset
- **Liver Disease**: https://www.kaggle.com/datasets/uciml/indian-liver-patient-records
- **Kidney Disease**: https://www.kaggle.com/datasets/mansoordaku/ckdisease

and place them here:

```
data/diabetes.csv   <- Pima Indians Diabetes Dataset (Kaggle / UCI)
data/heart.csv      <- UCI Cleveland Heart Disease Dataset (Kaggle / UCI)
data/liver.csv      <- Indian Liver Patient Dataset (Kaggle / UCI)
data/kidney.csv     <- UCI Chronic Kidney Disease Dataset (Kaggle / UCI)
```

**Breast Cancer Risk needs no file at all** - it uses the Wisconsin Diagnostic
Breast Cancer dataset that ships built into scikit-learn, so it's always
using real data with zero setup.

Column names must match what's expected in `utils.py`:

- **diabetes.csv**: `Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome`
- **heart.csv**: `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`
- **liver.csv**: `Age, Gender, Total_Bilirubin, Alkaline_Phosphotase, Alamine_Aminotransferase, Aspartate_Aminotransferase, Albumin, Albumin_and_Globulin_Ratio, Outcome` (the original ILPD file uses `Dataset` as the target column with values 1/2; rename it to `Outcome` with 1 = disease, 0 = no disease)
- **kidney.csv**: `age, bp, bgr, bu, sc, sod, hemo, htn, Outcome` (the original UCI file's `classification` column with values `ckd`/`notckd` needs renaming to `Outcome` with 1/0)

These are the standard column names used in the common Kaggle/UCI versions of
each dataset, so a direct download usually needs no renaming.

If a CSV isn't found, the app automatically falls back to synthetic data and
shows a warning banner in the sidebar so it's always clear which mode you're
running in.

## 3. Run the app

```bash
streamlit run app.py
```

This opens the dashboard in your browser (usually `http://localhost:8501`).

## 5. (Optional) Enable report image auto-fill (OCR)

The Predict tab lets you upload a photo/scan of a lab report and auto-fill
matching fields (Glucose, BMI, Blood Pressure, Cholesterol, etc.) instead of
typing them in. This needs one extra piece **beyond** `pip install`:

1. Download and install **Tesseract-OCR** for Windows:
   https://github.com/UB-Mannheim/tesseract/wiki
   (use the default install location - the app looks for it automatically at
   `C:\Program Files\Tesseract-OCR\tesseract.exe`)
2. That's it - `pytesseract` (already in `requirements.txt`) talks to that
   engine to read the image.

**Important limitations to know:**
- This is best-effort text recognition, not a certified medical document
  parser. Results depend heavily on image quality, font, and report layout.
- Only common numeric lab values (Glucose, BMI, Blood Pressure, Cholesterol,
  Age, Insulin) are auto-detected. Clinical/derived fields (like Diabetes
  Pedigree Function, or ECG-derived heart fields) always need manual entry.
- Extracted values only **pre-fill** the sliders - nothing is submitted
  automatically. Always double-check the numbers before clicking Predict.
- If Tesseract isn't installed, the app shows a clear error instead of
  crashing, and manual entry still works normally.

## What's in each file

- **`app.py`** - the Streamlit UI: login/signup screen, input form, prediction,
  SHAP chart, performance metrics, progress tracking, and prediction history.
- **`utils.py`** - dataset config, data loading (real or synthetic fallback),
  model training (Random Forest), SHAP explanation logic, user accounts
  (password-hashed with PBKDF2), and per-user prediction storage in SQLite
  (`data/predictions.db`, created automatically).
- **`requirements.txt`** - Python packages needed to run the app.
- **`.streamlit/config.toml`** - the app's color theme.

## Accounts & Progress Tracking

- On first visit, create an account with a username and password (stored
  locally, password never stored in plain text).
- Every prediction you make is saved under your account.
- Once you have **2 or more predictions** for the same condition, the
  **"My Progress"** tab shows whether your risk has improved or worsened
  since your last check, plus a trend line across all your readings.
- Accounts and history are stored locally in `data/predictions.db` - there is
  no external server, so this is meant for local/demo use, not production
  deployment with real patient data.

## Notes for your report / viva

- **Models compared**: Logistic Regression, Random Forest, and Gradient
  Boosting are all trained per dataset with an 80/20 train-test split; the
  one with the best F1-score is automatically selected for predictions,
  directly matching the objective of comparing multiple algorithms rather
  than committing to a single model.
- **Explainability**: `shap.TreeExplainer` for tree-based models (Random
  Forest, Gradient Boosting) or `shap.LinearExplainer` for Logistic
  Regression, whichever the comparison selects - gives exact Shapley-value
  attributions per prediction either way.
- **Storage**: user accounts and predictions are stored in a local SQLite
  database (`data/predictions.db`), with passwords hashed (PBKDF2) rather
  than stored in plain text.
- **Reporting**: after each prediction, a downloadable plain-text report is
  generated summarizing the inputs, risk score, model used, and the top
  SHAP-driven factors - useful evidence to attach to your project report.
- The "Model Performance" tab shows the full comparison table across all
  three algorithms - useful numbers to quote when asked how you evaluated
  and selected the model.

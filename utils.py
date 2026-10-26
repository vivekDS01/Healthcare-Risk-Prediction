"""
utils.py
--------
Backend-ish helper logic for the Explainable AI Healthcare Risk Prediction
frontend: dataset configuration, data loading (real CSV if present, else a
synthetic fallback), model training, SHAP explanation, and a tiny SQLite
history log.

This file is intentionally framework-light (no Streamlit imports) so it can
be unit-tested or reused if the model/backend is later split into its own
service.
"""

import os
import sqlite3
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "predictions.db")

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Dataset configuration
# ---------------------------------------------------------------------------
# Each feature is described as: (column_name, label, widget, kwargs)
# widget is one of "slider" or "select" -> tells app.py which input to render.

DATASETS = {
    "diabetes": {
        "display_name": "Diabetes Risk (Pima Indians Diabetes Dataset)",
        "csv_filename": "diabetes.csv",
        "target_col": "Outcome",
        "condition_name": "diabetes",
        "features": [
            ("Pregnancies", "Number of Pregnancies", "slider", dict(min_value=0, max_value=17, value=2, step=1)),
            ("Glucose", "Plasma Glucose Concentration (mg/dL)", "slider", dict(min_value=0, max_value=200, value=120, step=1)),
            ("BloodPressure", "Diastolic Blood Pressure (mm Hg)", "slider", dict(min_value=0, max_value=140, value=70, step=1)),
            ("SkinThickness", "Triceps Skin Fold Thickness (mm)", "slider", dict(min_value=0, max_value=100, value=20, step=1)),
            ("Insulin", "2-Hour Serum Insulin (mu U/mL)", "slider", dict(min_value=0, max_value=850, value=80, step=1)),
            ("BMI", "Body Mass Index", "slider", dict(min_value=10.0, max_value=70.0, value=25.0, step=0.1)),
            ("DiabetesPedigreeFunction", "Diabetes Pedigree Function", "slider", dict(min_value=0.0, max_value=2.5, value=0.5, step=0.01)),
            ("Age", "Age (years)", "slider", dict(min_value=18, max_value=100, value=33, step=1)),
        ],
    },
    "heart": {
        "display_name": "Heart Disease Risk (UCI Cleveland Dataset)",
        "csv_filename": "heart.csv",
        "target_col": "target",
        "condition_name": "heart disease",
        "features": [
            ("age", "Age (years)", "slider", dict(min_value=20, max_value=100, value=50, step=1)),
            ("sex", "Sex", "select", dict(options=[("Female", 0), ("Male", 1)])),
            ("cp", "Chest Pain Type", "select", dict(options=[
                ("Typical angina", 0), ("Atypical angina", 1),
                ("Non-anginal pain", 2), ("Asymptomatic", 3)])),
            ("trestbps", "Resting Blood Pressure (mm Hg)", "slider", dict(min_value=80, max_value=220, value=130, step=1)),
            ("chol", "Serum Cholesterol (mg/dL)", "slider", dict(min_value=100, max_value=600, value=240, step=1)),
            ("fbs", "Fasting Blood Sugar > 120 mg/dL", "select", dict(options=[("No", 0), ("Yes", 1)])),
            ("restecg", "Resting ECG Result", "select", dict(options=[
                ("Normal", 0), ("ST-T wave abnormality", 1), ("Left ventricular hypertrophy", 2)])),
            ("thalach", "Max Heart Rate Achieved", "slider", dict(min_value=60, max_value=220, value=150, step=1)),
            ("exang", "Exercise-Induced Angina", "select", dict(options=[("No", 0), ("Yes", 1)])),
            ("oldpeak", "ST Depression (exercise vs rest)", "slider", dict(min_value=0.0, max_value=6.5, value=1.0, step=0.1)),
            ("slope", "Slope of Peak Exercise ST Segment", "select", dict(options=[
                ("Upsloping", 0), ("Flat", 1), ("Downsloping", 2)])),
            ("ca", "Major Vessels Colored by Fluoroscopy", "select", dict(options=[(str(i), i) for i in range(5)])),
            ("thal", "Thalassemia", "select", dict(options=[
                ("Normal", 0), ("Fixed defect", 1), ("Reversible defect", 2), ("Unknown", 3)])),
        ],
    },
    "breast_cancer": {
        "display_name": "Breast Cancer Risk (Wisconsin Diagnostic Dataset)",
        "csv_filename": None,  # ships built into scikit-learn, no CSV needed
        "target_col": "Outcome",
        "condition_name": "breast cancer",
        "features": [
            ("mean_radius", "Mean Cell Radius", "slider", dict(min_value=6.0, max_value=29.0, value=14.0, step=0.1)),
            ("mean_texture", "Mean Cell Texture", "slider", dict(min_value=9.0, max_value=40.0, value=19.0, step=0.1)),
            ("mean_perimeter", "Mean Cell Perimeter", "slider", dict(min_value=43.0, max_value=189.0, value=92.0, step=0.5)),
            ("mean_area", "Mean Cell Area", "slider", dict(min_value=140.0, max_value=2501.0, value=655.0, step=1.0)),
            ("mean_smoothness", "Mean Smoothness", "slider", dict(min_value=0.05, max_value=0.17, value=0.10, step=0.001)),
            ("mean_compactness", "Mean Compactness", "slider", dict(min_value=0.02, max_value=0.35, value=0.10, step=0.001)),
            ("mean_concavity", "Mean Concavity", "slider", dict(min_value=0.0, max_value=0.43, value=0.09, step=0.001)),
            ("mean_concave_points", "Mean Concave Points", "slider", dict(min_value=0.0, max_value=0.20, value=0.05, step=0.001)),
        ],
    },
    "liver": {
        "display_name": "Liver Disease Risk (Indian Liver Patient Dataset)",
        "csv_filename": "liver.csv",
        "target_col": "Outcome",
        "condition_name": "liver disease",
        "features": [
            ("Age", "Age (years)", "slider", dict(min_value=4, max_value=90, value=45, step=1)),
            ("Gender", "Gender", "select", dict(options=[("Female", 0), ("Male", 1)])),
            ("Total_Bilirubin", "Total Bilirubin (mg/dL)", "slider", dict(min_value=0.1, max_value=75.0, value=1.0, step=0.1)),
            ("Alkaline_Phosphotase", "Alkaline Phosphotase (IU/L)", "slider", dict(min_value=60, max_value=2110, value=290, step=5)),
            ("Alamine_Aminotransferase", "Alamine Aminotransferase / ALT (IU/L)", "slider", dict(min_value=10, max_value=2000, value=80, step=5)),
            ("Aspartate_Aminotransferase", "Aspartate Aminotransferase / AST (IU/L)", "slider", dict(min_value=10, max_value=5000, value=110, step=5)),
            ("Albumin", "Albumin (g/dL)", "slider", dict(min_value=0.9, max_value=5.5, value=3.1, step=0.1)),
            ("Albumin_and_Globulin_Ratio", "Albumin and Globulin Ratio", "slider", dict(min_value=0.3, max_value=2.8, value=0.9, step=0.01)),
        ],
    },
    "kidney": {
        "display_name": "Kidney Disease Risk (UCI Chronic Kidney Disease Dataset)",
        "csv_filename": "kidney.csv",
        "target_col": "Outcome",
        "condition_name": "chronic kidney disease",
        "features": [
            ("age", "Age (years)", "slider", dict(min_value=2, max_value=90, value=50, step=1)),
            ("bp", "Blood Pressure (mm Hg)", "slider", dict(min_value=50, max_value=180, value=80, step=1)),
            ("bgr", "Blood Glucose Random (mg/dL)", "slider", dict(min_value=22, max_value=490, value=120, step=1)),
            ("bu", "Blood Urea (mg/dL)", "slider", dict(min_value=1.5, max_value=390.0, value=45.0, step=0.5)),
            ("sc", "Serum Creatinine (mg/dL)", "slider", dict(min_value=0.4, max_value=76.0, value=1.2, step=0.1)),
            ("sod", "Sodium (mEq/L)", "slider", dict(min_value=4.5, max_value=163.0, value=137.0, step=0.5)),
            ("hemo", "Hemoglobin (g/dL)", "slider", dict(min_value=3.1, max_value=17.8, value=13.0, step=0.1)),
            ("htn", "Hypertension", "select", dict(options=[("No", 0), ("Yes", 1)])),
        ],
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _generate_synthetic_diabetes(n=768):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "Pregnancies": rng.integers(0, 15, n),
        "Glucose": rng.normal(120, 30, n).clip(50, 200),
        "BloodPressure": rng.normal(70, 15, n).clip(30, 130),
        "SkinThickness": rng.normal(23, 10, n).clip(0, 90),
        "Insulin": rng.normal(85, 90, n).clip(0, 700),
        "BMI": rng.normal(31, 7, n).clip(15, 65),
        "DiabetesPedigreeFunction": rng.gamma(2, 0.2, n).clip(0.05, 2.4),
        "Age": rng.integers(18, 80, n),
    })
    def z(s):
        return (s - s.mean()) / s.std()

    logit = (
        1.3 * z(df["Glucose"])
        + 1.0 * z(df["BMI"])
        + 0.6 * z(df["Age"])
        + 0.9 * z(df["DiabetesPedigreeFunction"])
        + rng.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Outcome"] = (rng.random(n) < prob).astype(int)
    return df


def _generate_synthetic_heart(n=740):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "age": rng.integers(29, 77, n),
        "sex": rng.integers(0, 2, n),
        "cp": rng.integers(0, 4, n),
        "trestbps": rng.normal(131, 17, n).clip(90, 200).round(),
        "chol": rng.normal(246, 51, n).clip(120, 560).round(),
        "fbs": (rng.random(n) < 0.15).astype(int),
        "restecg": rng.integers(0, 3, n),
        "thalach": rng.normal(149, 23, n).clip(70, 210).round(),
        "exang": (rng.random(n) < 0.33).astype(int),
        "oldpeak": rng.gamma(1.3, 0.9, n).clip(0, 6.2).round(1),
        "slope": rng.integers(0, 3, n),
        "ca": rng.integers(0, 5, n),
        "thal": rng.integers(0, 4, n),
    })
    def z(s):
        return (s - s.mean()) / s.std()

    logit = (
        0.7 * z(df["age"])
        + 0.6 * (df["sex"] - 0.5) * 2
        + 0.7 * z(df["cp"])
        - 0.8 * z(df["thalach"])
        + 0.7 * (df["exang"] - 0.5) * 2
        + 0.9 * z(df["oldpeak"])
        + 0.8 * z(df["ca"])
        + rng.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["target"] = (rng.random(n) < prob).astype(int)
    return df


def _generate_synthetic_liver(n=580):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "Age": rng.integers(4, 90, n),
        "Gender": rng.integers(0, 2, n),
        "Total_Bilirubin": rng.gamma(1.5, 1.2, n).clip(0.1, 75.0).round(1),
        "Alkaline_Phosphotase": rng.normal(290, 240, n).clip(60, 2110).round(),
        "Alamine_Aminotransferase": rng.gamma(2.0, 40, n).clip(10, 2000).round(),
        "Aspartate_Aminotransferase": rng.gamma(2.0, 55, n).clip(10, 5000).round(),
        "Albumin": rng.normal(3.1, 0.8, n).clip(0.9, 5.5).round(1),
        "Albumin_and_Globulin_Ratio": rng.normal(0.95, 0.32, n).clip(0.3, 2.8).round(2),
    })

    def z(s):
        return (s - s.mean()) / s.std()

    logit = (
        0.9 * z(df["Total_Bilirubin"])
        + 0.8 * z(df["Alamine_Aminotransferase"])
        + 0.7 * z(df["Aspartate_Aminotransferase"])
        + 0.5 * z(df["Alkaline_Phosphotase"])
        - 0.6 * z(df["Albumin"])
        - 0.5 * z(df["Albumin_and_Globulin_Ratio"])
        + 0.3 * z(df["Age"])
        + rng.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Outcome"] = (rng.random(n) < prob).astype(int)
    return df


def _generate_synthetic_kidney(n=400):
    rng = np.random.default_rng(RANDOM_STATE)
    df = pd.DataFrame({
        "age": rng.integers(2, 90, n),
        "bp": rng.normal(80, 15, n).clip(50, 180).round(),
        "bgr": rng.normal(130, 55, n).clip(22, 490).round(),
        "bu": rng.gamma(2.0, 22, n).clip(1.5, 390.0).round(1),
        "sc": rng.gamma(1.5, 1.0, n).clip(0.4, 76.0).round(1),
        "sod": rng.normal(137, 8, n).clip(4.5, 163.0).round(1),
        "hemo": rng.normal(12.5, 2.8, n).clip(3.1, 17.8).round(1),
        "htn": (rng.random(n) < 0.35).astype(int),
    })

    def z(s):
        return (s - s.mean()) / s.std()

    logit = (
        1.0 * z(df["sc"])
        + 0.8 * z(df["bu"])
        - 0.7 * z(df["hemo"])
        + 0.6 * z(df["bgr"])
        + 0.6 * (df["htn"] - 0.5) * 2
        + 0.4 * z(df["bp"])
        + rng.normal(0, 1.0, n)
    )
    prob = 1 / (1 + np.exp(-logit))
    df["Outcome"] = (rng.random(n) < prob).astype(int)
    return df


def _load_breast_cancer():
    """
    Loads the Wisconsin Diagnostic Breast Cancer dataset, which ships built
    into scikit-learn (no CSV, no download, no synthetic fallback needed).
    Only the 8 "mean" measurements used in the form are kept, and the target
    is flipped so 1 = malignant (matches this app's "1 = higher risk"
    convention; scikit-learn's own encoding is 0 = malignant, 1 = benign).
    """
    from sklearn.datasets import load_breast_cancer

    data = load_breast_cancer(as_frame=True)
    raw = data.frame
    df = pd.DataFrame({
        "mean_radius": raw["mean radius"],
        "mean_texture": raw["mean texture"],
        "mean_perimeter": raw["mean perimeter"],
        "mean_area": raw["mean area"],
        "mean_smoothness": raw["mean smoothness"],
        "mean_compactness": raw["mean compactness"],
        "mean_concavity": raw["mean concavity"],
        "mean_concave_points": raw["mean concave points"],
    })
    df["Outcome"] = 1 - raw["target"]  # flip: 1 = malignant/higher risk
    return df


def load_dataset(dataset_key: str):
    """
    Returns (dataframe, is_synthetic: bool).
    Looks for a real CSV in ./data first (e.g. data/diabetes.csv). If missing,
    falls back to a synthetic dataset that mimics the real one's structure,
    so the app is runnable end-to-end without external files. Datasets with
    csv_filename=None (like breast_cancer) load directly from their built-in
    source instead and are never synthetic.
    """
    config = DATASETS[dataset_key]

    if config["csv_filename"] is None:
        if dataset_key == "breast_cancer":
            return _load_breast_cancer(), False
        raise ValueError(f"No loader defined for built-in dataset: {dataset_key}")

    csv_path = os.path.join(DATA_DIR, config["csv_filename"])

    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
        df.columns = df.columns.str.strip()  # guards against stray whitespace in headers
        return df, False

    if dataset_key == "diabetes":
        return _generate_synthetic_diabetes(), True
    elif dataset_key == "heart":
        return _generate_synthetic_heart(), True
    elif dataset_key == "liver":
        return _generate_synthetic_liver(), True
    elif dataset_key == "kidney":
        return _generate_synthetic_kidney(), True
    raise ValueError(f"Unknown dataset key: {dataset_key}")


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

def _candidate_models():
    """The set of algorithms compared for every dataset, as per the project objectives."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=RANDOM_STATE, class_weight="balanced"
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=150, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def train_model(df: pd.DataFrame, dataset_key: str):
    """
    Trains several candidate algorithms (Logistic Regression, Random Forest,
    Gradient Boosting), evaluates each on a held-out test split, and selects
    the best-performing one by F1-score, matching the project objective of
    comparing multiple ML algorithms rather than committing to a single model.

    Returns (best_model, X_train, comparison_dict, feature_cols, best_model_name)
    where comparison_dict maps model name -> its metrics dict.
    """
    config = DATASETS[dataset_key]
    feature_cols = [f[0] for f in config["features"]]
    target_col = config["target_col"]

    X = df[feature_cols]
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    comparison = {}
    fitted_models = {}

    for name, clf in _candidate_models().items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        comparison[name] = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "auc": roc_auc_score(y_test, y_proba) if len(set(y_test)) > 1 else float("nan"),
        }
        fitted_models[name] = clf

    best_name = max(comparison, key=lambda n: comparison[n]["f1"])
    best_model = fitted_models[best_name]

    return best_model, X_train, comparison, feature_cols, best_name


def get_feature_importance(model, feature_cols):
    """
    Works across model types: tree ensembles expose feature_importances_,
    linear models expose coef_. Falls back to zeros otherwise.
    """
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_, index=feature_cols)
    if hasattr(model, "coef_"):
        return pd.Series(np.abs(model.coef_[0]), index=feature_cols)
    return pd.Series([0.0] * len(feature_cols), index=feature_cols)


def get_risk_band(proba: float, condition_name: str):
    """
    Bands a raw probability into Low / Moderate / High, the way clinical
    dashboards typically present risk rather than a stark 50% cutoff.
    Returns (band_name, color_hex, full_label).
    """
    if proba < 0.33:
        band, color = "Low", "#16A34A"
    elif proba < 0.66:
        band, color = "Moderate", "#D97706"
    else:
        band, color = "High", "#DC2626"
    return band, color, f"{band} risk of {condition_name}"


# ---------------------------------------------------------------------------
# Explainability (SHAP)
# ---------------------------------------------------------------------------

def explain_instance(model, background_df: pd.DataFrame, instance_df: pd.DataFrame):
    """
    Returns a DataFrame with columns [feature, value, shap_value] for one
    prediction, sorted by absolute impact (largest first).

    Uses TreeExplainer for tree-based models (Random Forest, Gradient
    Boosting) since it's fast and exact, and LinearExplainer for Logistic
    Regression, because the best-performing model can be either depending
    on the dataset, model comparison picks whichever wins.
    """
    import shap  # imported lazily so utils.py can be inspected without shap installed

    if hasattr(model, "feature_importances_"):
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(instance_df)
        if isinstance(shap_values, list):
            values = shap_values[1][0]
        else:
            values = shap_values[0]
            if values.ndim == 2:  # (n_features, n_classes)
                values = values[:, 1]
    else:
        # Linear model (e.g. Logistic Regression)
        background_sample = background_df.sample(min(100, len(background_df)), random_state=RANDOM_STATE)
        explainer = shap.LinearExplainer(model, background_sample)
        shap_values = explainer.shap_values(instance_df)
        values = shap_values[0] if not isinstance(shap_values, list) else shap_values[1][0]

    result = pd.DataFrame({
        "feature": instance_df.columns,
        "value": instance_df.iloc[0].values,
        "shap_value": values,
    })
    result["abs_impact"] = result["shap_value"].abs()
    return result.sort_values("abs_impact", ascending=False).drop(columns="abs_impact")


# ---------------------------------------------------------------------------
# User accounts (SQLite, password hashed with PBKDF2, no extra dependency)
# ---------------------------------------------------------------------------

def init_db():
    os.makedirs(DATA_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            salt TEXT,
            password_hash TEXT,
            created_at TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            dataset TEXT,
            timestamp TEXT,
            inputs_json TEXT,
            risk_probability REAL,
            risk_label TEXT
        )
    """)
    conn.commit()

    # Migration safety net: if an older predictions table exists without a
    # username column (from before accounts were added), add it so existing
    # local databases don't break.
    existing_cols = [row[1] for row in conn.execute("PRAGMA table_info(predictions)")]
    if "username" not in existing_cols:
        conn.execute("ALTER TABLE predictions ADD COLUMN username TEXT DEFAULT 'guest'")
        conn.commit()

    conn.close()


def _hash_password(password: str, salt: bytes = None):
    import hashlib
    if salt is None:
        salt = os.urandom(16)
    pwd_hash = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex(), pwd_hash.hex()


def create_user(username: str, password: str) -> bool:
    """Returns True if the account was created, False if the username is taken."""
    from datetime import datetime

    init_db()
    conn = sqlite3.connect(DB_PATH)
    existing = conn.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone()
    if existing:
        conn.close()
        return False

    salt_hex, hash_hex = _hash_password(password)
    conn.execute(
        "INSERT INTO users (username, salt, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (username, salt_hex, hash_hex, datetime.now().isoformat(timespec="seconds")),
    )
    conn.commit()
    conn.close()
    return True


def verify_user(username: str, password: str) -> bool:
    init_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute(
        "SELECT salt, password_hash FROM users WHERE username = ?", (username,)
    ).fetchone()
    conn.close()
    if not row:
        return False
    salt_hex, stored_hash_hex = row
    _, computed_hash_hex = _hash_password(password, bytes.fromhex(salt_hex))
    return computed_hash_hex == stored_hash_hex


def get_user_info(username: str):
    """Returns {'created_at': str, 'total_predictions': int} for the account panel."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    row = conn.execute("SELECT created_at FROM users WHERE username = ?", (username,)).fetchone()
    created_at = row[0] if row else None
    count = conn.execute(
        "SELECT COUNT(*) FROM predictions WHERE username = ?", (username,)
    ).fetchone()[0]
    conn.close()
    return {"created_at": created_at, "total_predictions": count}


# ---------------------------------------------------------------------------
# Per-user prediction history (SQLite)
# ---------------------------------------------------------------------------

def log_prediction(username, dataset_key, inputs_dict, risk_probability, risk_label):
    import json
    from datetime import datetime

    init_db()
    conn = sqlite3.connect(DB_PATH)
    conn.execute(
        "INSERT INTO predictions (username, dataset, timestamp, inputs_json, risk_probability, risk_label) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (username, dataset_key, datetime.now().isoformat(timespec="seconds"),
         json.dumps(inputs_dict), float(risk_probability), risk_label),
    )
    conn.commit()
    conn.close()


def get_history(username, dataset_key=None, limit=200):
    """Returns this user's prediction history, most recent first.
    Pass dataset_key=None to get history across every condition."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    query = "SELECT timestamp, dataset, risk_probability, risk_label FROM predictions WHERE username = ?"
    params = [username]
    if dataset_key:
        query += " AND dataset = ?"
        params.append(dataset_key)
    query += " ORDER BY id DESC LIMIT ?"
    params.append(limit)
    df = pd.read_sql_query(query, conn, params=params)
    conn.close()
    if not df.empty:
        df["condition"] = df["dataset"].map(lambda k: DATASETS.get(k, {}).get("display_name", k))
        df["risk_probability"] = (df["risk_probability"] * 100).round(1)
        df = df.rename(columns={
            "timestamp": "Date", "condition": "Condition",
            "risk_probability": "Risk (%)", "risk_label": "Result",
        })[["Date", "Condition", "Risk (%)", "Result"]]
    return df


def get_progress(username, dataset_key):
    """
    Returns the full chronological (oldest -> newest) history for this user
    and dataset, used to compare the latest reading against the previous one
    and to plot a trend line.
    """
    init_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT timestamp, risk_probability, risk_label FROM predictions "
        "WHERE username = ? AND dataset = ? ORDER BY id ASC",
        conn, params=(username, dataset_key),
    )
    conn.close()
    return df


# ---------------------------------------------------------------------------
# Downloadable report
# ---------------------------------------------------------------------------

def build_report_text(username, dataset_name, model_name, inputs_dict,
                       risk_probability, risk_label, explanation_df) -> str:
    """Builds a simple plain-text report summarizing one prediction."""
    from datetime import datetime

    lines = [
        "=" * 60,
        "EXPLAINABLE AI HEALTHCARE RISK PREDICTION REPORT",
        "=" * 60,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"User: {username}",
        f"Assessment: {dataset_name}",
        f"Model used: {model_name}",
        "",
        "-" * 60,
        "RESULT",
        "-" * 60,
        f"Predicted risk probability: {risk_probability * 100:.1f}%",
        f"Risk category: {risk_label}",
        "",
        "-" * 60,
        "PATIENT INPUT VALUES",
        "-" * 60,
    ]
    for key, value in inputs_dict.items():
        lines.append(f"  {key}: {value}")

    lines += [
        "",
        "-" * 60,
        "TOP FACTORS BEHIND THIS PREDICTION (SHAP)",
        "-" * 60,
    ]
    for _, row in explanation_df.head(8).iterrows():
        direction = "increased" if row["shap_value"] > 0 else "decreased"
        lines.append(f"  {row['feature']} = {row['value']}  ->  {direction} risk (impact {row['shap_value']:+.3f})")

    lines += [
        "",
        "-" * 60,
        "DISCLAIMER",
        "-" * 60,
        "This report is generated by an AI-assisted decision support tool and",
        "is not a medical diagnosis. Please consult a qualified healthcare",
        "professional for any medical decisions.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# OCR: extract health parameters from an uploaded report image
# ---------------------------------------------------------------------------
# This is best-effort text extraction, not a medical-grade document parser.
# Report layouts vary wildly, so results should always be shown to the user
# for verification/correction before being used in a prediction, never
# auto-submitted silently.

# Regex label variants that commonly appear on lab reports / printed charts
# for each numeric field we can realistically expect to find in free text.
# Clinical/derived fields (e.g. DiabetesPedigreeFunction, ECG-derived codes
# like 'thal' or 'slope') are intentionally excluded because they don't appear on
# a typical printed report and shouldn't be guessed at.
FIELD_LABEL_PATTERNS = {
    "diabetes": {
        "Glucose": [r"glucose", r"blood\s*sugar", r"\bfbs\b"],
        "BloodPressure": [r"blood\s*pressure", r"\bbp\b", r"diastolic"],
        "SkinThickness": [r"skin\s*thickness", r"triceps"],
        "Insulin": [r"insulin"],
        "BMI": [r"\bbmi\b", r"body\s*mass\s*index"],
        "Age": [r"\bage\b"],
        "Pregnancies": [r"pregnanc\w*"],
    },
    "heart": {
        "age": [r"\bage\b"],
        "trestbps": [r"resting\s*blood\s*pressure", r"blood\s*pressure", r"\bbp\b"],
        "chol": [r"cholesterol", r"\bchol\b"],
        "thalach": [r"max(?:imum)?\s*heart\s*rate", r"heart\s*rate", r"\bhr\b"],
        "oldpeak": [r"st\s*depression", r"oldpeak"],
    },
    "liver": {
        "Age": [r"\bage\b"],
        "Total_Bilirubin": [r"total\s*bilirubin"],
        "Alkaline_Phosphotase": [r"alkaline\s*phosphotase", r"alkaline\s*phosphatase", r"\balp\b"],
        "Alamine_Aminotransferase": [r"alamine\s*aminotransferase", r"\balt\b"],
        "Aspartate_Aminotransferase": [r"aspartate\s*aminotransferase", r"\bast\b"],
        "Albumin": [r"\balbumin\b(?!\s*and)"],
        "Albumin_and_Globulin_Ratio": [r"albumin\s*and\s*globulin\s*ratio", r"a\s*/\s*g\s*ratio"],
    },
    "kidney": {
        "age": [r"\bage\b"],
        "bp": [r"blood\s*pressure", r"\bbp\b"],
        "bgr": [r"blood\s*glucose\s*random", r"glucose"],
        "bu": [r"blood\s*urea", r"\bbu\b"],
        "sc": [r"serum\s*creatinine", r"\bsc\b"],
        "sod": [r"sodium", r"\bsod\b"],
        "hemo": [r"hemoglobin", r"\bhemo\b"],
    },
}


def extract_text_from_image(image_bytes: bytes) -> str:
    """
    Runs OCR on an uploaded report image and returns the raw extracted text.
    Requires the `pytesseract` package AND the Tesseract-OCR system binary
    to be installed separately (see README for the Windows install link).
    """
    import io
    import platform
    import pytesseract
    from PIL import Image

    # On Windows, pytesseract can't find the engine unless it's on PATH.
    # If it's installed at the default location, point pytesseract at it.
    if platform.system() == "Windows":
        default_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        if os.path.exists(default_path):
            pytesseract.pytesseract.tesseract_cmd = default_path

    image = Image.open(io.BytesIO(image_bytes))
    return pytesseract.image_to_string(image)


def extract_fields_from_text(text: str, dataset_key: str) -> dict:
    """
    Scans OCR text for known lab-report labels and pulls out the first
    number that follows each one. Returns {field_name: float_value} only
    for fields it's reasonably confident it found. Missing fields are
    simply left out, not guessed.
    """
    import re

    results = {}
    for field, label_patterns in FIELD_LABEL_PATTERNS.get(dataset_key, {}).items():
        for label in label_patterns:
            match = re.search(rf"{label}[^\d\-]{{0,35}}(-?\d+\.?\d*)", text, flags=re.IGNORECASE)
            if match:
                try:
                    results[field] = float(match.group(1))
                    break
                except ValueError:
                    continue
    return results

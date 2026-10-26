"""
Run this once from your frontend folder to fix liver.csv and kidney.csv so
they match what the app expects:

    python fix_datasets.py

It rewrites both files in place (after backing up the originals) and prints
a summary of what changed.
"""
import pandas as pd
import os

DATA_DIR = "data"


def fix_liver():
    path = os.path.join(DATA_DIR, "liver.csv")
    if not os.path.exists(path):
        print("liver.csv not found, skipping.")
        return

    df = pd.read_csv(path)
    os.rename(path, path + ".bak")

    # Gender in the original ILPD file is text ("Male"/"Female"); the app
    # expects 0/1. (Always coerce to string first - newer pandas versions use
    # a "str" dtype for text columns instead of the old "object" dtype, so a
    # dtype check here can't be relied on.)
    df["Gender"] = df["Gender"].astype(str).str.strip().str.lower().map({"female": 0, "male": 1})

    # The target column is called "Dataset" with 1 = liver disease,
    # 2 = no liver disease. The app expects a column called "Outcome"
    # with 1 = disease, 0 = no disease.
    df["Outcome"] = df["Dataset"].map({1: 1, 2: 0})
    df = df.drop(columns=["Dataset"])

    # A few rows in the original file have a missing Albumin_and_Globulin_Ratio;
    # fill with the column median rather than losing those rows.
    df["Albumin_and_Globulin_Ratio"] = df["Albumin_and_Globulin_Ratio"].fillna(
        df["Albumin_and_Globulin_Ratio"].median()
    )

    df = df.dropna(subset=["Outcome"])

    df.to_csv(path, index=False)
    print(f"Fixed liver.csv -> {len(df)} rows. Original backed up as liver.csv.bak")


def fix_kidney():
    path = os.path.join(DATA_DIR, "kidney.csv")
    if not os.path.exists(path):
        print("kidney.csv not found, skipping.")
        return

    df = pd.read_csv(path)
    os.rename(path, path + ".bak")

    # This UCI file often has stray whitespace/tabs in text columns
    # (e.g. "ckd\t" instead of "ckd"). Always coerce to string first - newer
    # pandas versions use a "str" dtype for text columns instead of the old
    # "object" dtype, so a dtype check here can't be relied on.
    for col in ["classification", "htn"]:
        df[col] = df[col].astype(str).str.strip().str.lower()

    df["Outcome"] = df["classification"].map({"ckd": 1, "notckd": 0})
    df["htn"] = df["htn"].map({"yes": 1, "no": 0})

    # Numeric columns sometimes contain "?" for missing values in this dataset.
    numeric_cols = ["age", "bp", "bgr", "bu", "sc", "sod", "hemo"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median())  # fill gaps rather than losing rows

    # htn ("yes"/"no") sometimes has a stray missing value too; fill with the
    # most common answer rather than dropping the row.
    df["htn"] = df["htn"].fillna(df["htn"].mode()[0])

    # Keep only the columns the app actually needs, then drop any row still
    # missing its Outcome label (that one can't be safely filled in).
    keep_cols = numeric_cols + ["htn", "Outcome"]
    df = df[keep_cols].dropna(subset=["Outcome"])

    df.to_csv(path, index=False)
    print(f"Fixed kidney.csv -> {len(df)} rows. Original backed up as kidney.csv.bak")


if __name__ == "__main__":
    fix_liver()
    fix_kidney()
    print("Done. Restart the Streamlit app now.")

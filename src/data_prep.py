from __future__ import annotations

import os
import sys
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "creditcard.csv"
PROCESSED = ROOT / "data" / "processed"

def download_dataset() -> None:
    """Download via kaggle CLI if the raw file is missing."""
    if RAW.exists():
        print(f"Dataset already present at {RAW}")
        return

    print("Downloading creditcardfraud dataset via kaggle CLI …")
    RAW.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            sys.executable, "-m", "kaggle",
            "datasets", "download",
            "-d", "mlg-ulb/creditcardfraud",
            "-p", str(RAW.parent),
            "--unzip",
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"kaggle download failed:\n{result.stderr}\n"
            "Ensure ~/.kaggle/kaggle.json exists with valid credentials."
        )
    print("Download complete.")

def load_and_validate() -> pd.DataFrame:
    """Load CSV and run basic sanity checks."""
    print(f"Loading {RAW} …")
    df = pd.read_csv(RAW)

    assert "Class" in df.columns, "Expected 'Class' column (0=legit, 1=fraud)"
    assert df.shape[1] >= 30, f"Expected ≥30 columns, got {df.shape[1]}"
    assert df["Class"].nunique() == 2, "Expected binary Class label"

    n_total = len(df)
    n_fraud = df["Class"].sum()
    pct_fraud = 100 * n_fraud / n_total
    print(
        f"{n_total:,} transactions | "
        f"{n_fraud:,} fraud ({pct_fraud:.3f}%)"
    )

    missing = df.isnull().sum().sum()
    if missing > 0:
        print(f"WARNING: {missing} missing values found — investigate before training")
    else:
        print("No missing values.")

    return df

def split_and_save(df: pd.DataFrame) -> None:
    """
    Stratified 60/20/20 split.
    test.csv is isolated immediately — it must never be used for
    training, validation, or threshold tuning.
    """
    PROCESSED.mkdir(parents=True, exist_ok=True)

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval,
        test_size=0.25,
        stratify=y_trainval,
        random_state=42,
    )

    for name, X_split, y_split in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        split_df = X_split.copy()
        split_df["Class"] = y_split.values
        out_path = PROCESSED / f"{name}.csv"
        split_df.to_csv(out_path, index=False)
        n = len(split_df)
        f = split_df["Class"].sum()
        print(f"{name:5s}: {n:,} rows | {f:,} fraud ({100*f/n:.3f}%) -> {out_path}")

    print("Splits saved. TEST SET IS NOW LOCKED — do not touch until final eval.")

def main() -> None:
    download_dataset()
    df = load_and_validate()
    split_and_save(df)

if __name__ == "__main__":
    main()

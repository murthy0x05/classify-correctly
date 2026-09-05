from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"

FP_COST_INR = 150.0
TRANSACTIONS_PER_1K = 1000


def load_test() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED / "test.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y


def evaluate_model(
    name: str,
    probs: np.ndarray,
    y_true: np.ndarray,
    threshold: float,
    avg_fraud_amount: float,
) -> dict:
    """Compute full metric suite at the given threshold."""
    preds = (probs >= threshold).astype(int)

    tp = int(((preds == 1) & (y_true == 1)).sum())
    fp = int(((preds == 1) & (y_true == 0)).sum())
    fn = int(((preds == 0) & (y_true == 1)).sum())
    tn = int(((preds == 0) & (y_true == 0)).sum())

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    roc_auc = roc_auc_score(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    n_total = len(y_true)
    n_per_1k = n_total / TRANSACTIONS_PER_1K
    fp_per_1k = fp / n_per_1k if n_per_1k > 0 else 0
    fn_per_1k = fn / n_per_1k if n_per_1k > 0 else 0

    total_fp_cost = fp * FP_COST_INR
    total_fn_cost = fn * avg_fraud_amount
    total_cost = total_fp_cost + total_fn_cost

    metrics = {
        "model": name,
        "threshold": round(threshold, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "false_positive_cost": {
            "fp_count": fp,
            "fp_per_1000_transactions": round(fp_per_1k, 2),
            "cost_per_fp_inr": FP_COST_INR,
            "total_fp_cost_inr": round(total_fp_cost, 2),
            "justification": (
                "Estimated at ~10 minutes of a support agent's time at a blended "
                "hourly cost of ₹900 (₹15/min × 10 min = ₹150 per false positive). "
                "This covers agent time + customer friction (drop in payment success rate)."
            ),
        },
        "false_negative_cost": {
            "fn_count": fn,
            "fn_per_1000_transactions": round(fn_per_1k, 2),
            "avg_fraud_amount_inr": round(avg_fraud_amount, 2),
            "total_fn_cost_inr": round(total_fn_cost, 2),
        },
        "total_expected_cost_inr": round(total_cost, 2),
    }

    print(f"\n{'='*60}")
    print(f"  {name} — Test Set Results")
    print(f"{'='*60}")
    print(f"  Threshold        : {threshold:.4f}")
    print(f"  Precision        : {precision:.4f}")
    print(f"  Recall           : {recall:.4f}")
    print(f"  F1               : {f1:.4f}")
    print(f"  ROC-AUC          : {roc_auc:.4f}")
    print(f"  PR-AUC           : {pr_auc:.4f}")
    print(f"  Confusion Matrix : TP={tp} FP={fp} FN={fn} TN={tn}")
    print(f"  FP per 1,000 txn : {fp_per_1k:.2f}")
    print(f"  FP cost (total)  : ₹{total_fp_cost:,.2f}")
    print(f"  FN cost (total)  : ₹{total_fn_cost:,.2f}")
    print(f"  Total exp. cost  : ₹{total_cost:,.2f}")

    return metrics


def plot_confusion_matrix(name: str, y_true: np.ndarray, preds: np.ndarray) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name} — Confusion Matrix (Test Set)")
    plt.tight_layout()
    fname = f"confusion_matrix_{name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURES / fname, dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / fname}")


def plot_pr_curve(name: str, y_true: np.ndarray, probs: np.ndarray, threshold: float) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    precision_vals, recall_vals, thresholds = precision_recall_curve(y_true, probs)
    pr_auc = average_precision_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(recall_vals, precision_vals, lw=2, label=f"PR curve (AUC = {pr_auc:.4f})")
    ax.axhline(y=sum(y_true) / len(y_true), color="gray", linestyle="--", label="Random baseline")

    # Mark the chosen threshold point
    idx = np.argmin(np.abs(thresholds - threshold))
    ax.scatter(recall_vals[idx], precision_vals[idx], s=120, zorder=5,
               color="red", label=f"Chosen threshold ({threshold:.3f})")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"{name} — Precision-Recall Curve (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"pr_curve_{name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURES / fname, dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / fname}")


def plot_roc_curve(name: str, y_true: np.ndarray, probs: np.ndarray) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fpr, tpr, _ = roc_curve(y_true, probs)
    roc_auc = roc_auc_score(y_true, probs)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"{name} — ROC Curve (Test Set)")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    fname = f"roc_curve_{name.lower().replace(' ', '_')}.png"
    plt.savefig(FIGURES / fname, dpi=150)
    plt.close()
    print(f"Saved: {FIGURES / fname}")


def main() -> None:
    print("Loading held-out test set …")
    X_test, y_test = load_test()
    y_arr = y_test.values

    # Load XGB meta (contains chosen threshold + avg fraud amount)
    with open(MODELS / "xgb_meta.json") as f:
        xgb_meta = json.load(f)

    avg_fraud_amount = xgb_meta["avg_fraud_amount_inr"]

        xgb_model = joblib.load(MODELS / "xgb_fraud.pkl")
    xgb_probs = xgb_model.predict_proba(X_test)[:, 1]
    xgb_threshold = xgb_meta["chosen_threshold"]

    xgb_metrics = evaluate_model(
        "XGBoost", xgb_probs, y_arr, xgb_threshold, avg_fraud_amount
    )
    plot_confusion_matrix("XGBoost", y_arr, (xgb_probs >= xgb_threshold).astype(int))
    plot_pr_curve("XGBoost", y_arr, xgb_probs, xgb_threshold)
    plot_roc_curve("XGBoost", y_arr, xgb_probs)

        lr_model = joblib.load(MODELS / "lr_baseline.pkl")
    lr_scaler = joblib.load(MODELS / "lr_scaler.pkl")
    X_test_scaled = X_test.copy()
    X_test_scaled[["Amount", "Time"]] = lr_scaler.transform(X_test[["Amount", "Time"]])
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]

    # LR threshold: cost-optimal on val probs (recomputed here as reference)
    # We use 0.5 default if no dedicated lr threshold was stored; load from meta if present
    lr_meta_path = MODELS / "lr_meta.json"
    if lr_meta_path.exists():
        with open(lr_meta_path) as f:
            lr_meta = json.load(f)
        lr_threshold = lr_meta["chosen_threshold"]
    else:
        lr_threshold = 0.5

    lr_metrics = evaluate_model(
        "Logistic Regression", lr_probs, y_arr, lr_threshold, avg_fraud_amount
    )
    plot_confusion_matrix("Logistic Regression", y_arr, (lr_probs >= lr_threshold).astype(int))
    plot_pr_curve("Logistic Regression", y_arr, lr_probs, lr_threshold)
    plot_roc_curve("Logistic Regression", y_arr, lr_probs)

        REPORTS.mkdir(parents=True, exist_ok=True)
    all_metrics = {
        "dataset": "Kaggle Credit Card Fraud Detection (mlg-ulb/creditcardfraud)",
        "test_set_size": len(y_arr),
        "test_fraud_count": int(y_arr.sum()),
        "test_fraud_rate_pct": round(100 * y_arr.sum() / len(y_arr), 4),
        "xgboost": xgb_metrics,
        "logistic_regression": lr_metrics,
    }
    with open(REPORTS / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nmetrics.json written to {REPORTS / 'metrics.json'}")
    print("All figures saved to reports/figures/")
    print("DONE — these are the final, honest test-set numbers.")


if __name__ == "__main__":
    main()

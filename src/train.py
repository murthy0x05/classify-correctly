from __future__ import annotations

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"

FP_COST_INR = 150.0  


def load_split(name: str) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(PROCESSED / f"{name}.csv")
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

def compute_avg_fraud_amount(y_train: pd.Series, X_train: pd.DataFrame) -> float:
    """Compute average fraud transaction amount from training data."""
    fraud_mask = y_train == 1
    avg = float(X_train.loc[fraud_mask, "Amount"].mean())
    print(f"Average fraud Amount (train set): ₹{avg:.2f}")
    return avg

def cost_optimal_threshold(
    probs: np.ndarray,
    y_true: np.ndarray,
    avg_fraud_amount: float,
    fp_cost: float = FP_COST_INR,
) -> dict:
    """
    Sweep thresholds [0.01, 0.99] and return the one that minimises
        total_cost = FP × fp_cost + FN × avg_fraud_amount
    Also returns the recall≥0.85 point as a secondary reference.
    """
    thresholds = np.linspace(0.01, 0.99, 500)
    best_threshold = 0.5
    best_cost = float("inf")
    best_cost_row: dict = {}

    recall_085_threshold = None
    recall_085_row: dict = {}

    for t in thresholds:
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())

        total_fraud = tp + fn
        total_legit = fp + tn

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / total_fraud if total_fraud > 0 else 0.0

        cost = fp * fp_cost + fn * avg_fraud_amount

        row = {
            "threshold": round(float(t), 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "fp": fp,
            "fn": fn,
            "total_cost_inr": round(cost, 2),
        }

        if cost < best_cost:
            best_cost = cost
            best_threshold = float(t)
            best_cost_row = row

        if recall >= 0.85:
            if recall_085_threshold is None or precision > recall_085_row.get("precision", 0):
                recall_085_threshold = float(t)
                recall_085_row = row

    return {
        "cost_optimal": best_cost_row,
        "recall_085_ref": recall_085_row if recall_085_row else None,
    }


def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    avg_fraud_amount: float,
) -> dict:
    print("\n=== Logistic Regression Baseline ===")

    scaler = StandardScaler()
    cols_to_scale = ["Amount", "Time"]
    X_train_s = X_train.copy()
    X_val_s = X_val.copy()
    X_train_s[cols_to_scale] = scaler.fit_transform(X_train[cols_to_scale])
    X_val_s[cols_to_scale] = scaler.transform(X_val[cols_to_scale])

    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        solver="lbfgs",
    )
    lr.fit(X_train_s, y_train)

    val_probs = lr.predict_proba(X_val_s)[:, 1]
    roc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    print(f"LR  val ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")

    threshold_info = cost_optimal_threshold(val_probs, y_val.values, avg_fraud_amount)
    chosen_t = threshold_info["cost_optimal"]["threshold"]
    print(f"LR  cost-optimal threshold: {chosen_t:.4f}")
    print(f"         → {threshold_info['cost_optimal']}")
    if threshold_info["recall_085_ref"]:
        print(f"LR  recall≥0.85 ref: {threshold_info['recall_085_ref']}")

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(lr, MODELS / "lr_baseline.pkl")
    joblib.dump(scaler, MODELS / "lr_scaler.pkl")

    lr_meta = {
        "chosen_threshold": chosen_t,
        "threshold_info": threshold_info,
        "val_roc_auc": roc,
        "val_pr_auc": pr_auc,
        "avg_fraud_amount_inr": avg_fraud_amount,
        "fp_cost_inr": FP_COST_INR,
    }
    with open(MODELS / "lr_meta.json", "w") as f:
        json.dump(lr_meta, f, indent=2)

    return {
        "model": lr,
        "scaler": scaler,
        "val_roc_auc": roc,
        "val_pr_auc": pr_auc,
        "threshold_info": threshold_info,
        "chosen_threshold": chosen_t,
    }


def _xgb_objective(
    trial: optuna.Trial,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    scale_pos_weight: float,
) -> float:
    """Optuna objective: maximise PR-AUC on validation set."""
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 800),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 1.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 1.0, log=True),
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "aucpr",
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    val_probs = model.predict_proba(X_val)[:, 1]
    return average_precision_score(y_val, val_probs)

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    avg_fraud_amount: float,
    n_trials: int = 40,
) -> dict:
    print("\n=== XGBoost (Optuna tuning, {n_trials} trials) ===")

    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / n_pos
    print(f"scale_pos_weight base = {spw:.1f} (neg/pos ratio)")

    study = optuna.create_study(direction="maximize")
    study.optimize(
        lambda trial: _xgb_objective(trial, X_train, y_train, X_val, y_val, spw),
        n_trials=n_trials,
        show_progress_bar=True,
    )

    best_params = study.best_params
    best_params.update(
        {
            "scale_pos_weight": spw,
            "eval_metric": "aucpr",
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
    )
    print(f"Best trial PR-AUC: {study.best_value:.4f}")
    print(f"Best params: {best_params}")

    xgb_model = xgb.XGBClassifier(**best_params)
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    val_probs = xgb_model.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, val_probs)
    pr_auc = average_precision_score(y_val, val_probs)
    print(f"XGB val ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f}")

    threshold_info = cost_optimal_threshold(val_probs, y_val.values, avg_fraud_amount)
    chosen_t = threshold_info["cost_optimal"]["threshold"]
    print(f"XGB cost-optimal threshold: {chosen_t:.4f}")
    print(f"         → {threshold_info['cost_optimal']}")
    if threshold_info["recall_085_ref"]:
        print(f"XGB recall≥0.85 ref: {threshold_info['recall_085_ref']}")

    MODELS.mkdir(parents=True, exist_ok=True)
    joblib.dump(xgb_model, MODELS / "xgb_fraud.pkl")

    meta = {
        "chosen_threshold": chosen_t,
        "threshold_info": threshold_info,
        "val_roc_auc": roc,
        "val_pr_auc": pr_auc,
        "avg_fraud_amount_inr": avg_fraud_amount,
        "fp_cost_inr": FP_COST_INR,
        "best_xgb_params": {k: v for k, v in best_params.items()
                            if k not in ("eval_metric",)},
    }
    with open(MODELS / "xgb_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    return {
        "model": xgb_model,
        "val_roc_auc": roc,
        "val_pr_auc": pr_auc,
        "threshold_info": threshold_info,
        "chosen_threshold": chosen_t,
    }


def main() -> None:
    print("Loading splits …")
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")

    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    avg_fraud_amount = compute_avg_fraud_amount(y_train, X_train)

    train_logistic_regression(X_train, y_train, X_val, y_val, avg_fraud_amount)
    train_xgboost(X_train, y_train, X_val, y_val, avg_fraud_amount, n_trials=40)

    print("\nTraining complete. Models saved to models/")
    print("Next: run src/evaluate.py to get test-set metrics.")

if __name__ == "__main__":
    main()

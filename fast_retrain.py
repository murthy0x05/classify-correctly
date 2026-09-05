import joblib, json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

ROOT = Path(__file__).resolve().parent
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
MODELS.mkdir(exist_ok=True)
FP_COST_INR = 150.0

def load_split(name):
    df = pd.read_csv(PROCESSED / f"{name}.csv")
    return df.drop(columns=["Class"]), df["Class"]

def cost_optimal_threshold(probs, y_true, avg_fraud_amount, fp_cost=FP_COST_INR):
    best_cost = float("inf")
    best_row, r085_row = {}, {}
    for t in np.linspace(0.01, 0.99, 500):
        preds = (probs >= t).astype(int)
        tp = int(((preds == 1) & (y_true == 1)).sum())
        fp = int(((preds == 1) & (y_true == 0)).sum())
        fn = int(((preds == 0) & (y_true == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        cost = fp * fp_cost + fn * avg_fraud_amount
        row = dict(threshold=round(float(t), 4), precision=round(prec, 4),
                   recall=round(rec, 4), fp=fp, fn=fn, total_cost_inr=round(cost, 2))
        if cost < best_cost:
            best_cost = cost; best_row = row
        if rec >= 0.85 and (not r085_row or prec > r085_row.get("precision", 0)):
            r085_row = row
    return {"cost_optimal": best_row, "recall_085_ref": r085_row or None}

print("Loading splits ...")
X_train, y_train = load_split("train")
X_val,   y_val   = load_split("val")
avg_fraud = float(X_train[y_train == 1]["Amount"].mean())
print(f"avg_fraud_amount: Rs.{avg_fraud:.2f}")

print("\n=== Logistic Regression ===")
scaler = StandardScaler()
cols = ["Amount", "Time"]
Xtr_s = X_train.copy(); Xv_s = X_val.copy()
Xtr_s[cols] = scaler.fit_transform(X_train[cols])
Xv_s[cols]  = scaler.transform(X_val[cols])

lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42, solver="lbfgs")
lr.fit(Xtr_s, y_train)

val_probs_lr = lr.predict_proba(Xv_s)[:, 1]
roc_lr = roc_auc_score(y_val, val_probs_lr)
pr_lr  = average_precision_score(y_val, val_probs_lr)
print(f"LR val ROC-AUC: {roc_lr:.4f} | PR-AUC: {pr_lr:.4f}")

ti_lr = cost_optimal_threshold(val_probs_lr, y_val.values, avg_fraud)
chosen_lr = ti_lr["cost_optimal"]["threshold"]
print(f"LR cost-optimal threshold: {chosen_lr:.4f}")
print(f"  -> {ti_lr['cost_optimal']}")
if ti_lr["recall_085_ref"]:
    print(f"  recall>=0.85 ref: {ti_lr['recall_085_ref']}")

joblib.dump(lr, MODELS / "lr_baseline.pkl")
joblib.dump(scaler, MODELS / "lr_scaler.pkl")
json.dump(
    {"chosen_threshold": chosen_lr, "threshold_info": ti_lr,
     "val_roc_auc": roc_lr, "val_pr_auc": pr_lr,
     "avg_fraud_amount_inr": avg_fraud, "fp_cost_inr": FP_COST_INR},
    open(MODELS / "lr_meta.json", "w"), indent=2,
)

print("\n=== XGBoost (pre-tuned best params) ===")
best_params = {
    "n_estimators": 643, "max_depth": 5,
    "learning_rate": 0.11576132524839076,
    "subsample": 0.816110823289517,
    "colsample_bytree": 0.8013536349642921,
    "min_child_weight": 5, "gamma": 0.8828227123103398,
    "reg_alpha": 0.00016570087817828708,
    "reg_lambda": 4.778134290857175e-08,
    "scale_pos_weight": 578.264406779661,
    "eval_metric": "aucpr", "random_state": 42, "n_jobs": -1, "verbosity": 0,
}
xgb_model = xgb.XGBClassifier(**best_params)
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

val_probs_xgb = xgb_model.predict_proba(X_val)[:, 1]
roc_xgb = roc_auc_score(y_val, val_probs_xgb)
pr_xgb  = average_precision_score(y_val, val_probs_xgb)
print(f"XGB val ROC-AUC: {roc_xgb:.4f} | PR-AUC: {pr_xgb:.4f}")

ti_xgb = cost_optimal_threshold(val_probs_xgb, y_val.values, avg_fraud)
chosen_xgb = ti_xgb["cost_optimal"]["threshold"]
print(f"XGB cost-optimal threshold: {chosen_xgb:.4f}")
print(f"  -> {ti_xgb['cost_optimal']}")
if ti_xgb["recall_085_ref"]:
    print(f"  recall>=0.85 ref: {ti_xgb['recall_085_ref']}")

joblib.dump(xgb_model, MODELS / "xgb_fraud.pkl")
json.dump(
    {"chosen_threshold": chosen_xgb, "threshold_info": ti_xgb,
     "val_roc_auc": roc_xgb, "val_pr_auc": pr_xgb,
     "avg_fraud_amount_inr": avg_fraud, "fp_cost_inr": FP_COST_INR,
     "best_xgb_params": {k: v for k, v in best_params.items() if k != "eval_metric"}},
    open(MODELS / "xgb_meta.json", "w"), indent=2,
)

print("\nAll models saved. Training complete.")
print("Next: python src/evaluate.py")

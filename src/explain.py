from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import shap


_FEATURE_ALIASES: dict[str, str] = {
    "Time": "Time since first transaction (seconds)",
    "Amount": "Transaction amount (₹)",
    **{f"V{i}": f"Behavioural pattern {i}" for i in range(1, 29)},
}

def _alias(feature: str) -> str:
    return _FEATURE_ALIASES.get(feature, feature)

def build_explainer(model: Any) -> shap.TreeExplainer:
    """Build a cached SHAP TreeExplainer for an XGBoost model."""
    return shap.TreeExplainer(model)

def explain_prediction(
    row: pd.Series | dict,
    explainer: shap.TreeExplainer,
    feature_names: list[str],
    top_n: int = 3,
) -> list[dict]:
    """
    Compute SHAP values for a single transaction and return the top-N
    contributing features in descending order of absolute SHAP value.

    Returns a list of dicts:
      {
        "feature": "V4",
        "display_name": "Behavioural pattern 4",
        "shap_value": -0.82,
        "direction": "increases" | "decreases",
        "magnitude": 0.82,
        "plain_english": "Behavioural pattern 4 strongly increases fraud risk",
      }
    """
    if isinstance(row, dict):
        row = pd.Series(row)

    X = row[feature_names].values.reshape(1, -1)
    shap_values = explainer.shap_values(X)[0]  

    contributions = []
    for i, fname in enumerate(feature_names):
        sv = float(shap_values[i])
        contributions.append(
            {
                "feature": fname,
                "display_name": _alias(fname),
                "shap_value": round(sv, 4),
                "direction": "increases" if sv > 0 else "decreases",
                "magnitude": round(abs(sv), 4),
            }
        )

    contributions.sort(key=lambda d: d["magnitude"], reverse=True)
    top = contributions[:top_n]

    for c in top:
        direction = c["direction"]
        c["plain_english"] = (
            f"{c['display_name']} {direction} fraud risk "
            f"(SHAP = {c['shap_value']:+.3f})"
        )

    return top

def top_reasons_text(contributions: list[dict]) -> list[str]:
    """Return a list of plain-English reason strings (for the audit log / UI)."""
    return [c["plain_english"] for c in contributions]

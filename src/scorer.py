from __future__ import annotations

import json
import logging
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
LOGS = ROOT / "logs"
AUDIT_LOG = LOGS / "audit_trail.jsonl"

# Model version tag — bump this if you retrain
MODEL_VERSION = "xgb-v1.0"

# Feature columns expected by the model (same order as training data)
FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ScoringResult:
    decision: str          # "FLAGGED" | "CLEARED"
    confidence: float      # model probability [0.0, 1.0]
    threshold: float       # threshold used for this decision
    top_reasons: list[str] # plain-English SHAP explanations (top 3)
    timestamp: str         # ISO-8601 UTC
    model_version: str
    input_hash: str        # sha256 of input for audit linkage
    error: str | None = None  # populated if scoring failed

    def to_dict(self) -> dict:
        return asdict(self)


class FraudScorer:
    """
    Thin wrapper around the trained XGBoost model + SHAP explainer.

    Strictly defensive: computes a risk score and reasons only.
    Does not generate, suggest, or facilitate fraudulent transactions.
    """

    def __init__(self) -> None:
        self._model: xgb.XGBClassifier | None = None
        self._meta: dict = {}
        self._explainer = None
        self._threshold: float = 0.5

    def load(self) -> "FraudScorer":
        """Load model, metadata, and build SHAP explainer."""
        import shap
        # Support both `python src/scorer.py` and `from src.scorer import ...`
        try:
            from src.explain import build_explainer
        except ImportError:
            from explain import build_explainer  # type: ignore[no-redef]

        meta_path = MODELS / "xgb_meta.json"
        model_path = MODELS / "xgb_fraud.pkl"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run `python src/train.py` first."
            )

        self._model = joblib.load(model_path)

        with open(meta_path) as f:
            self._meta = json.load(f)

        self._threshold = self._meta.get("chosen_threshold", 0.5)
        self._explainer = build_explainer(self._model)

        LOGS.mkdir(parents=True, exist_ok=True)
        logger.info(f"FraudScorer loaded. Model: {MODEL_VERSION} | Threshold: {self._threshold:.4f}")
        return self

    def score_transaction(self, transaction: dict) -> ScoringResult:
        """
        Score a single transaction.

        Args:
            transaction: dict with keys matching FEATURE_COLUMNS.
                         Missing features default to 0.0 with a warning.

        Returns:
            ScoringResult — always returns a result; populates `error` on failure.
        """
        import hashlib
        try:
            from src.explain import explain_prediction, top_reasons_text
        except ImportError:
            from explain import explain_prediction, top_reasons_text  # type: ignore[no-redef]

        timestamp = datetime.now(timezone.utc).isoformat()
        input_hash = hashlib.sha256(
            json.dumps(transaction, sort_keys=True).encode()
        ).hexdigest()[:16]

        try:
            row = self._build_feature_row(transaction)
            prob = float(self._model.predict_proba(row)[0, 1])
            decision = "FLAGGED" if prob >= self._threshold else "CLEARED"

            contributions = explain_prediction(
                pd.Series(row.iloc[0]),
                self._explainer,
                FEATURE_COLUMNS,
                top_n=3,
            )
            reasons = top_reasons_text(contributions)

            result = ScoringResult(
                decision=decision,
                confidence=round(prob, 6),
                threshold=self._threshold,
                top_reasons=reasons,
                timestamp=timestamp,
                model_version=MODEL_VERSION,
                input_hash=input_hash,
            )

        except Exception as exc:
            logger.error(f"Scoring failed for input_hash={input_hash}: {exc}")
            result = ScoringResult(
                decision="ERROR",
                confidence=0.0,
                threshold=self._threshold,
                top_reasons=[],
                timestamp=timestamp,
                model_version=MODEL_VERSION,
                input_hash=input_hash,
                error=str(exc),
            )

        self._append_audit(transaction, result)
        return result

    def score_batch(self, transactions: list[dict]) -> list[ScoringResult]:
        """Score a list of transactions."""
        return [self.score_transaction(t) for t in transactions]

    
    def _build_feature_row(self, transaction: dict) -> pd.DataFrame:
        """Convert a transaction dict to a properly-ordered DataFrame row."""
        missing = [f for f in FEATURE_COLUMNS if f not in transaction]
        if missing:
            logger.warning(f"Missing features defaulted to 0.0: {missing}")

        row = {f: float(transaction.get(f, 0.0)) for f in FEATURE_COLUMNS}
        return pd.DataFrame([row], columns=FEATURE_COLUMNS)

    def _append_audit(self, transaction: dict, result: ScoringResult) -> None:
        """Append-only audit log entry. Never raises — a failed write is logged."""
        try:
            AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
            entry = {
                "input_hash": result.input_hash,
                "timestamp": result.timestamp,
                "decision": result.decision,
                "confidence": result.confidence,
                "threshold": result.threshold,
                "top_reasons": result.top_reasons,
                "model_version": result.model_version,
                "error": result.error,
                # Truncated transaction for privacy — only log Amount
                "amount": transaction.get("Amount"),
            }
            with open(AUDIT_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.error(f"Audit write failed: {e}")



_scorer: FraudScorer | None = None


def load_scorer() -> FraudScorer:
    """Return a cached, loaded FraudScorer (singleton)."""
    global _scorer
    if _scorer is None:
        _scorer = FraudScorer().load()
    return _scorer


def score_transaction(transaction: dict) -> dict:
    """
    Module-level convenience function.
    Loads the scorer on first call (cached thereafter).

    Returns a plain dict (JSON-serialisable).
    """
    scorer = load_scorer()
    result = scorer.score_transaction(transaction)
    return result.to_dict()


if __name__ == "__main__":
    # Quick smoke test
    print("Loading scorer …")
    s = load_scorer()
    dummy = {f: 0.0 for f in FEATURE_COLUMNS}
    dummy["Amount"] = 500.0
    result = s.score_transaction(dummy)
    print(json.dumps(result.to_dict(), indent=2))

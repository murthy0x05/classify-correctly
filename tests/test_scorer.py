from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

def make_dummy_transaction(amount: float = 100.0) -> dict:
    return {f: 0.0 for f in FEATURE_COLUMNS} | {"Amount": amount}


class TestFeatureRow:
    """Test FraudScorer._build_feature_row"""

    def setup_method(self):
        from src.scorer import FraudScorer
        self.scorer = FraudScorer()

    def test_basic_row(self):
        txn = make_dummy_transaction(250.0)
        row = self.scorer._build_feature_row(txn)
        assert list(row.columns) == FEATURE_COLUMNS
        assert row.shape == (1, len(FEATURE_COLUMNS))
        assert float(row["Amount"].iloc[0]) == 250.0

    def test_missing_features_default_to_zero(self):
        txn = {"Amount": 100.0}  
        row = self.scorer._build_feature_row(txn)
        assert float(row["V1"].iloc[0]) == 0.0
        assert float(row["Amount"].iloc[0]) == 100.0

    def test_feature_order_preserved(self):
        txn = make_dummy_transaction()
        row = self.scorer._build_feature_row(txn)
        assert list(row.columns) == FEATURE_COLUMNS

class TestScoringResult:
    """Test ScoringResult dataclass"""

    def test_to_dict_contains_required_fields(self):
        from src.scorer import ScoringResult
        result = ScoringResult(
            decision="FLAGGED",
            confidence=0.91,
            threshold=0.35,
            top_reasons=["Reason A", "Reason B"],
            timestamp="2026-09-05T00:00:00+00:00",
            model_version="xgb-v1.0",
            input_hash="abc123",
        )
        d = result.to_dict()
        assert d["decision"] == "FLAGGED"
        assert d["confidence"] == 0.91
        assert "top_reasons" in d
        assert "timestamp" in d
        assert "model_version" in d
        assert "input_hash" in d

    def test_to_dict_is_json_serialisable(self):
        from src.scorer import ScoringResult
        result = ScoringResult(
            decision="CLEARED",
            confidence=0.12,
            threshold=0.35,
            top_reasons=["Normal transaction amount"],
            timestamp="2026-09-05T00:00:00+00:00",
            model_version="xgb-v1.0",
            input_hash="def456",
        )
        
        json.dumps(result.to_dict())

class TestAuditTrail:
    """Test that audit log is written correctly."""

    def test_audit_appends_on_each_call(self, tmp_path):
        from src.scorer import FraudScorer, FEATURE_COLUMNS

        scorer = FraudScorer()
        scorer._threshold = 0.5

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.9, 0.1]])
        scorer._model = mock_model

        mock_explainer = MagicMock()
        mock_explainer.shap_values.return_value = np.zeros(len(FEATURE_COLUMNS))
        scorer._explainer = mock_explainer

        import src.scorer as scorer_module
        original_audit = scorer_module.AUDIT_LOG
        scorer_module.AUDIT_LOG = tmp_path / "test_audit.jsonl"

        try:
            txn = make_dummy_transaction(100.0)
            scorer.score_transaction(txn)
            scorer.score_transaction(txn)

            lines = (tmp_path / "test_audit.jsonl").read_text().strip().split("\n")
            assert len(lines) == 2, f"Expected 2 audit entries, got {len(lines)}"

            entry = json.loads(lines[0])
            assert "decision" in entry
            assert "timestamp" in entry
            assert "input_hash" in entry
        finally:
            scorer_module.AUDIT_LOG = original_audit

class TestExplainModule:
    """Test explain.py aliases and structure."""

    def test_alias_amount(self):
        from src.explain import _alias
        assert "amount" in _alias("Amount").lower() or "₹" in _alias("Amount")

    def test_alias_v_features(self):
        from src.explain import _alias
        for i in range(1, 29):
            alias = _alias(f"V{i}")
            assert alias != f"V{i}", f"V{i} should have a human-readable alias"

    def test_top_reasons_text(self):
        from src.explain import top_reasons_text
        contributions = [
            {"plain_english": "Behavioural pattern 4 increases fraud risk (SHAP = +0.82)"},
            {"plain_english": "Transaction amount decreases fraud risk (SHAP = -0.31)"},
        ]
        reasons = top_reasons_text(contributions)
        assert len(reasons) == 2
        assert "SHAP" in reasons[0]

class TestCostOptimalThreshold:
    """Test threshold selection logic."""

    def test_picks_minimum_cost_threshold(self):
        from src.train import cost_optimal_threshold
        import numpy as np

        probs = np.array([0.0] * 100 + [1.0] * 10)
        y_true = np.array([0] * 100 + [1] * 10)
        avg_fraud_amount = 500.0

        result = cost_optimal_threshold(probs, y_true, avg_fraud_amount)
        cost_opt = result["cost_optimal"]

        assert cost_opt["fp"] == 0
        assert cost_opt["fn"] == 0
        assert cost_opt["total_cost_inr"] == 0.0

    def test_recall_085_ref_is_reported(self):
        from src.train import cost_optimal_threshold
        import numpy as np

        np.random.seed(42)
        probs = np.concatenate([
            np.random.uniform(0.0, 0.4, 900),  
            np.random.uniform(0.4, 1.0, 100),  
        ])
        y_true = np.array([0] * 900 + [1] * 100)

        result = cost_optimal_threshold(probs, y_true, 500.0)
        ref = result["recall_085_ref"]
        assert ref is not None
        assert ref["recall"] >= 0.84  

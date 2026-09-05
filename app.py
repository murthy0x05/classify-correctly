from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

st.set_page_config(
    page_title="Classify Correctly — Fraud Risk Detector",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
MODELS = ROOT / "models"
REPORTS = ROOT / "reports"
LOGS = ROOT / "logs"
PROCESSED = ROOT / "data" / "processed"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.risk-badge-flagged {
    display: inline-block;
    background: linear-gradient(135deg, #ff4b4b, #c0392b);
    color: white; border-radius: 8px; padding: 4px 12px;
    font-weight: 700; font-size: 13px; letter-spacing: 0.5px;
}
.risk-badge-cleared {
    display: inline-block;
    background: linear-gradient(135deg, #00c9a7, #00897b);
    color: white; border-radius: 8px; padding: 4px 12px;
    font-weight: 700; font-size: 13px; letter-spacing: 0.5px;
}
.risk-badge-error {
    display: inline-block;
    background: #f39c12; color: white; border-radius: 8px;
    padding: 4px 12px; font-weight: 700; font-size: 13px;
}
.metric-card {
    background: #1e2130; border-radius: 12px; padding: 20px;
    border: 1px solid #2d3246; margin-bottom: 12px;
}
.failure-box {
    background: #2a1f2d; border-left: 4px solid #f39c12;
    padding: 16px; border-radius: 8px; margin: 12px 0;
}
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading fraud detection model …")
def get_scorer():
    from src.scorer import load_scorer
    return load_scorer()


@st.cache_data
def load_test_batch(n: int = 200):
    """Load a stratified sample from the test set."""
    test_path = PROCESSED / "test.csv"
    if not test_path.exists():
        return None
    df = pd.read_csv(test_path)
    fraud = df[df["Class"] == 1].head(n // 4)
    legit = df[df["Class"] == 0].head(n - len(fraud))
    return pd.concat([fraud, legit]).sample(frac=1, random_state=42).reset_index(drop=True)


@st.cache_data
def load_metrics() -> dict | None:
    p = REPORTS / "metrics.json"
    if p.exists():
        return json.loads(p.read_text())
    return None


def badge(decision: str) -> str:
    if decision == "FLAGGED":
        return '<span class="risk-badge-flagged">🔴 FLAGGED</span>'
    elif decision == "CLEARED":
        return '<span class="risk-badge-cleared">🟢 CLEARED</span>'
    else:
        return '<span class="risk-badge-error">⚠️ ERROR</span>'


def confidence_bar(conf: float, decision: str) -> str:
    display_conf = conf if decision == "FLAGGED" else (1.0 - conf)
    pct = int(display_conf * 100)
    color = "#ff4b4b" if decision == "FLAGGED" else "#00c9a7"
    return f"""
    <div style="background:#2d3246;border-radius:4px;height:8px;width:100%">
      <div style="background:{color};width:{pct}%;height:8px;border-radius:4px;
                  transition:width 0.3s ease"></div>
    </div>
    <small style="color:#aaa">{pct}% confidence</small>
    """


with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/shield.png", width=64)
    st.title("Classify Correctly")
    st.caption("Credit Card Fraud Risk Detection")
    st.markdown("---")

    model_ready = (MODELS / "xgb_fraud.pkl").exists()
    data_ready = (PROCESSED / "test.csv").exists()

    st.markdown("**System Status**")
    st.markdown(f"{'🟢' if model_ready else '🔴'} Model: {'Loaded' if model_ready else 'Not trained'}")
    st.markdown(f"{'🟢' if data_ready else '🔴'} Data: {'Ready' if data_ready else 'Not prepared'}")

    if model_ready:
        meta_path = MODELS / "xgb_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            st.markdown("---")
            st.markdown("**Model Info**")
            st.markdown(f"- Threshold: `{meta.get('chosen_threshold', 'N/A'):.4f}`")
            st.markdown(f"- Val PR-AUC: `{meta.get('val_pr_auc', 'N/A'):.4f}`")

    st.markdown("---")
    st.markdown(
        "<small>Strictly defensive — scores risk only, "
        "never generates or facilitates fraud.</small>",
        unsafe_allow_html=True,
    )

if not model_ready:
    st.warning("⚠️ Model not trained yet. Run the pipeline first:")
    st.code(
        "pip install -r requirements.txt\n"
        "python src/data_prep.py\n"
        "python src/train.py\n"
        "python src/evaluate.py",
        language="bash",
    )
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    ["📊 Score Transactions", "⚠️ Failure Cases", "📈 Metrics Dashboard", "📋 Audit Trail"]
)

# Tab 1 — Score Transactions
with tab1:
    st.header("Score Transactions")

    source = st.radio(
        "Data source",
        ["Built-in test batch (200 transactions)", "Upload CSV"],
        horizontal=True,
    )

    df_to_score: pd.DataFrame | None = None
    has_labels = False

    if source == "Built-in test batch (200 transactions)":
        df_to_score = load_test_batch(200)
        if df_to_score is None:
            st.error("Test data not found. Run `python src/data_prep.py` first.")
        else:
            has_labels = "Class" in df_to_score.columns
            st.success(f"Loaded {len(df_to_score)} test transactions "
                       f"({int(df_to_score['Class'].sum())} fraud).")
    else:
        uploaded = st.file_uploader("Upload transaction CSV", type=["csv"])
        if uploaded:
            df_to_score = pd.read_csv(uploaded)
            has_labels = "Class" in df_to_score.columns
            st.success(f"Loaded {len(df_to_score)} transactions.")

    if df_to_score is not None and st.button("🚀 Run Scoring", type="primary"):
        scorer = get_scorer()

        with st.spinner("Scoring transactions …"):
            results = []
            for _, row in df_to_score.iterrows():
                txn = {f: float(row.get(f, 0.0)) for f in FEATURE_COLUMNS}
                r = scorer.score_transaction(txn).to_dict()
                results.append(r)

        st.markdown("---")

        # Running metrics
        if has_labels:
            y_true = df_to_score["Class"].values
            tp = sum(1 for r, l in zip(results, y_true)
                     if r["decision"] == "FLAGGED" and l == 1)
            fp = sum(1 for r, l in zip(results, y_true)
                     if r["decision"] == "FLAGGED" and l == 0)
            fn = sum(1 for r, l in zip(results, y_true)
                     if r["decision"] == "CLEARED" and l == 1)
            tn = sum(1 for r, l in zip(results, y_true)
                     if r["decision"] == "CLEARED" and l == 0)

            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Precision", f"{prec:.3f}")
            c2.metric("Recall", f"{rec:.3f}")
            c3.metric("F1", f"{f1:.3f}")
            c4.metric("False Positives", fp)
            c5.metric("False Negatives", fn)
            st.caption("Metrics computed on this batch only — see Metrics Dashboard for held-out test-set numbers.")

        flagged = [r for r in results if r["decision"] == "FLAGGED"]
        cleared = [r for r in results if r["decision"] == "CLEARED"]
        col_a, col_b = st.columns(2)
        col_a.metric("🔴 Flagged", len(flagged))
        col_b.metric("🟢 Cleared", len(cleared))

        # Results table
        st.markdown("### Transaction Results")
        rows_html = []
        for i, (r, (_, row)) in enumerate(zip(results, df_to_score.iterrows())):
            label_str = ""
            if has_labels:
                val = row.get("Class", -1)
                true_label = int(val) if pd.notna(val) else -1
                if true_label in (0, 1):
                    is_correct = (r["decision"] == "FLAGGED") == bool(true_label)
                    label_str = "✅" if is_correct else "❌"

            reason_short = r["top_reasons"][0][:55] if r["top_reasons"] else "N/A"
            rows_html.append(
                f"<tr>"
                f"<td style='padding:8px'>{i+1}</td>"
                f"<td>{badge(r['decision'])}</td>"
                f"<td style='padding:8px'>{confidence_bar(r['confidence'], r['decision'])}</td>"
                f"<td style='padding:8px'>₹{row.get('Amount', 0):.2f}</td>"
                f"<td style='padding:8px;font-size:12px;color:#aaa'>{reason_short}</td>"
                f"<td style='padding:8px'>{label_str}</td>"
                f"</tr>"
            )

        table_html = f"""
        <table style='width:100%;border-collapse:collapse;font-size:13px'>
          <thead>
            <tr style='border-bottom:1px solid #2d3246;color:#888'>
              <th style='padding:8px;text-align:left'>#</th>
              <th style='padding:8px;text-align:left'>Decision</th>
              <th style='padding:8px;text-align:left'>Confidence</th>
              <th style='padding:8px;text-align:left'>Amount</th>
              <th style='padding:8px;text-align:left'>Top Reason</th>
              <th style='padding:8px;text-align:left'>Correct?</th>
            </tr>
          </thead>
          <tbody>{''.join(rows_html)}</tbody>
        </table>
        """
        st.markdown(table_html, unsafe_allow_html=True)
        st.success("✅ All results logged to `logs/audit_trail.jsonl`")


# Tab 2 — Failure Cases
with tab2:
    st.header("⚠️ Failure Cases — Handled Gracefully")
    st.markdown(
        """
        Every production model has failure modes. This tab deliberately surfaces them.
        Both cases are logged to the audit trail — the system never silently drops them.
        """
    )

    test_df = load_test_batch(500)
    if test_df is None:
        st.error("Test data not found.")
    else:
        scorer = get_scorer()
        results = []
        for _, row in test_df.iterrows():
            txn = {f: float(row.get(f, 0.0)) for f in FEATURE_COLUMNS}
            r = scorer.score_transaction(txn).to_dict()
            results.append((r, row))

        # Find first FP and FN
        fp_case = next(
            ((r, row) for r, row in results
             if r["decision"] == "FLAGGED" and int(row["Class"]) == 0),
            None,
        )
        fn_case = next(
            ((r, row) for r, row in results
             if r["decision"] == "CLEARED" and int(row["Class"]) == 1),
            None,
        )

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🟠 False Positive")
            st.markdown("*A legitimate transaction wrongly flagged as fraud.*")
            if fp_case:
                r, row = fp_case
                st.markdown(f"""
<div class="failure-box">
<b>Decision:</b> {badge(r['decision'])}<br><br>
<b>Confidence:</b> {r['confidence']:.4f} (threshold: {r['threshold']:.4f})<br>
<b>Amount:</b> ₹{row.get('Amount', 0):.2f}<br>
<b>True Label:</b> ✅ Legitimate<br>
<b>Input Hash:</b> <code>{r['input_hash']}</code><br><br>
<b>Model's top reasons:</b><br>
{"<br>".join(f"• {reason}" for reason in r['top_reasons'])}<br><br>
<b>System response:</b><br>
Logged to audit trail. Routed to human review queue. Customer notified of temporary hold.
</div>
""", unsafe_allow_html=True)
                st.markdown("**What happened:** The model saw behavioral patterns (V-features) "
                            "that overlap with fraud signatures, but the transaction was genuine. "
                            "**Cost:** ₹150 manual review + potential customer friction.")
            else:
                st.info("No false positive found in this batch.")

        with col2:
            st.subheader("🔴 False Negative")
            st.markdown("*A fraudulent transaction that slipped through undetected.*")
            if fn_case:
                r, row = fn_case
                st.markdown(f"""
<div class="failure-box">
<b>Decision:</b> {badge(r['decision'])}<br><br>
<b>Confidence:</b> {r['confidence']:.4f} (threshold: {r['threshold']:.4f})<br>
<b>Amount:</b> ₹{row.get('Amount', 0):.2f}<br>
<b>True Label:</b> ❌ Fraud<br>
<b>Input Hash:</b> <code>{r['input_hash']}</code><br><br>
<b>Model's top reasons (all pointed away from fraud):</b><br>
{"<br>".join(f"• {reason}" for reason in r['top_reasons'])}<br><br>
<b>System response:</b><br>
Logged to audit trail. Flagged for post-hoc chargeback analysis. Used for future retraining.
</div>
""", unsafe_allow_html=True)
                st.markdown("**What happened:** This fraud had behavioral patterns that resembled "
                            "legitimate transactions. At the cost-optimal threshold, the model "
                            "traded some recall for lower false-positive cost. "
                            f"**Cost:** ₹{row.get('Amount', 0):.2f} (unrecovered fraud amount).")
            else:
                st.success("No false negatives in this batch — impressive but check your threshold!")

        st.markdown("---")
        st.markdown(
            "**Takeaway:** Both failure types are logged, auditable, and handled. "
            "The threshold was chosen to minimise total expected cost — not to hide failures."
        )


# Tab 3 — Metrics Dashboard
with tab3:
    st.header("📈 Held-Out Test Set Metrics")
    st.markdown(
        "> These numbers come from a test set the model **never saw** during training or tuning. "
        "Accuracy is not reported — it's near-meaningless at 0.17% fraud rate."
    )

    metrics = load_metrics()
    if metrics is None:
        st.warning("Metrics not yet computed. Run `python src/evaluate.py`.")
    else:
        xgb_m = metrics.get("xgboost", {})
        lr_m = metrics.get("logistic_regression", {})

        # Summary table
        st.markdown("### Model Comparison")
        comp_df = pd.DataFrame([
            {
                "Model": "XGBoost (primary)",
                "Precision": xgb_m.get("precision"),
                "Recall": xgb_m.get("recall"),
                "F1": xgb_m.get("f1"),
                "ROC-AUC": xgb_m.get("roc_auc"),
                "PR-AUC": xgb_m.get("pr_auc"),
                "Threshold": xgb_m.get("threshold"),
            },
            {
                "Model": "Logistic Regression (baseline)",
                "Precision": lr_m.get("precision"),
                "Recall": lr_m.get("recall"),
                "F1": lr_m.get("f1"),
                "ROC-AUC": lr_m.get("roc_auc"),
                "PR-AUC": lr_m.get("pr_auc"),
                "Threshold": lr_m.get("threshold"),
            },
        ])
        st.dataframe(comp_df, use_container_width=True)

        # FP cost detail
        fp_info = xgb_m.get("false_positive_cost", {})
        fn_info = xgb_m.get("false_negative_cost", {})

        st.markdown("### False-Positive Cost (XGBoost)")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("FP Count", fp_info.get("fp_count", "N/A"))
        c2.metric("FP / 1,000 txn", f"{fp_info.get('fp_per_1000_transactions', 0):.2f}")
        c3.metric("Total FP Cost", f"₹{fp_info.get('total_fp_cost_inr', 0):,.0f}")
        c4.metric("Total FN Cost", f"₹{fn_info.get('total_fn_cost_inr', 0):,.0f}")
        st.caption(fp_info.get("justification", ""))
        st.metric("Total Expected Cost", f"₹{xgb_m.get('total_expected_cost_inr', 0):,.0f}")

        # Plots
        st.markdown("### Confusion Matrix & Curves")
        figures_dir = REPORTS / "figures"
        img_files = {
            "Confusion Matrix (XGBoost)": figures_dir / "confusion_matrix_xgboost.png",
            "PR Curve (XGBoost)": figures_dir / "pr_curve_xgboost.png",
            "ROC Curve (XGBoost)": figures_dir / "roc_curve_xgboost.png",
        }
        cols = st.columns(3)
        for col, (title, img_path) in zip(cols, img_files.items()):
            if img_path.exists():
                col.markdown(f"**{title}**")
                col.image(str(img_path))
            else:
                col.warning(f"{title} not found.")


# Tab 4 — Audit Trail
with tab4:
    st.header("📋 Audit Trail")
    st.markdown(
        "Every scored transaction is appended here. Append-only — no deletions."
    )

    audit_path = LOGS / "audit_trail.jsonl"
    if not audit_path.exists():
        st.info("No audit entries yet. Score some transactions first.")
    else:
        lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
        entries = [json.loads(l) for l in lines if l.strip()]

        st.metric("Total Entries", len(entries))

        # Last 100 entries reversed (newest first)
        df_audit = pd.DataFrame(reversed(entries[-100:]))
        if not df_audit.empty:
            # Colour decision column
            def colour_decision(val):
                if val == "FLAGGED":
                    return "color: #ff4b4b; font-weight: bold"
                elif val == "CLEARED":
                    return "color: #00c9a7; font-weight: bold"
                return "color: #f39c12"

            st.dataframe(
                df_audit[["timestamp", "decision", "confidence", "threshold",
                           "amount", "model_version", "input_hash", "error"]]
                .style.applymap(colour_decision, subset=["decision"]),
                use_container_width=True,
            )
        st.caption("Showing last 100 entries. Full log at `logs/audit_trail.jsonl`.")

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.scorer import FEATURE_COLUMNS, load_scorer


ANSI_RED = "\033[91m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"
ANSI_RESET = "\033[0m"
ANSI_BOLD = "\033[1m"


def fmt_decision(decision: str) -> str:
    if decision == "FLAGGED":
        return f"{ANSI_RED}{ANSI_BOLD}🔴 FLAGGED{ANSI_RESET}"
    elif decision == "CLEARED":
        return f"{ANSI_GREEN}🟢 CLEARED{ANSI_RESET}"
    return f"{ANSI_YELLOW}⚠️  {decision}{ANSI_RESET}"


def run_batch(df: pd.DataFrame, scorer, has_labels: bool) -> None:
    tp = fp = fn = tn = 0
    results = []

    print(f"\n{'─'*80}")
    print(f"{'#':>4}  {'Decision':<14} {'Conf':>6}  {'Amount':>10}  Reason (top 1)")
    print(f"{'─'*80}")

    for i, (_, row) in enumerate(df.iterrows()):
        txn = row[FEATURE_COLUMNS].to_dict()
        result = scorer.score_transaction(txn).to_dict()
        results.append(result)

        reason = result["top_reasons"][0] if result["top_reasons"] else "N/A"
        amount = txn.get("Amount", 0.0)

        print(
            f"{i+1:>4}  {fmt_decision(result['decision']):<14} "
            f"{result['confidence']:>6.3f}  ₹{amount:>9.2f}  {reason[:60]}"
        )

        if has_labels:
            label = int(row["Class"])
            pred = 1 if result["decision"] == "FLAGGED" else 0
            if pred == 1 and label == 1:
                tp += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            else:
                tn += 1

    print(f"{'─'*80}\n")

    if has_labels:
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        total = len(results)
        print(f"{ANSI_BOLD}Batch Metrics (ground truth available):{ANSI_RESET}")
        print(f"  Transactions : {total}")
        print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
        print(f"  Precision    : {prec:.4f}")
        print(f"  Recall       : {rec:.4f}")
        print(f"  F1           : {f1:.4f}")

        # Show one explicit failure case
        fp_idx = next(
            (i for i, r in enumerate(results)
             if r["decision"] == "FLAGGED" and int(df.iloc[i]["Class"]) == 0),
            None,
        )
        fn_idx = next(
            (i for i, r in enumerate(results)
             if r["decision"] == "CLEARED" and int(df.iloc[i]["Class"]) == 1),
            None,
        )

        if fp_idx is not None:
            r = results[fp_idx]
            print(f"\n{ANSI_YELLOW}⚠️  False Positive (row {fp_idx+1}):{ANSI_RESET}")
            print(f"   Confidence: {r['confidence']:.4f} | Amount: ₹{df.iloc[fp_idx]['Amount']:.2f}")
            print(f"   Model saw: {r['top_reasons']}")
            print(f"   Action: Logged to audit trail. Flagged for human review.")

        if fn_idx is not None:
            r = results[fn_idx]
            print(f"\n{ANSI_YELLOW}⚠️  False Negative (row {fn_idx+1}):{ANSI_RESET}")
            print(f"   Confidence: {r['confidence']:.4f} | Amount: ₹{df.iloc[fn_idx]['Amount']:.2f}")
            print(f"   Note: This fraud transaction was not caught at this threshold.")
            print(f"   Action: Logged to audit trail. Represents model blind spot.")

    print(f"\n✅ Audit trail appended to logs/audit_trail.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fraud Risk Scorer — strictly defensive transaction screener"
    )
    parser.add_argument("--input", required=True, help="Path to CSV file of transactions")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Override decision threshold (default: cost-optimal from training)")
    parser.add_argument("--limit", type=int, default=50,
                        help="Max rows to score (default: 50)")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: {input_path} not found")
        sys.exit(1)

    print(f"Loading scorer …")
    scorer = load_scorer()

    if args.threshold is not None:
        scorer._threshold = args.threshold
        print(f"Threshold overridden to {args.threshold:.4f}")

    print(f"Loading {input_path} …")
    df = pd.read_csv(input_path)
    has_labels = "Class" in df.columns

    # Ensure all feature columns present
    missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols} — will default to 0.0")

    df = df.head(args.limit)
    print(f"Scoring {len(df)} transactions (has_labels={has_labels}) …\n")
    run_batch(df, scorer, has_labels)


if __name__ == "__main__":
    main()

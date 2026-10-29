
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main():
    results = pd.read_csv(DATA_DIR / "model_probability_audit.csv")
    y_true = results["actual_late"].astype(int)
    probabilities = results["late_probability"]

    rows = []

    for threshold in THRESHOLDS:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, predictions, labels=[0, 1]
        ).ravel()

        rows.append({
            "threshold": threshold,
            "accuracy": round(accuracy_score(y_true, predictions), 4),
            "precision": round(
                precision_score(y_true, predictions, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_true, predictions, zero_division=0), 4
            ),
            "f1_score": round(
                f1_score(y_true, predictions, zero_division=0), 4
            ),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_negatives": int(tn),
        })

    report = pd.DataFrame(rows)

    print("\nTHRESHOLD COMPARISON ON THE TEST SET")
    print(report.to_string(index=False))

    output_path = DATA_DIR / "threshold_comparison.csv"
    report.to_csv(output_path, index=False)
    print(f"\nSaved report to: {output_path}")


if __name__ == "__main__":
    main()

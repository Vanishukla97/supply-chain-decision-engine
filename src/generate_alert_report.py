
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "processed" / "final_test_predictions.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "alert_summary.csv"


def main():
    df = pd.read_csv(INPUT_PATH)

    # Count the four possible prediction outcomes.
    actual = df["actual_is_late"]
    predicted = df["predicted_late"]

    true_positives = int(((actual == 1) & (predicted == 1)).sum())
    false_positives = int(((actual == 0) & (predicted == 1)).sum())
    false_negatives = int(((actual == 1) & (predicted == 0)).sum())
    true_negatives = int(((actual == 0) & (predicted == 0)).sum())

    total_orders = len(df)
    alerted_orders = int((predicted == 1).sum())

    precision = (
        true_positives / alerted_orders
        if alerted_orders else 0
    )
    alert_rate = (
        alerted_orders / total_orders
        if total_orders else 0
    )

    summary = pd.DataFrame([
        {"metric": "Total orders evaluated", "value": total_orders},
        {"metric": "Orders flagged for late-delivery risk", "value": alerted_orders},
        {"metric": "Orders not flagged", "value": total_orders - alerted_orders},
        {"metric": "Correct late alerts (true positives)", "value": true_positives},
        {"metric": "False alarms", "value": false_positives},
        {"metric": "Missed late orders", "value": false_negatives},
        {"metric": "Correctly unflagged orders", "value": true_negatives},
        {"metric": "Alert rate (%)", "value": round(alert_rate * 100, 2)},
        {"metric": "Precision (%)", "value": round(precision * 100, 2)},
    ])

    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nALERT OPERATIONS SUMMARY")
    print("------------------------")
    print(summary.to_string(index=False))
    print(f"\nReport saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

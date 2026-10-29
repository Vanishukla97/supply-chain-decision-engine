
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = ROOT / "data" / "processed" / "final_test_predictions.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "prioritized_alerts.csv"

ALERT_THRESHOLD = 0.40


def assign_priority(probability):
    if probability >= 0.80:
        return "High"
    elif probability >= 0.60:
        return "Medium"
    elif probability >= ALERT_THRESHOLD:
        return "Low"
    else:
        return "No Alert"


def main():
    df = pd.read_csv(INPUT_PATH)

    df["alert_priority"] = df["late_probability"].apply(
        assign_priority
    )

    # Sort the highest-risk orders to the top.
    df = df.sort_values(
        by="late_probability",
        ascending=False
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nALERT PRIORITY SUMMARY")
    print("----------------------")
    print(
        df["alert_priority"]
        .value_counts()
        .reindex(
            ["High", "Medium", "Low", "No Alert"],
            fill_value=0
        )
        .to_string()
    )

    print("\nTop 10 orders requiring attention:")
    print(
        df[
            [
                "Order Id",
                "order_date",
                "late_probability",
                "alert_priority",
                "actual_is_late",
            ]
        ].head(10).to_string(index=False)
    )

    print(f"\nPrioritized alerts saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

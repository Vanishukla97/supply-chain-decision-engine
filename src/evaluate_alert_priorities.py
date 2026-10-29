
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = ROOT / "data" / "processed" / "prioritized_alerts.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "priority_evaluation.csv"

PRIORITY_ORDER = ["High", "Medium", "Low", "No Alert"]


def main():
    df = pd.read_csv(INPUT_PATH)

    summary = (
        df.groupby("alert_priority", observed=False)
        .agg(
            total_orders=("Order Id", "count"),
            actual_late_orders=("actual_is_late", "sum"),
            average_late_probability=("late_probability", "mean"),
        )
        .reindex(PRIORITY_ORDER)
        .reset_index()
    )

    summary["actual_late_rate_pct"] = (
        summary["actual_late_orders"]
        / summary["total_orders"]
        * 100
    ).round(2)

    summary["average_late_probability_pct"] = (
        summary["average_late_probability"] * 100
    ).round(2)

    summary = summary.drop(columns=["average_late_probability"])

    summary.to_csv(OUTPUT_PATH, index=False)

    print("\nPRIORITY CATEGORY EVALUATION")
    print("----------------------------")
    print(summary.to_string(index=False))

    print(f"\nEvaluation saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

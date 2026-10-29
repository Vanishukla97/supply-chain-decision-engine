
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    ROOT
    / "data"
    / "processed"
    / "final_test_predictions.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    # Group predictions into probability ranges.
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.01]
    labels = [
        "0–20%",
        "20–40%",
        "40–60%",
        "60–80%",
        "80–100%",
    ]

    df["probability_band"] = pd.cut(
        df["late_probability"],
        bins=bins,
        labels=labels,
        right=False,
        include_lowest=True,
    )

    summary = (
        df.groupby("probability_band", observed=False)
        .agg(
            total_orders=("actual_is_late", "size"),
            actual_late_rate=("actual_is_late", "mean"),
            average_predicted_probability=(
                "late_probability", "mean"
            ),
        )
        .reset_index()
    )

    summary["actual_late_rate_pct"] = (
        summary["actual_late_rate"] * 100
    ).round(2)

    summary["average_predicted_probability_pct"] = (
        summary["average_predicted_probability"] * 100
    ).round(2)

    print("\nPROBABILITY CALIBRATION CHECK")
    print("-----------------------------")
    print(
        summary[
            [
                "probability_band",
                "total_orders",
                "actual_late_rate_pct",
                "average_predicted_probability_pct",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()

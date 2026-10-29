
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "order_decision_dataset.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "final_test_predictions.csv"

FINAL_TEST_START = pd.Timestamp("2018-01-01")


def main():
    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    data["order_date"] = pd.to_datetime(data["order_date"])

    # Use the same January 2018 evaluation orders.
    test = data[data["order_date"] >= FINAL_TEST_START].copy()

    # A simple baseline: predict each shipping mode's late rate
    # using only data from before October 2017.
    training = data[data["order_date"] < pd.Timestamp("2017-10-01")]

    mode_late_rates = training.groupby("shipping_mode")["is_late"].mean()

    test["mode_probability"] = (
        test["shipping_mode"].map(mode_late_rates).fillna(
            training["is_late"].mean()
        )
    )

    # Keep the same 0.40 decision threshold as the full model.
    test["mode_prediction"] = (
        test["mode_probability"] >= 0.40
    ).astype(int)

    actual = test["is_late"]
    predicted = test["mode_prediction"]

    print("\nSHIPPING-MODE-ONLY BASELINE")
    print("---------------------------")
    print(f"Orders evaluated: {len(test)}")
    print(f"Accuracy:  {accuracy_score(actual, predicted):.4f}")
    print(
        f"Precision: {precision_score(actual, predicted, zero_division=0):.4f}"
    )
    print(f"Recall:    {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(actual, predicted, zero_division=0):.4f}")

    print("\nLate-delivery rate learned from training data:")
    print((mode_late_rates * 100).round(2).to_string())


if __name__ == "__main__":
    main()

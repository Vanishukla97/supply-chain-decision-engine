
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "order_decision_dataset.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "final_test_predictions.csv"

TRAIN_END = pd.Timestamp("2017-10-01")
TEST_START = pd.Timestamp("2018-01-01")

FALSE_ALARM_COST = 1
MISSED_LATE_COST = 3


def calculate_cost(actual, predicted):
    false_alarms = int(((actual == 0) & (predicted == 1)).sum())
    missed_late = int(((actual == 1) & (predicted == 0)).sum())

    return {
        "false_alarms": false_alarms,
        "missed_late_orders": missed_late,
        "total_cost_units": (
            false_alarms * FALSE_ALARM_COST
            + missed_late * MISSED_LATE_COST
        ),
    }


def main():
    data = pd.read_csv(DATA_PATH)
    full_predictions = pd.read_csv(PREDICTIONS_PATH)

    data["order_date"] = pd.to_datetime(data["order_date"])

    training = data[data["order_date"] < TRAIN_END]
    test = data[data["order_date"] >= TEST_START].copy()

    # Reproduce the shipping-mode-only baseline.
    mode_rates = training.groupby("shipping_mode")["is_late"].mean()
    test["mode_prediction"] = (
        test["shipping_mode"]
        .map(mode_rates)
        .fillna(training["is_late"].mean())
        >= 0.40
    ).astype(int)

    # Align full-model predictions to the same order IDs.
    full_predictions = full_predictions.set_index("Order Id")
    test["full_prediction"] = (
        test["Order Id"]
        .map(full_predictions["predicted_late"])
        .astype(int)
    )

    results = []

    for name, column in [
        ("Full model", "full_prediction"),
        ("Shipping-mode baseline", "mode_prediction"),
    ]:
        cost = calculate_cost(test["is_late"], test[column])
        results.append({"model": name, **cost})

    report = pd.DataFrame(results)
    report.to_csv(
        ROOT / "data" / "processed" / "operational_cost_comparison.csv",
        index=False,
    )

    print("\nILLUSTRATIVE OPERATIONAL COST COMPARISON")
    print("---------------------------------------")
    print(f"False-alarm cost: {FALSE_ALARM_COST}")
    print(f"Missed-late cost: {MISSED_LATE_COST}")
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()

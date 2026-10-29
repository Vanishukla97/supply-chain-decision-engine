
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "processed" / "order_decision_dataset.csv"
PREDICTIONS_PATH = ROOT / "data" / "processed" / "final_test_predictions.csv"

TRAIN_END = pd.Timestamp("2017-10-01")
TEST_START = pd.Timestamp("2018-01-01")

# Illustrative business-cost scenarios:
# Each tuple is (scenario name, false-alarm cost, missed-late cost).
SCENARIOS = [
    ("Balanced", 1, 3),
    ("Service-focused", 1, 5),
    ("Investigation-cost-focused", 3, 3),
]


def calculate_errors(actual, predicted):
    false_alarms = int(((actual == 0) & (predicted == 1)).sum())
    missed_late = int(((actual == 1) & (predicted == 0)).sum())

    return false_alarms, missed_late


def main():
    data = pd.read_csv(DATA_PATH)
    predictions = pd.read_csv(PREDICTIONS_PATH)

    data["order_date"] = pd.to_datetime(data["order_date"])

    # Use the same historical evaluation period for both strategies.
    test = data[data["order_date"] >= TEST_START].copy()

    # Shipping-mode baseline: learn rates from the training period only.
    training = data[data["order_date"] < TRAIN_END]
    mode_rates = training.groupby("shipping_mode")["is_late"].mean()

    test["baseline_prediction"] = (
        test["shipping_mode"]
        .map(mode_rates)
        .fillna(training["is_late"].mean())
        >= 0.40
    ).astype(int)

    # Match full-model predictions to the same orders.
    predictions = predictions.set_index("Order Id")
    test["full_model_prediction"] = (
        test["Order Id"]
        .map(predictions["predicted_late"])
        .astype(int)
    )

    strategies = {
        "Full model": "full_model_prediction",
        "Shipping-mode baseline": "baseline_prediction",
    }

    results = []

    for strategy_name, prediction_column in strategies.items():
        false_alarms, missed_late = calculate_errors(
            test["is_late"],
            test[prediction_column],
        )

        for scenario_name, false_alarm_cost, missed_late_cost in SCENARIOS:
            total_cost = (
                false_alarms * false_alarm_cost
                + missed_late * missed_late_cost
            )

            results.append({
                "scenario": scenario_name,
                "strategy": strategy_name,
                "false_alarm_cost": false_alarm_cost,
                "missed_late_cost": missed_late_cost,
                "false_alarms": false_alarms,
                "missed_late_orders": missed_late,
                "total_cost_units": total_cost,
            })

    report = pd.DataFrame(results)

    output_path = ROOT / "data" / "processed" / "cost_scenario_comparison.csv"
    report.to_csv(output_path, index=False)

    print("\nCOST SCENARIO COMPARISON")
    print("-----------------------")
    print(report.to_string(index=False))

    print("\nLowest-cost strategy in each scenario:")
    winners = report.loc[
        report.groupby("scenario")["total_cost_units"].idxmin(),
        ["scenario", "strategy", "total_cost_units"],
    ]
    print(winners.to_string(index=False))

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()


from pathlib import Path

import pandas as pd


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
INPUT_PATH = PROJECT_ROOT / "data" / "processed" / "order_decision_dataset.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

# Use a chronological split to simulate predicting future orders.
TEST_START_DATE = pd.Timestamp("2017-10-01")

# Candidate features: information assumed to be available at order placement.
CATEGORICAL_FEATURES = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment",
]

NUMERIC_FEATURES = [
    "scheduled_shipping_days",
    "order_value",
    "total_quantity",
    "item_count",
    "distinct_categories",
]

TARGET = "is_late"
DATE_COLUMN = "order_date"


def main():
    print("Loading order-level dataset...")
    df = pd.read_csv(INPUT_PATH)

    # Parse dates and validate the basic structure.
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")

    required_columns = (
        CATEGORICAL_FEATURES
        + NUMERIC_FEATURES
        + [TARGET, DATE_COLUMN]
    )

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")

    if df[DATE_COLUMN].isna().any():
        raise ValueError("Some order dates could not be parsed.")

    if df[required_columns].isna().any().any():
        raise ValueError("Missing values found in model data.")

    if not set(df[TARGET].unique()).issubset({0, 1}):
        raise ValueError("Target is expected to contain only 0 and 1.")

    # Split chronologically: earlier orders train the model,
    # later orders are held out for evaluation.
    train_df = df[df[DATE_COLUMN] < TEST_START_DATE].copy()
    test_df = df[df[DATE_COLUMN] >= TEST_START_DATE].copy()

    if train_df.empty or test_df.empty:
        raise ValueError("Training or test set is empty.")

    # Keep target and features separate.
    feature_columns = CATEGORICAL_FEATURES + NUMERIC_FEATURES

    X_train = train_df[feature_columns].copy()
    y_train = train_df[TARGET].copy()

    X_test = test_df[feature_columns].copy()
    y_test = test_df[TARGET].copy()

    # Save feature data and labels separately.
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train.to_csv(OUTPUT_DIR / "X_train.csv", index=False)
    y_train.to_csv(OUTPUT_DIR / "y_train.csv", index=False)
    X_test.to_csv(OUTPUT_DIR / "X_test.csv", index=False)
    y_test.to_csv(OUTPUT_DIR / "y_test.csv", index=False)

    print("\nCHRONOLOGICAL SPLIT")
    print(f"Test period starts: {TEST_START_DATE.date()}")
    print(f"Training orders: {len(X_train):,}")
    print(f"Test orders:     {len(X_test):,}")
    print(f"Training dates:  {train_df[DATE_COLUMN].min()} to "
          f"{train_df[DATE_COLUMN].max()}")
    print(f"Test dates:      {test_df[DATE_COLUMN].min()} to "
          f"{test_df[DATE_COLUMN].max()}")

    print("\nLATE RATE")
    print(f"Training: {y_train.mean():.2%}")
    print(f"Test:     {y_test.mean():.2%}")

    print("\nFEATURE CHECK")
    print(f"Categorical features: {len(CATEGORICAL_FEATURES)}")
    print(f"Numeric features:     {len(NUMERIC_FEATURES)}")
    print(f"Target:               {TARGET}")
    print("Outcome fields excluded from model inputs.")

    print("\nSaved X_train.csv, y_train.csv, X_test.csv, y_test.csv")


if __name__ == "__main__":
    main()

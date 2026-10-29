
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = (
    PROJECT_DIR
    / "data"
    / "processed"
    / "order_decision_dataset.csv"
)

def main():
    print("Loading order-level dataset...")

    df = pd.read_csv(DATA_FILE, low_memory=False)

    print(f"\nRows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    # These fields describe the final outcome.
    outcome_columns = [
        "delivery_status",
        "is_late",
    ]

    print("\n1. OUTCOME FIELDS — DO NOT USE AS PREDICTORS")
    for column in outcome_columns:
        if column in df.columns:
            print(f"{column}: present")

    # Candidate fields that may be available at order creation.
    candidate_columns = [
        "order_date",
        "shipping_mode",
        "market",
        "order_region",
        "order_country",
        "customer_segment",
        "scheduled_shipping_days",
        "order_value",
        "total_quantity",
        "item_count",
        "distinct_categories",
    ]

    print("\n2. CANDIDATE PREDICTOR AUDIT")

    for column in candidate_columns:
        if column not in df.columns:
            print(f"{column}: MISSING")
            continue

        missing_pct = df[column].isna().mean() * 100
        unique_count = df[column].nunique(dropna=True)

        print(
            f"{column}: "
            f"missing={missing_pct:.2f}%, "
            f"unique_values={unique_count:,}"
        )

    print("\n3. TARGET DISTRIBUTION")

    if "is_late" in df.columns:
        print(df["is_late"].value_counts(dropna=False).sort_index())

        print(
            f"\nLate rate: "
            f"{df['is_late'].mean() * 100:.2f}%"
        )

    print("\n4. DATE RANGE")

    if "order_date" in df.columns:
        dates = pd.to_datetime(df["order_date"], errors="coerce")

        print(f"Earliest order: {dates.min()}")
        print(f"Latest order:   {dates.max()}")
        print(f"Invalid dates:  {dates.isna().sum():,}")

    print("\nAudit complete.")
    print(
        "Reminder: confirm when each candidate feature becomes "
        "available before including it in a prediction model."
    )


if __name__ == "__main__":
    main()

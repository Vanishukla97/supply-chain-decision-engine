
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
    df = pd.read_csv(DATA_FILE, low_memory=False)

    print("1. DATASET SIZE")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns)}")

    print("\n2. UNIQUE ORDER CHECK")
    print(f"Unique order IDs: {df['Order Id'].nunique():,}")
    print(f"Duplicate order IDs: {df['Order Id'].duplicated().sum()}")

    print("\n3. MISSING VALUES")
    missing = df.isna().sum()
    print(missing[missing > 0].to_string() if missing.any() else "No missing values")

    print("\n4. OUTCOME LABEL CHECK")
    print(df["is_late"].value_counts(dropna=False).sort_index().to_string())
    print(f"Late rate: {df['is_late'].mean() * 100:.2f}%")

    print("\n5. BUSINESS MEASURE CHECKS")
    print(f"Orders with non-positive value: {(df['order_value'] <= 0).sum()}")
    print(f"Orders with non-positive quantity: {(df['total_quantity'] <= 0).sum()}")
    print(f"Orders with non-positive item count: {(df['item_count'] <= 0).sum()}")

    print("\n6. SHIPPING MODE SUMMARY")
    summary = df.groupby("shipping_mode").agg(
        orders=("Order Id", "nunique"),
        late_orders=("is_late", "sum"),
        late_rate=("is_late", "mean"),
        average_order_value=("order_value", "mean"),
    )

    summary["late_rate"] = (summary["late_rate"] * 100).round(2)
    print(summary.round(2).to_string())

    print("\n7. DECISION-TIME FEATURE WARNING")
    print("Delivery status and is_late are outcome fields, not predictors.")
    print("Actual delivery duration must not be used for an order-time prediction.")

    print("\nValidation completed.")

if __name__ == "__main__":
    main()

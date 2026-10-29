
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_FILE = PROJECT_DIR / "data" / "processed" / "order_decision_dataset.csv"

COLUMNS = [
    "Order Id",
    "Order Item Id",
    "order date (DateOrders)",
    "Shipping Mode",
    "Market",
    "Order Region",
    "Order Country",
    "Customer Segment",
    "Days for shipment (scheduled)",
    "Delivery Status",
    "Order Item Total",
    "Order Item Quantity",
    "Category Name",
]

def main():
    print("Loading required columns...")

    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        usecols=COLUMNS,
        low_memory=False,
    )

    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="coerce",
    )

    print(f"Input order-item rows: {len(df):,}")
    print(f"Unique orders: {df['Order Id'].nunique():,}")

    # Check that order-level fields agree across an order's items.
    consistency_columns = [
        "order date (DateOrders)",
        "Shipping Mode",
        "Market",
        "Order Region",
        "Order Country",
        "Customer Segment",
        "Days for shipment (scheduled)",
        "Delivery Status",
    ]

    inconsistent = (
        df.groupby("Order Id")[consistency_columns]
        .nunique(dropna=False)
        .gt(1)
        .any(axis=1)
    )

    inconsistent_count = int(inconsistent.sum())
    print(f"Orders with inconsistent order-level fields: {inconsistent_count:,}")

    if inconsistent_count:
        raise ValueError(
            "Some orders have inconsistent fields. Investigate them before aggregating."
        )

    # Canceled orders are not classified as late or not late.
    df = df[df["Delivery Status"] != "Shipping canceled"].copy()

    # Create a final-outcome label for historical analysis.
    df["is_late"] = (
        df["Delivery Status"] == "Late delivery"
    ).astype(int)

    # Aggregate item-level data into exactly one row per order.
    orders = (
        df.groupby("Order Id", as_index=False)
        .agg(
            order_date=("order date (DateOrders)", "first"),
            shipping_mode=("Shipping Mode", "first"),
            market=("Market", "first"),
            order_region=("Order Region", "first"),
            order_country=("Order Country", "first"),
            customer_segment=("Customer Segment", "first"),
            scheduled_shipping_days=(
                "Days for shipment (scheduled)", "first"
            ),
            delivery_status=("Delivery Status", "first"),
            order_value=("Order Item Total", "sum"),
            total_quantity=("Order Item Quantity", "sum"),
            item_count=("Order Item Id", "nunique"),
            distinct_categories=("Category Name", "nunique"),
            is_late=("is_late", "first"),
        )
    )

    if orders["Order Id"].duplicated().any():
        raise ValueError("Duplicate order IDs remain after aggregation.")

    # Keep the historical outcome for analysis, but do not use it
    # as an input feature in a future prediction model.
    orders.to_csv(OUTPUT_FILE, index=False)

    print("\nORDER-LEVEL DATASET")
    print(f"Orders after excluding cancellations: {len(orders):,}")
    print(f"Unique order IDs: {orders['Order Id'].nunique():,}")
    print(f"Late orders: {orders['is_late'].sum():,}")
    print(f"Historical late rate: {orders['is_late'].mean() * 100:.2f}%")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\nPREVIEW")
    print(orders.head(5).to_string(index=False))

if __name__ == "__main__":
    main()

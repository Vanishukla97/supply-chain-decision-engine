
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

COLUMNS = [
    "Order Id",
    "Order Item Id",
    "Delivery Status",
    "Shipping Mode",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Late_delivery_risk",
    "Order Item Total",
]


def main():
    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        usecols=COLUMNS,
        low_memory=False,
    )

    # Create one record per order for order-level comparisons.
    orders = (
        df.groupby("Order Id", as_index=False)
        .agg(
            status=("Delivery Status", "first"),
            shipping_mode=("Shipping Mode", "first"),
            actual_days=("Days for shipping (real)", "first"),
            scheduled_days=("Days for shipment (scheduled)", "first"),
            risk_label=("Late_delivery_risk", "first"),
            order_value=("Order Item Total", "sum"),
        )
    )

    print("1. DELIVERY STATUS COUNTS BY SHIPPING MODE")
    print(
        pd.crosstab(
            orders["Shipping Mode"] if "Shipping Mode" in orders.columns else orders["shipping_mode"],
            orders["status"],
        ).to_string()
    )

    print("\n2. ACTUAL VS SCHEDULED SHIPPING DAYS")
    print(
        orders.groupby("shipping_mode").agg(
            orders=("Order Id", "nunique"),
            avg_actual_days=("actual_days", "mean"),
            avg_scheduled_days=("scheduled_days", "mean"),
        ).round(2).to_string()
    )

    print("\n3. STATUS VS SHIPPING-DAY COMPARISON")
    print(
        orders.groupby("status").agg(
            orders=("Order Id", "nunique"),
            avg_actual_days=("actual_days", "mean"),
            avg_scheduled_days=("scheduled_days", "mean"),
        ).round(2).to_string()
    )

    print("\n4. FIRST CLASS SAMPLE")
    print(
        orders.loc[
            orders["shipping_mode"] == "First Class",
            ["status", "actual_days", "scheduled_days"],
        ].head(10).to_string(index=False)
    )

    print("\nInvestigation completed.")


if __name__ == "__main__":
    main()

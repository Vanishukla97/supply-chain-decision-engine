
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

COLUMNS = [
    "Order Id",
    "Order Item Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Shipping Mode",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Item Total",
    "Order Item Quantity",
    "Order Country",
    "Order Region",
]

def main():
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

    df["shipping date (DateOrders)"] = pd.to_datetime(
        df["shipping date (DateOrders)"],
        errors="coerce",
    )

    print("1. DATE QUALITY")
    print(
        "Invalid order dates:",
        df["order date (DateOrders)"].isna().sum(),
    )
    print(
        "Invalid shipping dates:",
        df["shipping date (DateOrders)"].isna().sum(),
    )

    print("\n2. TIME BETWEEN ORDER AND SHIPPING")
    df["order_to_ship_hours"] = (
        df["shipping date (DateOrders)"]
        - df["order date (DateOrders)"]
    ).dt.total_seconds() / 3600

    print(df["order_to_ship_hours"].describe().round(2).to_string())

    print("\n3. NEGATIVE ORDER-TO-SHIP INTERVALS")
    print((df["order_to_ship_hours"] < 0).sum())

    print("\n4. FIELDS AVAILABLE FOR HISTORICAL OUTCOME ANALYSIS")
    print(
        df.groupby("Delivery Status").agg(
            rows=("Order Item Id", "count"),
            unique_orders=("Order Id", "nunique"),
            avg_actual_days=("Days for shipping (real)", "mean"),
            avg_scheduled_days=("Days for shipment (scheduled)", "mean"),
        ).round(2).to_string()
    )

    print("\n5. IMPORTANT MODELING RULE")
    print("Do not use Delivery Status, Late_delivery_risk,")
    print("or actual shipping duration as predictors for a")
    print("prediction made at order placement.")

    print("\nDecision-time audit completed.")

if __name__ == "__main__":
    main()

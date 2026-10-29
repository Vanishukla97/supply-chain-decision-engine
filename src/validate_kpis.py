
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

COLUMNS = [
    "Order Id",
    "Order Item Id",
    "Delivery Status",
    "Shipping Mode",
    "Order Country",
    "Order Item Total",
]


def main():
    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        usecols=COLUMNS,
        low_memory=False,
    )

    print("1. CHECKING ORDER-LEVEL CONSISTENCY")

    for column in ["Delivery Status", "Shipping Mode", "Order Country"]:
        counts = df.groupby("Order Id")[column].nunique(dropna=False)
        inconsistent = (counts > 1).sum()

        print(f"Orders with inconsistent {column}: {inconsistent:,}")

    print("\n2. VALIDATING SHIPPING-MODE COUNTS")

    order_mode_counts = (
        df.groupby("Order Id")["Shipping Mode"].nunique(dropna=False)
    )

    print(f"Orders with multiple shipping modes: {(order_mode_counts > 1).sum():,}")

    print("\n3. COUNTRY RATES WITH AT LEAST 100 NON-CANCELED ORDERS")

    # Calculate one row per order to avoid counting each item as an order.
    orders = (
        df.groupby("Order Id", as_index=False)
        .agg(
            status=("Delivery Status", "first"),
            country=("Order Country", "first"),
            shipping_mode=("Shipping Mode", "first"),
        )
    )

    delivered = orders[orders["status"] != "Shipping canceled"].copy()
    delivered["is_late"] = delivered["status"].eq("Late delivery")

    country_summary = delivered.groupby("country").agg(
        orders=("Order Id", "nunique"),
        late_orders=("is_late", "sum"),
    )

    country_summary["late_rate_pct"] = (
        country_summary["late_orders"] / country_summary["orders"] * 100
    )

    country_summary = country_summary[
        country_summary["orders"] >= 100
    ].sort_values("late_rate_pct", ascending=False)

    print(country_summary.head(15).round(2).to_string())

    print("\n4. SHIPPING-MODE RATES")
    shipping = delivered.groupby("shipping_mode").agg(
        orders=("Order Id", "nunique"),
        late_orders=("is_late", "sum"),
    )
    shipping["late_rate_pct"] = (
        shipping["late_orders"] / shipping["orders"] * 100
    )

    print(shipping.sort_values("late_rate_pct", ascending=False).round(2).to_string())

    print("\nValidation checks completed.")


if __name__ == "__main__":
    main()

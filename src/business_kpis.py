
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"
OUTPUT_DIR = PROJECT_DIR / "data" / "processed"

COLUMNS = [
    "Order Id",
    "Order Item Id",
    "Delivery Status",
    "Shipping Mode",
    "Order Country",
    "Order Item Total",
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        usecols=COLUMNS,
        low_memory=False,
    )

    df["Order Item Total"] = pd.to_numeric(
        df["Order Item Total"], errors="coerce"
    )

    print(f"Order-item rows: {len(df):,}")

    # Combine all items belonging to the same order.
    orders = (
        df.groupby("Order Id", as_index=False)
        .agg(
            delivery_status=("Delivery Status", "first"),
            shipping_mode=("Shipping Mode", "first"),
            country=("Order Country", "first"),
            order_value=("Order Item Total", "sum"),
            item_count=("Order Item Id", "count"),
        )
    )

    print(f"Unique orders: {len(orders):,}")

    # Keep canceled orders separate from delivery performance.
    canceled = orders["delivery_status"].eq("Shipping canceled")
    delivered = orders.loc[~canceled].copy()

    delivered["is_late"] = delivered["delivery_status"].eq(
        "Late delivery"
    )

    print("\n=== OVERALL BUSINESS KPIs ===")
    print(f"Total orders: {len(orders):,}")
    print(f"Canceled orders: {canceled.sum():,}")
    print(f"Non-canceled orders: {len(delivered):,}")
    print(f"Late orders: {delivered['is_late'].sum():,}")

    late_rate = (
        delivered["is_late"].mean() * 100
        if not delivered.empty else 0
    )
    print(f"Late-delivery rate: {late_rate:.2f}%")

    late_value = delivered.loc[
        delivered["is_late"], "order_value"
    ].sum()
    print(f"Order value associated with late orders: {late_value:,.2f}")

    # Compare delivery performance across shipping modes.
    shipping = delivered.groupby("shipping_mode").agg(
        orders=("delivery_status", "size"),
        late_orders=("is_late", "sum"),
        order_value=("order_value", "sum"),
    )
    shipping["late_rate_pct"] = (
        shipping["late_orders"] / shipping["orders"] * 100
    )
    shipping = shipping.sort_values("late_rate_pct", ascending=False)

    print("\n=== PERFORMANCE BY SHIPPING MODE ===")
    print(shipping.round(2).to_string())

    # Compare delivery performance across destination countries.
    country = delivered.groupby("country").agg(
        orders=("delivery_status", "size"),
        late_orders=("is_late", "sum"),
        order_value=("order_value", "sum"),
    )
    country["late_rate_pct"] = (
        country["late_orders"] / country["orders"] * 100
    )
    country = country.sort_values(
        ["late_rate_pct", "orders"],
        ascending=[False, False],
    )

    print("\n=== TOP 10 COUNTRIES BY LATE RATE ===")
    print(country.head(10).round(2).to_string())

    # Save reusable outputs for later database and dashboard work.
    orders.to_csv(OUTPUT_DIR / "orders_summary.csv", index=False)
    shipping.to_csv(OUTPUT_DIR / "shipping_mode_kpis.csv")
    country.to_csv(OUTPUT_DIR / "country_kpis.csv")

    print("\nOutput files saved in data/processed/")
    print("Business KPI analysis completed!")


if __name__ == "__main__":
    main()

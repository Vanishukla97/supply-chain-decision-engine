
from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

COLUMNS = [
    "Order Id",
    "Order Item Id",
    "order date (DateOrders)",
    "shipping date (DateOrders)",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
    "Order Country",
    "Order State",
    "Shipping Mode",
    "Order Item Total",
    "Benefit per order",
]


def main():
    print("Loading the columns needed for structural analysis...")

    df = pd.read_csv(
        DATA_FILE,
        encoding="latin1",
        usecols=COLUMNS,
        low_memory=False,
    )

    print("\n1. ROW GRAIN AND IDENTIFIERS")
    print(f"Total rows: {len(df):,}")
    print(f"Unique order IDs: {df['Order Id'].nunique():,}")
    print(f"Unique order-item IDs: {df['Order Item Id'].nunique():,}")
    print(
        "Duplicate order-item IDs: "
        f"{df['Order Item Id'].duplicated().sum():,}"
    )

    items_per_order = df.groupby("Order Id").size()

    print(f"Orders with multiple rows: {(items_per_order > 1).sum():,}")
    print(f"Maximum rows for one order: {items_per_order.max()}")

    print("\n2. DATE COVERAGE")
    for column in [
        "order date (DateOrders)",
        "shipping date (DateOrders)",
    ]:
        dates = pd.to_datetime(df[column], errors="coerce")

        print(f"\n{column}")
        print(f"Unparseable dates: {dates.isna().sum():,}")
        print(f"Earliest date: {dates.min()}")
        print(f"Latest date: {dates.max()}")

    print("\n3. DELIVERY STATUS COUNTS")
    print(df["Delivery Status"].value_counts(dropna=False).to_string())

    print("\n4. DELIVERY STATUS VS. LATE-DELIVERY LABEL")
    print(
        pd.crosstab(
            df["Delivery Status"],
            df["Late_delivery_risk"],
            dropna=False,
        ).to_string()
    )

    print("\n5. DELIVERY PERFORMANCE BY SHIPPING MODE")
    summary = df.groupby("Shipping Mode").agg(
        rows=("Order Item Id", "size"),
        unique_orders=("Order Id", "nunique"),
        avg_actual_days=("Days for shipping (real)", "mean"),
        avg_scheduled_days=("Days for shipment (scheduled)", "mean"),
        avg_item_total=("Order Item Total", "mean"),
    )

    print(summary.round(2).to_string())

    report_path = PROJECT_DIR / "docs" / "structure_analysis.txt"
    report_path.parent.mkdir(exist_ok=True)

    with report_path.open("w", encoding="utf-8") as report:
        report.write("DATA STRUCTURE ANALYSIS\n\n")
        report.write(f"Rows: {len(df):,}\n")
        report.write(f"Unique orders: {df['Order Id'].nunique():,}\n")
        report.write(
            f"Unique order items: {df['Order Item Id'].nunique():,}\n"
        )
        report.write(
            "\nDELIVERY STATUS VS. LATE LABEL\n"
        )
        report.write(
            pd.crosstab(
                df["Delivery Status"],
                df["Late_delivery_risk"],
                dropna=False,
            ).to_string()
        )
        report.write("\n\nSHIPPING MODE SUMMARY\n")
        report.write(summary.round(2).to_string())

    print(f"\nReport saved to: {report_path}")


if __name__ == "__main__":
    main()

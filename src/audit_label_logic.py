
import pandas as pd
from pathlib import Path

DATA_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "raw"
    / "DataCoSupplyChainDataset.csv"
)

df = pd.read_csv(DATA_PATH, encoding="latin1", low_memory=False)

required = [
    "Order Id",
    "Delivery Status",
    "Late_delivery_risk",
    "Days for shipping (real)",
    "Days for shipment (scheduled)",
]

missing = [c for c in required if c not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

print("1. ROW-LEVEL DELIVERY STATUS")
print(
    pd.crosstab(
        df["Delivery Status"],
        df["Late_delivery_risk"],
        margins=True,
    )
)

print("\n2. ACTUAL SHIPPING DAYS VS SCHEDULED DAYS")
df["duration_exceeds_schedule"] = (
    df["Days for shipping (real)"]
    > df["Days for shipment (scheduled)"]
)

print(
    pd.crosstab(
        df["Delivery Status"],
        df["duration_exceeds_schedule"],
        normalize="index",
    ).round(3)
)

print("\n3. LATE-RISK LABEL BY SHIPPING MODE")
print(
    df.groupby("Shipping Mode", dropna=False)
    .agg(
        rows=("Order Id", "size"),
        unique_orders=("Order Id", "nunique"),
        risk_rate=("Late_delivery_risk", "mean"),
        avg_actual_days=("Days for shipping (real)", "mean"),
        avg_scheduled_days=(
            "Days for shipment (scheduled)", "mean"
        ),
    )
    .assign(
        risk_rate_pct=lambda x: (100 * x["risk_rate"]).round(2),
        avg_actual_days=lambda x: x["avg_actual_days"].round(2),
        avg_scheduled_days=lambda x: x["avg_scheduled_days"].round(2),
    )
    .drop(columns="risk_rate")
    .round(2)
)

print("\n4. ORDER-LEVEL CONSISTENCY")
print(
    df.groupby("Order Id")
    .agg(
        status_count=("Delivery Status", "nunique"),
        risk_label_count=("Late_delivery_risk", "nunique"),
    )
    .agg(
        orders_with_mixed_status=("status_count", lambda s: (s > 1).sum()),
        orders_with_mixed_risk_label=(
            "risk_label_count", lambda s: (s > 1).sum()
        ),
    )
)

print("\nAudit completed.")


from pathlib import Path

import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = PROJECT_DIR / "data" / "raw" / "DataCoSupplyChainDataset.csv"

df = pd.read_csv(
    DATA_FILE,
    encoding="latin1",
    nrows=5,
    low_memory=False,
)

print(f"Number of columns: {len(df.columns)}")

print("\n1. ALL COLUMN NAMES")
for column in df.columns:
    print(f"- {column}")

print("\n2. CANDIDATE DATE AND OPERATIONAL FIELDS")
keywords = [
    "date", "time", "ship", "delivery", "order",
    "customer", "market", "region", "country",
    "latitude", "longitude", "department", "category",
    "benefit", "sales", "profit", "status", "risk",
]

for column in df.columns:
    if any(word in column.lower() for word in keywords):
        print(f"- {column}: {df[column].dtype}")

print("\n3. SAMPLE DATE VALUES")
date_columns = [
    column for column in df.columns
    if "date" in column.lower() or "time" in column.lower()
]

for column in date_columns:
    print(f"\n{column}")
    print(df[column].head(3).to_string(index=False))

print("\nColumn inspection completed.")

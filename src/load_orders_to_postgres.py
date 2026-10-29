
import getpass
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import URL

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = PROJECT_ROOT / "data" / "processed" / "order_decision_dataset.csv"

if not CSV_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {CSV_PATH}")

password = getpass.getpass("Enter PostgreSQL postgres password: ")

connection_url = URL.create(
    drivername="postgresql+psycopg",
    username="postgres",
    password=password,
    host="localhost",
    port=5432,
    database="supply_chain_db",
)

engine = create_engine(connection_url)

try:
    print("Reading cleaned order dataset...")
    df = pd.read_csv(CSV_PATH)

    print(f"Rows in CSV: {len(df):,}")
    print("Loading records into PostgreSQL...")

    df.to_sql(
    name="orders",
    con=engine,
    schema="public",
    if_exists="replace",
    index=False,
    chunksize=1000,
    )

    with engine.connect() as connection:
        result = connection.execute(
            text("SELECT COUNT(*) FROM public.orders")
        ).scalar_one()

        print("\nData loaded successfully!")
        print(f"Rows in CSV: {len(df):,}")
        print(f"Rows in PostgreSQL: {result:,}")

        print("\nSample records:")
        sample = connection.execute(
            text("""
                SELECT "Order Id", order_date, shipping_mode,
                       order_value, is_late
                FROM public.orders
                ORDER BY "Order Id"
                LIMIT 5
            """)
        )

        for row in sample:
            print(row)

finally:
    engine.dispose()

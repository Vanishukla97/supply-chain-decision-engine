
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "order_decision_dataset.csv"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "late_delivery_model.joblib"

# Use only information available at order placement.
CATEGORICAL_FEATURES = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment",
]

NUMERIC_FEATURES = [
    "scheduled_shipping_days",
    "order_value",
    "total_quantity",
    "item_count",
    "distinct_categories",
]

FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES
TARGET = "is_late"

TRAIN_END = pd.Timestamp("2017-10-01")


def main():
    print("Loading order decision dataset...")
    df = pd.read_csv(DATA_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Train only on orders placed before October 1, 2017.
    train = df[df["order_date"] < TRAIN_END].copy()

    if train.empty:
        raise ValueError("No training rows found.")

    X_train = train[FEATURES]
    y_train = train[TARGET]

    print(f"Training orders: {len(train):,}")
    print(f"Training period ends before: {TRAIN_END.date()}")
    print(f"Training late rate: {y_train.mean():.2%}")

    # Preprocess numerical and categorical columns.
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])

    print("\nTraining deployment model...")
    model.fit(X_train, y_train)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nModel saved to: {MODEL_PATH}")
    print("Training completed successfully.")


if __name__ == "__main__":
    main()

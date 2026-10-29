
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"

CATEGORICAL = [
    "shipping_mode",
    "market",
    "order_region",
    "order_country",
    "customer_segment",
]

NUMERIC = [
    "scheduled_shipping_days",
    "order_value",
    "total_quantity",
    "item_count",
    "distinct_categories",
]


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC),
        ("categorical", categorical_pipeline, CATEGORICAL),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    results = X_test[["shipping_mode"]].copy()
    results["actual_late"] = y_test.to_numpy()
    results["late_probability"] = model.predict_proba(X_test)[:, 1]
    results["predicted_late_at_50pct"] = (
        results["late_probability"] >= 0.50
    ).astype(int)

    print("\nPREDICTED PROBABILITY DISTRIBUTION BY SHIPPING MODE")
    summary = results.groupby("shipping_mode").agg(
        orders=("actual_late", "size"),
        actual_late_rate=("actual_late", "mean"),
        mean_probability=("late_probability", "mean"),
        min_probability=("late_probability", "min"),
        median_probability=("late_probability", "median"),
        max_probability=("late_probability", "max"),
        predicted_late_rate=("predicted_late_at_50pct", "mean"),
    )

    summary["actual_late_rate"] *= 100
    summary["mean_probability"] *= 100
    summary["min_probability"] *= 100
    summary["median_probability"] *= 100
    summary["max_probability"] *= 100
    summary["predicted_late_rate"] *= 100

    print(summary.round(2).to_string())

    output_path = DATA_DIR / "model_probability_audit.csv"
    results.to_csv(output_path, index=False)
    print(f"\nSaved detailed results to: {output_path}")


if __name__ == "__main__":
    main()

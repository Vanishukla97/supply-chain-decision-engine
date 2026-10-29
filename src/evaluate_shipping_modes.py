
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, f1_score
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

FEATURES = CATEGORICAL + NUMERIC


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

    print("Training model on historical training data...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    # Attach outcomes to the corresponding test orders.
    results = X_test[["shipping_mode"]].copy()
    results["actual_late"] = y_test.to_numpy()
    results["predicted_late"] = predictions

    rows = []

    for mode, group in results.groupby("shipping_mode"):
        rows.append({
            "shipping_mode": mode,
            "orders": len(group),
            "actual_late_rate_pct": round(
                group["actual_late"].mean() * 100, 2
            ),
            "predicted_late_rate_pct": round(
                group["predicted_late"].mean() * 100, 2
            ),
            "precision": round(
                precision_score(
                    group["actual_late"],
                    group["predicted_late"],
                    zero_division=0,
                ), 3
            ),
            "recall": round(
                recall_score(
                    group["actual_late"],
                    group["predicted_late"],
                    zero_division=0,
                ), 3
            ),
            "f1_score": round(
                f1_score(
                    group["actual_late"],
                    group["predicted_late"],
                    zero_division=0,
                ), 3
            ),
        })

    report = pd.DataFrame(rows)

    print("\nTEST PERFORMANCE BY SHIPPING MODE")
    print(report.to_string(index=False))

    output_path = DATA_DIR / "shipping_mode_model_metrics.csv"
    report.to_csv(output_path, index=False)

    print(f"\nReport saved to: {output_path}")


if __name__ == "__main__":
    main()

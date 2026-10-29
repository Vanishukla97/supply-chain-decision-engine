
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = ROOT / "models"
MODEL_PATH = MODEL_DIR / "late_delivery_model.joblib"

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


def evaluate_model(name, model, X_test, y_test):
    predictions = model.predict(X_test)

    print(f"\n{'=' * 55}")
    print(name)
    print("=" * 55)
    print(f"Accuracy:  {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision: {precision_score(y_test, predictions, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_test, predictions, zero_division=0):.4f}")
    print(f"F1-score:  {f1_score(y_test, predictions, zero_division=0):.4f}")

    print("\nConfusion matrix:")
    print("Rows = actual class [0, 1]")
    print("Columns = predicted class [0, 1]")
    print(confusion_matrix(y_test, predictions))

    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))


def main():
    print("Loading prepared training and test data...")

    X_train = pd.read_csv(DATA_DIR / "X_train.csv")
    y_train = pd.read_csv(DATA_DIR / "y_train.csv").squeeze("columns")
    X_test = pd.read_csv(DATA_DIR / "X_test.csv")
    y_test = pd.read_csv(DATA_DIR / "y_test.csv").squeeze("columns")

    # Confirm that the feature columns are as expected.
    if list(X_train.columns) != FEATURES:
        raise ValueError("Training feature columns do not match the expected features.")

    if list(X_test.columns) != FEATURES:
        raise ValueError("Test feature columns do not match the expected features.")

    print(f"Training rows: {len(X_train):,}")
    print(f"Test rows:     {len(X_test):,}")
    print(f"Training late rate: {y_train.mean():.2%}")
    print(f"Test late rate:     {y_test.mean():.2%}")

    # Baseline: always predict the most frequent training class.
    baseline = DummyClassifier(strategy="most_frequent")
    baseline.fit(X_train, y_train)
    evaluate_model("BASELINE: MOST FREQUENT CLASS", baseline, X_test, y_test)

    # Preprocess numeric and categorical columns separately.
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

    # Logistic regression is a simple, interpretable starting model.
    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", LogisticRegression(max_iter=1000, random_state=42)),
    ])

    print("\nTraining logistic regression...")
    model.fit(X_train, y_train)

    evaluate_model("MODEL: LOGISTIC REGRESSION", model, X_test, y_test)


    # Save the trained model
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    print(f"\nSaved trained model to: {MODEL_PATH}")
    print("\nInitial evaluation completed.")
    print("Reminder: these results measure historical dataset performance,")
    print("not guaranteed real-world early-warning capability.")


if __name__ == "__main__":
    main()

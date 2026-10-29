
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "order_decision_dataset.csv"
OUTPUT_PATH = ROOT / "data" / "processed" / "final_test_predictions.csv"

THRESHOLD = 0.40
TRAIN_END = pd.Timestamp("2017-10-01")
FINAL_TEST_START = pd.Timestamp("2018-01-01")

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
TARGET = "is_late"


def main():
    df = pd.read_csv(DATA_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])

    # Train only on the period before October 2017.
    train = df[df["order_date"] < TRAIN_END].copy()

    # Keep January 2018 completely separate from model fitting.
    final_test = df[
        df["order_date"] >= FINAL_TEST_START
    ].copy()

    if train.empty or final_test.empty:
        raise ValueError("Training or final test data is empty.")

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
        ("classifier", LogisticRegression(
            max_iter=1000,
            random_state=42,
        )),
    ])

    model.fit(train[FEATURES], train[TARGET])

    # Apply the threshold selected on validation data.
    probabilities = model.predict_proba(
        final_test[FEATURES]
    )[:, 1]

    actual = final_test[TARGET].to_numpy()
    predicted = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        actual, predicted, labels=[0, 1]
    ).ravel()

    print("FINAL TEST EVALUATION")
    print("---------------------")
    print(f"Training orders:       {len(train):,}")
    print(f"Final test orders:     {len(final_test):,}")
    print(f"Chosen threshold:      {THRESHOLD:.2f}")
    print(f"Actual late rate:      {actual.mean():.2%}")
    print(f"Predicted late rate:   {predicted.mean():.2%}")
    print(f"Accuracy:              {accuracy_score(actual, predicted):.4f}")
    print(f"Precision:             {precision_score(actual, predicted, zero_division=0):.4f}")
    print(f"Recall:                {recall_score(actual, predicted, zero_division=0):.4f}")
    print(f"F1-score:
                  {f1_score(actual, predicted, zero_division=0):.4f}")

    print("\nCONFUSION MATRIX")
    print(f"True negatives:        {tn:,}")
    print(f"False alarms:          {fp:,}")
    print(f"Missed late orders:    {fn:,}")
    print(f"Late orders caught:    {tp:,}")

    print("\nCLASSIFICATION REPORT")
    print(classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["On-time/not-late", "Late"],
        zero_division=0,
    ))

    # Save results for the later decision and replay engine.
    predictions = pd.DataFrame({
        "Order Id": final_test["Order Id"].to_numpy(),
        "order_date": final_test["order_date"].to_numpy(),
        "actual_is_late": actual,
        "late_probability": probabilities,
        "alert_threshold": THRESHOLD,
        "predicted_late": predicted,
        "correct_prediction": (actual == predicted).astype(int),
    })

    predictions.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved final predictions to: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

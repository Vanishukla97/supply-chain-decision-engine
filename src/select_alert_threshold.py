
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "order_decision_dataset.csv"

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

TRAIN_END = pd.Timestamp("2017-10-01")
VALIDATION_END = pd.Timestamp("2018-01-01")

THRESHOLDS = [0.35, 0.40, 0.45, 0.50, 0.55, 0.60]


def main():
    df = pd.read_csv(DATA_PATH)
    df["order_date"] = pd.to_datetime(df["order_date"])

    train = df[df["order_date"] < TRAIN_END].copy()

    validation = df[
        (df["order_date"] >= TRAIN_END)
        & (df["order_date"] < VALIDATION_END)
    ].copy()

    final_test = df[df["order_date"] >= VALIDATION_END].copy()

    print("CHRONOLOGICAL DATA SPLIT")
    print(f"Training:   {len(train):,} orders")
    print(f"Validation: {len(validation):,} orders")
    print(f"Final test: {len(final_test):,} orders")

    if train.empty or validation.empty or final_test.empty:
        raise ValueError("One of the time periods is empty.")

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

    model.fit(train[FEATURES], train[TARGET])

    # Use validation data only to compare and choose a threshold.
    probabilities = model.predict_proba(validation[FEATURES])[:, 1]
    actual = validation[TARGET].to_numpy()

    rows = []

    for threshold in THRESHOLDS:
        predicted = (probabilities >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            actual, predicted, zero_division=0
        )
        recall = recall_score(
            actual, predicted, zero_division=0
        )

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1_score": f1_score(
                actual, predicted, zero_division=0
            ),
            "false_alarms": int(fp),
            "missed_late_orders": int(fn),
            "true_positives": int(tp),
            "true_negatives": int(tn),
        })

    report = pd.DataFrame(rows)

    print("\nVALIDATION THRESHOLD COMPARISON")
    print(report.round(4).to_string(index=False))
    
    # Balanced alert policy: require at least 80% recall,
    # then choose the eligible threshold with the highest precision.
    MIN_RECALL = 0.80

    eligible = report[report["recall"] >= MIN_RECALL]

    if eligible.empty:
        raise ValueError(
            "No tested threshold meets the minimum recall requirement."
        )

    best = eligible.sort_values(
        ["precision", "threshold"],
        ascending=[False, True],
    ).iloc[0]

    chosen_threshold = float(best["threshold"])

    print("\nBALANCED ALERT POLICY")
    print(f"Minimum required recall: {MIN_RECALL:.0%}")
    print(f"Selected threshold:      {chosen_threshold:.2f}")
    print(f"Validation precision:    {best['precision']:.4f}")
    print(f"Validation recall:       {best['recall']:.4f}")
    print(f"Validation F1-score:     {best['f1_score']:.4f}")
    print(f"False alarms:            {int(best['false_alarms']):,}")
    print(f"Missed late orders:      {int(best['missed_late_orders']):,}")

    print("\nJanuary 2018 is reserved for evaluation.")
    print("Do not use final-test outcomes to change the policy.")



if __name__ == "__main__":
    main()
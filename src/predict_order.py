
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "late_delivery_model.joblib"

ALERT_THRESHOLD = 0.40


def predict_order(order_details):
    model = joblib.load(MODEL_PATH)

    order = pd.DataFrame([order_details])

    probability = model.predict_proba(order)[0][1]
    alert = probability >= ALERT_THRESHOLD

    return {
        "late_delivery_probability": round(float(probability), 4),
        "alert_threshold": ALERT_THRESHOLD,
        "late_delivery_alert": bool(alert),
    }


if __name__ == "__main__":
    sample_order = {
        "shipping_mode": "Standard Class",
        "market": "USCA",
        "order_region": "West of USA",
        "order_country": "United States",
        "customer_segment": "Consumer",
        "scheduled_shipping_days": 4,
        "order_value": 150.0,
        "total_quantity": 2,
        "item_count": 2,
        "distinct_categories": 2,
    }

    result = predict_order(sample_order)

    print("\nOrder Prediction")
    print("----------------")
    print(f"Late delivery probability: {result['late_delivery_probability']:.2%}")
    print(f"Alert threshold: {result['alert_threshold']:.0%}")
    print(f"Late delivery alert: {result['late_delivery_alert']}")

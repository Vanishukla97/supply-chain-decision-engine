import altair as alt
import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration — must come before other Streamlit commands
st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom dashboard styling
st.markdown("""
<style>
    /* Main page */
    .stApp {
        background-color: #F5F7FB;
        color: #172033;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Main headings */
    h1 {
        color: #14213D !important;
        font-weight: 750 !important;
        letter-spacing: -0.8px;
    }

    h2, h3 {
        color: #14213D !important;
        font-weight: 650 !important;
    }

    /* Metric cards */
    [data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.04);
    }

    [data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-size: 14px;
    }

    [data-testid="stMetricValue"] {
        color: #14213D !important;
        font-weight: 700;
    }

    /* Data tables */
    [data-testid="stDataFrame"] {
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        overflow: hidden;
    }

    /* Section separators */
    hr {
        border-color: #DCE3ED;
        margin-top: 2rem;
        margin-bottom: 2rem;
    }

    /* Download buttons */
    .stDownloadButton button {
        border-radius: 8px;
        border: 1px solid #CBD5E1;
        font-weight: 600;
    }

    /* Informational messages */
    [data-testid="stAlert"] {
        border-radius: 10px;
    }
    /* Improve form label visibility */
    [data-testid="stWidgetLabel"] p,
    [data-testid="stWidgetLabel"] label {
        color: #334155 !important;
        font-weight: 600 !important;
    }

    /* Improve input text visibility */
    [data-baseweb="select"] *,
    [data-baseweb="input"] input {
        color: #F8FAFC !important;
    }

    /* Improve slider label visibility */
    [data-testid="stSlider"] label,
    [data-testid="stSlider"] p {
        color: #334155 !important;
    }

    /* Improve download button visibility */
    .stDownloadButton button {
        background-color: #FFFFFF !important;
        color: #14213D !important;
    }
</style>
""", unsafe_allow_html=True)

# Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data" / "processed"

ORDERS_FILE = DATA_DIR / "order_decision_dataset.csv"
PREDICTIONS_FILE = DATA_DIR / "final_test_predictions.csv"
ALERTS_FILE = DATA_DIR / "prioritized_alerts.csv"
COST_FILE = DATA_DIR / "cost_scenario_comparison.csv"

st.title("📦 Supply Chain Decision Intelligence")
st.caption(
    "Historical delivery-risk analysis, alert monitoring, "
    "and operational decision support"
)

st.info(
    "This dashboard presents historical evaluation results. "
    "It does not represent live shipment monitoring."
)

# Load datasets safely
@st.cache_data
def load_data(path):
    if path.exists():
        return pd.read_csv(path)
    return None

orders = load_data(ORDERS_FILE)
predictions = load_data(PREDICTIONS_FILE)
alerts = load_data(ALERTS_FILE)
costs = load_data(COST_FILE)

# Check required files
if orders is None:
    st.error(f"Orders dataset not found: {ORDERS_FILE}")
    st.stop()

if predictions is None:
    st.error(f"Prediction results not found: {PREDICTIONS_FILE}")
    st.stop()

# Overview
st.header("1. Overview")

total_orders = len(predictions)
late_rate = predictions["actual_is_late"].mean() * 100
alert_rate = predictions["predicted_late"].mean() * 100

c1, c2, c3 = st.columns(3)

c1.metric("Orders Evaluated", f"{total_orders:,}")
c2.metric("Actual Late-Delivery Rate", f"{late_rate:.2f}%")
c3.metric("Model Alert Rate", f"{alert_rate:.2f}%")

# Model performance
st.subheader("Model Performance")

if {"actual_is_late", "predicted_late"}.issubset(predictions.columns):
    actual = predictions["actual_is_late"].astype(int)
    predicted = predictions["predicted_late"].astype(int)

    true_positive = int(((actual == 1) & (predicted == 1)).sum())
    false_positive = int(((actual == 0) & (predicted == 1)).sum())
    false_negative = int(((actual == 1) & (predicted == 0)).sum())
    true_negative = int(((actual == 0) & (predicted == 0)).sum())

    precision = (
        true_positive / (true_positive + false_positive)
        if true_positive + false_positive else 0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative else 0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    p1, p2, p3 = st.columns(3)
    p1.metric("Precision", f"{precision * 100:.2f}%")
    p2.metric("Recall", f"{recall * 100:.2f}%")
    p3.metric("F1 Score", f"{f1 * 100:.2f}%")

    st.caption(
        f"True positives: {true_positive:,} | "
        f"False alarms: {false_positive:,} | "
        f"Missed late orders: {false_negative:,} | "
        f"Correctly unflagged: {true_negative:,}"
    )

st.divider()


# Risk monitoring
st.header("2. Risk Monitoring")

if alerts is not None:
    st.caption(
        "Explore historical orders by predicted late-delivery risk and alert priority."
    )

    filtered_alerts = alerts.copy()

    # Priority filter
    if "alert_priority" in filtered_alerts.columns:
        priority_col = "alert_priority"
    elif "priority" in filtered_alerts.columns:
        priority_col = "priority"
    else:
        priority_col = None

    if priority_col:
        priority_order = ["High", "Medium", "Low", "No Alert"]

        available_priorities = (
            filtered_alerts[priority_col]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        priority_options = ["All"] + [
            priority
            for priority in priority_order
            if priority in available_priorities
        ]

        selected_priority = st.selectbox(
            "Alert priority",
            priority_options,
            key="risk_priority_filter"
        )

        if selected_priority != "All":
            filtered_alerts = filtered_alerts[
                filtered_alerts[priority_col].astype(str)
                == selected_priority
            ]

    # Minimum predicted risk
    if "late_probability" in filtered_alerts.columns:
        min_risk = st.slider(
            "Minimum predicted late-delivery probability",
            min_value=0,
            max_value=100,
            value=40,
            step=5,
            format="%d%%",
            key="risk_probability_filter"
        )

        filtered_alerts = filtered_alerts[
            filtered_alerts["late_probability"] >= min_risk / 100
        ]

    # Search by Order ID
    if "Order Id" in filtered_alerts.columns:
        search_order = st.text_input(
            "Search by Order ID",
            placeholder="Enter an order ID...",
            key="risk_order_search"
        )

        if search_order.strip():
            filtered_alerts = filtered_alerts[
                filtered_alerts["Order Id"]
                .astype(str)
                .str.contains(
                    search_order.strip(),
                    case=False,
                    na=False,
                    regex=False
                )
            ]

    # Filter by actual delivery outcome
    if "actual_is_late" in filtered_alerts.columns:
        outcome_filter = st.selectbox(
            "Actual delivery outcome",
            ["All", "Late", "On time"],
            key="risk_outcome_filter"
        )

        if outcome_filter == "Late":
            filtered_alerts = filtered_alerts[
                filtered_alerts["actual_is_late"] == 1
            ]
        elif outcome_filter == "On time":
            filtered_alerts = filtered_alerts[
                filtered_alerts["actual_is_late"] == 0
            ]

    # Filter by prediction correctness
    if "correct_prediction" in filtered_alerts.columns:
        prediction_filter = st.selectbox(
            "Prediction result",
            ["All", "Correct", "Incorrect"],
            key="risk_prediction_filter"
        )

        if prediction_filter == "Correct":
            filtered_alerts = filtered_alerts[
                filtered_alerts["correct_prediction"] == 1
            ]
        elif prediction_filter == "Incorrect":
            filtered_alerts = filtered_alerts[
                filtered_alerts["correct_prediction"] == 0
            ]

    # Highest-risk orders first
    if "late_probability" in filtered_alerts.columns:
        filtered_alerts = filtered_alerts.sort_values(
            "late_probability",
            ascending=False
        )

    st.write(
        f"**{len(filtered_alerts):,} orders match your filters**"
    )

    # Prepare business-friendly table
    display_alerts = filtered_alerts.copy()

    display_alerts = display_alerts.rename(columns={
        "Order Id": "Order ID",
        "order_date": "Order Date",
        "actual_is_late": "Actual Outcome",
        "late_probability": "Late-Delivery Risk",
        "alert_threshold": "Alert Threshold",
        "predicted_late": "Alert Status",
        "correct_prediction": "Prediction Result",
        "alert_priority": "Priority",
        "priority": "Priority"
    })

    # Format probabilities
    if "Late-Delivery Risk" in display_alerts.columns:
        display_alerts["Late-Delivery Risk"] = (
            display_alerts["Late-Delivery Risk"].map(
                lambda value: f"{value:.1%}"
                if pd.notna(value) else "N/A"
            )
        )

    if "Alert Threshold" in display_alerts.columns:
        display_alerts["Alert Threshold"] = (
            display_alerts["Alert Threshold"].map(
                lambda value: f"{value:.0%}"
                if pd.notna(value) else "N/A"
            )
        )

    # Convert numeric flags into readable labels
    if "Actual Outcome" in display_alerts.columns:
        display_alerts["Actual Outcome"] = (
            display_alerts["Actual Outcome"]
            .map({1: "Late", 0: "On Time"})
            .fillna("Unknown")
        )

    if "Alert Status" in display_alerts.columns:
        display_alerts["Alert Status"] = (
            display_alerts["Alert Status"]
            .map({1: "Alert", 0: "No Alert"})
            .fillna("Unknown")
        )

    if "Prediction Result" in display_alerts.columns:
        display_alerts["Prediction Result"] = (
            display_alerts["Prediction Result"]
            .map({1: "Correct", 0: "Incorrect"})
            .fillna("Unknown")
        )

    # Arrange columns in a business-friendly order
    preferred_columns = [
        "Order ID",
        "Order Date",
        "Actual Outcome",
        "Late-Delivery Risk",
        "Alert Threshold",
        "Alert Status",
        "Prediction Result",
        "Priority"
    ]

    ordered_columns = [
        column
        for column in preferred_columns
        if column in display_alerts.columns
    ]

    # Keep any additional columns that may exist
    remaining_columns = [
        column
        for column in display_alerts.columns
        if column not in ordered_columns
    ]

    display_alerts = display_alerts[
        ordered_columns + remaining_columns
    ]

    st.dataframe(
        display_alerts,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "No Alert means predicted late-delivery probability is below the "
        "selected threshold; it does not guarantee on-time delivery."
    )

    st.download_button(
        "Download filtered risk records",
        data=filtered_alerts.to_csv(index=False).encode("utf-8"),
        file_name="filtered_risk_records.csv",
        mime="text/csv"
    )

else:
    st.warning("Prioritized alert file was not found.")


st.divider()

# Shipping analysis
st.header("3. Shipping Mode Analysis")

st.caption(
    "Scope: all available non-cancelled historical orders, "
    "January 2015–January 2018. This differs from the "
    "January 2018 model evaluation shown above."
)

if "shipping_mode" in orders.columns and "is_late" in orders.columns:
    shipping_analysis = (
        orders.groupby("shipping_mode")
        .agg(
            total_orders=("is_late", "size"),
            late_orders=("is_late", "sum"),
            late_rate=("is_late", "mean")
        )
        .reset_index()
    )

    shipping_analysis["late_rate_pct"] = (
        shipping_analysis["late_rate"] * 100
    )

    shipping_chart = (
    alt.Chart(shipping_analysis)
    .mark_bar()
    .encode(
        x=alt.X(
            "shipping_mode:N",
            title="Shipping Mode",
            sort="-y",
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y(
            "late_rate_pct:Q",
            title="Late-delivery rate (%)",
            scale=alt.Scale(domain=[0, 100]),
            axis=alt.Axis(values=[0, 20, 40, 60, 80, 100])
        ),
        tooltip=[
            alt.Tooltip("shipping_mode:N", title="Shipping mode"),
            alt.Tooltip("total_orders:Q", title="Total orders"),
            alt.Tooltip("late_orders:Q", title="Late orders"),
            alt.Tooltip(
                "late_rate_pct:Q",
                title="Late rate",
                format=".2f"
            )
        ]
    )
    .properties(height=350)
)

    st.altair_chart(shipping_chart, use_container_width=True)

    st.dataframe(
        shipping_analysis[
            ["shipping_mode", "total_orders", "late_orders", "late_rate_pct"]
        ].round(2),
        use_container_width=True,
        hide_index=True
    )

st.caption(
    "Shipping-mode rates describe the historical dataset and should not "
    "be interpreted as guaranteed future delivery outcomes."
)

st.divider()


# Decision analysis
st.header("4. Decision Analysis")

if costs is not None:
    st.caption(
        "Compare strategies under illustrative cost assumptions. "
        "Cost units are not actual financial amounts."
    )

    required_columns = {
        "scenario",
        "strategy",
        "total_cost_units",
        "false_alarms",
        "missed_late_orders",
    }

    if required_columns.issubset(costs.columns):
        st.dataframe(
            costs,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Operational Cost by Scenario")

        cost_chart = (
            alt.Chart(
                costs,
                title="Compare strategies across cost scenarios"
            )
            .mark_bar()
            .encode(
                x=alt.X(
                    "scenario:N",
                    title="Cost scenario",
                    sort=[
                        "Balanced",
                        "Service-focused",
                        "Investigation-cost-focused",
                    ],
                    axis=alt.Axis(
                        labelAngle=0,
                        labelLimit=260,
                        labelOverlap=False,
                    ),
                ),
                xOffset=alt.XOffset("strategy:N"),
                y=alt.Y(
                    "total_cost_units:Q",
                    title="Total cost (illustrative units)",
                    scale=alt.Scale(zero=True),
                    axis=alt.Axis(format=",.0f")
                ),
                color=alt.Color(
                    "strategy:N",
                    title="Strategy",
                    scale=alt.Scale(
                        domain=[
                            "Full model",
                            "Shipping-mode baseline"
                        ],
                        range=["#2563EB", "#94A3B8"]
                    )
                ),
                tooltip=[
                    "scenario:N",
                    "strategy:N",
                    "false_alarms:Q",
                    "missed_late_orders:Q",
                    "total_cost_units:Q"
                ]
            )
            .properties(height=380)
        )

        st.altair_chart(
            cost_chart,
            use_container_width=True
        )
    else:
        st.warning(
            "The cost scenario file is missing required columns."
        )
else:
    st.warning(
        "Cost scenario comparison file was not found."
    )

st.divider()


st.caption(
    "Project note: Historical data spans 2015–2018. Model results and "
    "alert thresholds are experimental and require further validation "
    "before operational use."
)

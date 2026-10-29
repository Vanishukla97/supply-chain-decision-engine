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

MODEL_SUMMARY_FILE = DATA_DIR / "model_performance_summary.csv"
PRIORITY_FILE = DATA_DIR / "priority_evaluation.csv"
SHIPPING_FILE = DATA_DIR / "shipping_mode_kpis.csv"
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

# Load compact, aggregate datasets only
@st.cache_data
def load_data(path):
    if path.exists():
        return pd.read_csv(path)
    return None

model_summary = load_data(MODEL_SUMMARY_FILE)
priorities = load_data(PRIORITY_FILE)
shipping = load_data(SHIPPING_FILE)
costs = load_data(COST_FILE)

if model_summary is None or model_summary.empty:
    st.error(f"Model summary file not found or empty: {MODEL_SUMMARY_FILE}")
    st.stop()

if shipping is None or shipping.empty:
    st.error(f"Shipping KPI file not found or empty: {SHIPPING_FILE}")
    st.stop()

# Overview
st.header("1. Overview")
m = model_summary.iloc[0]

c1, c2, c3 = st.columns(3)
c1.metric("Orders Evaluated", f"{int(m['total_orders']):,}")
c2.metric("Actual Late-Delivery Rate", f"{m['actual_late_rate_pct']:.2f}%")
c3.metric("Model Alert Rate", f"{m['alert_rate_pct']:.2f}%")

st.subheader("Model Performance")
p1, p2, p3 = st.columns(3)
p1.metric("Precision", f"{m['precision_pct']:.2f}%")
p2.metric("Recall", f"{m['recall_pct']:.2f}%")
p3.metric("F1 Score", f"{m['f1_pct']:.2f}%")

st.caption(
    f"True positives: {int(m['true_positives']):,} | "
    f"False alarms: {int(m['false_positives']):,} | "
    f"Missed late orders: {int(m['false_negatives']):,} | "
    f"Correctly unflagged: {int(m['true_negatives']):,}"
)

st.divider()


# Risk monitoring
st.header("2. Risk Monitoring")
st.caption(
    "Aggregate historical outcomes by alert priority. Individual order records "
    "are intentionally excluded from the deployed dashboard."
)

if priorities is not None and not priorities.empty:
    required_priority_columns = {
        "alert_priority",
        "total_orders",
        "actual_late_orders",
        "actual_late_rate_pct",
        "average_late_probability_pct",
    }

    if required_priority_columns.issubset(priorities.columns):
        priority_order = ["High", "Medium", "Low", "No Alert"]
        priorities = priorities.copy()
        priorities["alert_priority"] = pd.Categorical(
            priorities["alert_priority"],
            categories=priority_order,
            ordered=True,
        )
        priorities = priorities.sort_values("alert_priority")

        r1, r2, r3 = st.columns(3)
        r1.metric("Orders in Priority Summary", f"{int(priorities['total_orders'].sum()):,}")
        r2.metric("Actual Late Orders", f"{int(priorities['actual_late_orders'].sum()):,}")
        r3.metric(
            "Average Predicted Risk",
            f"{priorities['average_late_probability_pct'].mean():.2f}%",
        )

        priority_chart = (
            alt.Chart(priorities)
            .mark_bar()
            .encode(
                x=alt.X("alert_priority:N", title="Alert priority", sort=priority_order),
                y=alt.Y(
                    "actual_late_rate_pct:Q",
                    title="Actual late-delivery rate (%)",
                    scale=alt.Scale(domain=[0, 100]),
                ),
                tooltip=[
                    alt.Tooltip("alert_priority:N", title="Alert priority"),
                    alt.Tooltip("total_orders:Q", title="Orders", format=","),
                    alt.Tooltip("actual_late_orders:Q", title="Actual late orders", format=","),
                    alt.Tooltip("actual_late_rate_pct:Q", title="Actual late rate", format=".2f"),
                    alt.Tooltip(
                        "average_late_probability_pct:Q",
                        title="Average predicted risk (%)",
                        format=".2f",
                    ),
                ],
            )
            .properties(height=340)
        )
        st.altair_chart(priority_chart, use_container_width=True)

        st.dataframe(
            priorities[
                [
                    "alert_priority",
                    "total_orders",
                    "actual_late_orders",
                    "actual_late_rate_pct",
                    "average_late_probability_pct",
                ]
            ].rename(
                columns={
                    "alert_priority": "Alert Priority",
                    "total_orders": "Total Orders",
                    "actual_late_orders": "Actual Late Orders",
                    "actual_late_rate_pct": "Actual Late Rate (%)",
                    "average_late_probability_pct": "Average Predicted Risk (%)",
                }
            ).round(2),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "These are grouped historical evaluation results, not live alerts. "
            "Priority-level rates may vary across datasets and time periods."
        )
    else:
        st.warning("The priority evaluation file is missing required summary columns.")
else:
    st.warning("Priority evaluation summary was not found.")

st.divider()


# Shipping analysis
st.header("3. Shipping Mode Analysis")
st.caption(
    "Scope: all available non-cancelled historical orders, January 2015–January 2018. "
    "This differs from the January 2018 model evaluation shown above."
)

shipping_analysis = shipping.copy()
required_shipping_columns = {
    "shipping_mode", "orders", "late_orders", "late_rate_pct"
}

if required_shipping_columns.issubset(shipping_analysis.columns):
    shipping_analysis = shipping_analysis.rename(columns={"orders": "total_orders"})

    shipping_chart = (
        alt.Chart(shipping_analysis)
        .mark_bar()
        .encode(
            x=alt.X(
                "shipping_mode:N",
                title="Shipping Mode",
                sort="-y",
                axis=alt.Axis(labelAngle=0),
            ),
            y=alt.Y(
                "late_rate_pct:Q",
                title="Late-delivery rate (%)",
                scale=alt.Scale(domain=[0, 100]),
                axis=alt.Axis(values=[0, 20, 40, 60, 80, 100]),
            ),
            tooltip=[
                alt.Tooltip("shipping_mode:N", title="Shipping mode"),
                alt.Tooltip("total_orders:Q", title="Total orders", format=","),
                alt.Tooltip("late_orders:Q", title="Late orders", format=","),
                alt.Tooltip("late_rate_pct:Q", title="Late rate (%)", format=".2f"),
            ],
        )
        .properties(height=350)
    )

    st.altair_chart(shipping_chart, use_container_width=True)
    st.dataframe(
        shipping_analysis[
            ["shipping_mode", "total_orders", "late_orders", "late_rate_pct"]
        ].rename(
            columns={
                "shipping_mode": "Shipping Mode",
                "total_orders": "Total Orders",
                "late_orders": "Late Orders",
                "late_rate_pct": "Late Rate (%)",
            }
        ).round(2),
        use_container_width=True,
        hide_index=True,
    )
else:
    st.warning("The shipping KPI file is missing required summary columns.")

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

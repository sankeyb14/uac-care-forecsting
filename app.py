import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="UAC Care Forecasting",
    page_icon="📊",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_dataset.csv")
    df['Date'] = pd.to_datetime(df['Date'])
    return df.sort_values('Date')

df = load_data()

# Dashboard heading
st.title("UAC Program | Care Load Intelligence")
st.caption(
    "Historical analytics and predictive forecasting "
    "for Unaccompanied Alien Children"
)

st.divider()

# KPI metrics
latest_care = df['Children in HHS Care'].iloc[-1]
latest_transfers = df['Children transferred out of CBP custody'].iloc[-1]
latest_discharges = df['Children discharged from HHS Care'].iloc[-1]

col1, col2, col3 = st.columns(3)

col1.metric("Latest HHS Care Load", f"{latest_care:,.0f}")
col2.metric("Latest Transfers", f"{latest_transfers:,.0f}")
col3.metric("Latest Discharges", f"{latest_discharges:,.0f}")

st.divider()

# Historical care load chart
st.subheader("Historical HHS Care Load")

fig = px.line(
    df,
    x='Date',
    y='Children in HHS Care',
    title='Children Under HHS Care Over Time'
)

st.plotly_chart(fig, use_container_width=True)

# Transfers vs discharges
st.subheader("Transfers vs Discharges")

flow_data = df.melt(
    id_vars=['Date'],
    value_vars=[
        'Children transferred out of CBP custody',
        'Children discharged from HHS Care'
    ],
    var_name='Category',
    value_name='Children'
)

fig2 = px.line(
    flow_data,
    x='Date',
    y='Children',
    color='Category'
)

st.plotly_chart(fig2, use_container_width=True)
# FORECASTING MODULE

st.divider()
st.header("Predictive Forecasting")
st.caption(
    "Forecast origin: December 21, 2025 — the latest observation in the dataset."
)
# Load corrected post-break model comparison
results = pd.read_csv("post_break_model_comparison.csv")

st.subheader("Post-Break Model Performance Comparison")
st.caption(
    "Model performance evaluated using data from January 2025 onward "
    "with a chronological train/test split."
)

st.dataframe(results, use_container_width=True)

# Forecast horizon selector
horizon = st.select_slider(
    "Select Forecast Horizon (observations)",
    options=[1, 7, 14],
    value=7
)

# Latest observed values
latest_hhs = df['Children in HHS Care'].iloc[-1]

# Corrected post-break discharge forecast
discharge_forecast = pd.read_csv(
    "post_break_future_discharge_forecast.csv"
)

selected_discharge = discharge_forecast[
    discharge_forecast["Horizon"] == horizon
]

latest_discharge = selected_discharge[
    "Forecast_Discharge_Count"
].iloc[0]

# Corrected post-break validation errors
post_break_validation = pd.read_csv(
    "post_break_multi_horizon_validation.csv"
)

error_ranges = dict(
    zip(
        post_break_validation["Horizon"],
        post_break_validation["MAE"]
    )
)

error = error_ranges[horizon]

# MODEL TOGGLE
# Latest observed values
latest_hhs = df['Children in HHS Care'].iloc[-1]

# Corrected post-break discharge forecast
discharge_forecast = pd.read_csv(
    "post_break_future_discharge_forecast.csv"
)

latest_discharge = discharge_forecast[
    "Forecast_Discharge_Count"
].iloc[0]


# ------------------------------------------
# MODEL TOGGLE
# ------------------------------------------

future_model_forecasts = pd.read_csv(
    "post_break_future_model_forecasts.csv"
)

toggle_validation = pd.read_csv(
    "post_break_toggle_model_validation.csv"
)


model_options = {
    "Naive Persistence": "Naive_Persistence",
    "Random Forest": "Random_Forest",
    "Gradient Boosting": "Gradient_Boosting"
}


selected_model = st.selectbox(
    "Select Forecast Model",
    list(model_options.keys())
)

selected_column = model_options[
    selected_model
]


# ------------------------------------------
# SELECT FORECAST
# ------------------------------------------

selected_forecast = future_model_forecasts[
    future_model_forecasts["Horizon"] == horizon
]

selected_forecast_value = selected_forecast[
    selected_column
].iloc[0]


# ------------------------------------------
# MODEL-SPECIFIC VALIDATION ERROR
# ------------------------------------------

selected_validation = toggle_validation[
    (toggle_validation["Model"] == selected_model) &
    (toggle_validation["Horizon"] == horizon)
]

model_mae = selected_validation[
    "MAE"
].iloc[0]


# ------------------------------------------
# FORECAST DATA
# ------------------------------------------

forecast_data = pd.DataFrame({
    "Step": [horizon],

    "HHS Forecast": [
        selected_forecast_value
    ],

    "Lower Range": [
        max(
            0,
            selected_forecast_value - model_mae
        )
    ],

    "Upper Range": [
        selected_forecast_value + model_mae
    ],

    "Discharge Forecast": [
        latest_discharge
    ]
})


# ------------------------------------------
# FUTURE HHS CARE LOAD FORECAST
# ------------------------------------------

st.subheader(
    "Future HHS Care Load Forecast"
)

st.metric(
    f"{selected_model} Forecast",
    f"{selected_forecast_value:,.0f}"
)


fig3 = px.scatter(
    forecast_data,
    x="Step",
    y="HHS Forecast",
    title=(
        f"{selected_model} Forecast "
        f"— Horizon {horizon}"
    ),
    error_y=(
        forecast_data["Upper Range"]
        - forecast_data["HHS Forecast"]
    ),
    error_y_minus=(
        forecast_data["HHS Forecast"]
        - forecast_data["Lower Range"]
    )
)


st.plotly_chart(
    fig3,
    use_container_width=True
)


# ------------------------------------------
# DISCHARGE DEMAND FORECAST
# ------------------------------------------

st.subheader(
    "Discharge Demand Forecast"
)

st.dataframe(
    forecast_data[
        ["Step", "Discharge Forecast"]
    ],
    use_container_width=True
)


# ------------------------------------------
# EXPLANATION
# ------------------------------------------

st.caption(
    "Forecasts use the selected model over the "
    "chosen future observation horizon. Error bars "
    "represent the historical MAE for the selected "
    "model and horizon from post-break validation. "
    "They are shown as uncertainty indicators, not "
    "calibrated statistical confidence or prediction "
    "intervals. Model performance should be interpreted "
    "alongside the validation results."
)

# HISTORICAL PERIOD COMPARISON

st.divider()
st.header("Historical Period Comparison")

df['Period'] = df['Date'].apply(
    lambda x: 'Before January 2025'
    if x < pd.Timestamp('2025-01-01')
    else 'January 2025 Onwards'
)

period_summary = df.groupby('Period').agg({
    'Children in HHS Care': 'mean',
    'Children transferred out of CBP custody': 'mean',
    'Children discharged from HHS Care': 'mean'
}).round(2)

st.subheader("Average Daily Reporting Values")
st.dataframe(period_summary, use_container_width=True)

fig4 = px.bar(
    period_summary.reset_index(),
    x='Period',
    y='Children in HHS Care',
    title='Average HHS Care Population: Before vs After January 2025',
    text_auto=True
)

st.plotly_chart(fig4, use_container_width=True)
# MONTHLY TREND ANALYSIS

st.divider()
st.header("Monthly Operational Trends")

monthly = (
    df.set_index('Date')
    .resample('ME')
    .agg({
        'Children in HHS Care': 'mean',
        'Children transferred out of CBP custody': 'mean',
        'Children discharged from HHS Care': 'mean'
    })
    .reset_index()
)

monthly['Month'] = monthly['Date'].dt.strftime('%b %Y')

fig5 = px.line(
    monthly,
    x='Date',
    y='Children in HHS Care',
    markers=True,
    title='Monthly Average HHS Care Population'
)

st.plotly_chart(fig5, use_container_width=True)

fig6 = px.line(
    monthly,
    x='Date',
    y=[
        'Children transferred out of CBP custody',
        'Children discharged from HHS Care'
    ],
    markers=True,
    title='Monthly Average Transfers vs Discharges'
)

st.plotly_chart(fig6, use_container_width=True)

st.subheader("Monthly Summary")
st.dataframe(monthly, use_container_width=True)
# DATA QUALITY & REPORTING COVERAGE

st.divider()
st.header("Data Quality & Reporting Coverage")

expected_days = (
    df['Date'].max() - df['Date'].min()
).days + 1

actual_records = df['Date'].nunique()
missing_days = expected_days - actual_records

coverage = (actual_records / expected_days) * 100

col1, col2, col3 = st.columns(3)

col1.metric("Expected Calendar Days", f"{expected_days:,}")
col2.metric("Recorded Observations", f"{actual_records:,}")
col3.metric("Reporting Coverage", f"{coverage:.1f}%")

st.warning(
    f"{missing_days} calendar days have no recorded observations. "
    "Forecasts should therefore be interpreted cautiously."
)

# Weekday reporting pattern
weekday_summary = (
    df.assign(Weekday=df['Date'].dt.day_name())
    .groupby('Weekday')
    .size()
    .reindex([
        'Monday', 'Tuesday', 'Wednesday',
        'Thursday', 'Friday', 'Saturday', 'Sunday'
    ], fill_value=0)
    .reset_index(name='Observations')
)

fig7 = px.bar(
    weekday_summary,
    x='Weekday',
    y='Observations',
    title='Reporting Frequency by Weekday'
)

st.plotly_chart(fig7, use_container_width=True)

st.dataframe(weekday_summary, use_container_width=True)
# AUTOMATED BUSINESS INSIGHTS

st.divider()
st.header("Operational Insights")

avg_hhs_before = before_hhs = df[
    df['Date'] < '2025-01-01'
]['Children in HHS Care'].mean()

avg_hhs_after = after_hhs = df[
    df['Date'] >= '2025-01-01'
]['Children in HHS Care'].mean()

change_pct = (
    (avg_hhs_after - avg_hhs_before)
    / avg_hhs_before
) * 100

st.subheader("HHS Care Population Change")

st.metric(
    "Change after January 2025",
    f"{change_pct:.1f}%",
    delta=f"{change_pct:.1f}%"
)

st.write(
    f"Average HHS care population changed from "
    f"{avg_hhs_before:,.0f} before January 2025 "
    f"to {avg_hhs_after:,.0f} from January 2025 onwards."
)

st.info(
    "This is a descriptive historical comparison. "
    "It does not establish the cause of the population decline."
)
# EXPANDED FORECAST VALIDATION

st.divider()
st.header("Forecast Model Validation")

st.caption(
    "Post-break persistence-baseline performance across "
    "multiple forecast horizons."
)

# Load corrected post-break validation datasets
expanded_validation = pd.read_csv(
    "post_break_multi_horizon_validation.csv"
)

discharge_validation = pd.read_csv(
    "post_break_discharge_validation.csv"
)

# Overall HHS care-load validation
st.subheader("Post-Break HHS Care Forecast Validation")

st.caption(
    "Persistence-baseline performance evaluated using data from "
    "January 2025 onward. Horizons represent future recorded observations, "
    "not necessarily calendar days."
)

st.dataframe(
    expanded_validation,
    use_container_width=True,
    hide_index=True
)

fig_validation = px.line(
    expanded_validation,
    x="Horizon",
    y="MAE",
    markers=True,
    title="HHS Care Forecast Error by Horizon"
)

st.plotly_chart(fig_validation, use_container_width=True)


# Discharge-demand validation
st.subheader("Post-Break Discharge Forecast Validation")

st.caption(
    "Discharge forecasts were evaluated separately using the same "
    "post-break validation approach. MAPE excludes observations where "
    "actual discharge demand was zero."
)

st.dataframe(
    discharge_validation,
    use_container_width=True,
    hide_index=True
)

fig_discharge = px.line(
    discharge_validation,
    x="Horizon",
    y="MAE",
    markers=True,
    title="Discharge Forecast Error by Horizon"
)

st.plotly_chart(fig_discharge, use_container_width=True)

st.info(
    "Forecast horizons represent recorded observations rather than "
    "calendar days. Historical validation performance does not guarantee "
    "future accuracy."
)
# KEY FINDINGS AND LIMITATIONS

st.divider()
st.header("Key Findings & Operational Implications")

st.markdown("""
### Key Findings

- **Short-term persistence:** The latest observed HHS care population
  provides the strongest baseline for near-term forecasting among the
  tested models.

- **Model comparison:** Naive Persistence showed the strongest
  short-horizon performance, while Random Forest achieved lower error
  than Naive Persistence at the 14-observation horizon in the
  post-break validation.

- **Robustness of the baseline:** A March 2025 sensitivity check also
  showed Naive Persistence outperforming both Random Forest and
  Gradient Boosting.

- **Increasing forecast error with horizon:** Persistence forecast error
  increased from approximately 9.23 MAE at 1 observation ahead to
  79.56 MAE at 14 observations ahead in the post-break validation.

- **Historical regime difference:** Average HHS care was approximately
  7,810 before January 2025 and 2,543 from January 2025 onward.

- **Reporting irregularity:** Forecast horizons represent recorded
  observations rather than fixed calendar days because the dataset
  contains missing calendar dates.

### Limitations

- The dataset contains 355 missing calendar dates. Missing dates have
  not been treated as zero-demand days.

- Forecast horizons refer to recorded observations, not calendar days.

- The persistence model is a benchmark and does not prove that future
  demand will remain constant.

- The observed decline in HHS care population does not establish its
  underlying cause.

- Discharge forecasting was validated separately, but its relative
  error is high because discharge counts are small and include zero
  observations.

- The displayed forecast error ranges are historical MAE-based ranges,
  not calibrated statistical prediction intervals.

### Operational Implication

Use the forecasts as planning signals rather than fixed estimates.
Short-horizon forecasts can support near-term capacity awareness, while
longer-horizon decisions should account for increasing forecast error,
reporting gaps, and potential changes in the underlying data pattern.
""")
# SCENARIO COMPARISON

st.divider()
st.header("Scenario Comparison")

scenario_comparison = pd.read_csv("scenario_comparison.csv")

st.caption(
    "March 2025 onward net-pressure scenarios based on the observed "
    "distribution of the post-break operating period."
)

st.dataframe(
    scenario_comparison,
    use_container_width=True,
    hide_index=True
)

fig_scenario = px.bar(
    scenario_comparison,
    x="Scenario",
    y="Net_Pressure",
    title="March 2025 Onward Net Pressure Scenarios"
)

fig_scenario.add_hline(
    y=0,
    line_dash="dash"
)

st.plotly_chart(
    fig_scenario,
    use_container_width=True
)

st.caption(
    "Central scenario represents the March 2025 onward mean net pressure. "
    "Lower and higher scenarios represent approximately one standard deviation "
    "below and above the mean. These are scenario indicators, not probability forecasts."
)


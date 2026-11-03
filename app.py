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

# Load model comparison
results = pd.read_csv("model_comparison.csv")

st.subheader("Model Performance Comparison")
st.dataframe(results, use_container_width=True)

# Forecast horizon selector
horizon = st.select_slider(
    "Select Forecast Horizon (observations)",
    options=[1, 7, 14],
    value=7
)

# Latest observed values
latest_hhs = df['Children in HHS Care'].iloc[-1]
latest_discharge = df['Children discharged from HHS Care'].iloc[-1]

# Walk-forward validation errors
error_ranges = {1: 8.38, 7: 43.76, 14: 89.50}
error = error_ranges[horizon]

# Baseline forecast
forecast_data = pd.DataFrame({
    'Step': range(1, horizon + 1),
    'HHS Forecast': [latest_hhs] * horizon,
    'Lower Range': [max(0, latest_hhs - error)] * horizon,
    'Upper Range': [latest_hhs + error] * horizon,
    'Discharge Forecast': [latest_discharge] * horizon
})

# Display forecast
st.subheader("Future HHS Care Load Forecast")

st.metric(
    "Baseline HHS Care Forecast",
    f"{latest_hhs:,.0f}"
)

fig3 = px.line(
    forecast_data,
    x='Step',
    y='HHS Forecast',
    title='Naive Persistence Forecast'
)

fig3.add_scatter(
    x=forecast_data['Step'],
    y=forecast_data['Upper Range'],
    mode='lines',
    line=dict(width=0),
    showlegend=False
)

fig3.add_scatter(
    x=forecast_data['Step'],
    y=forecast_data['Lower Range'],
    mode='lines',
    fill='tonexty',
    fillcolor='rgba(100,149,237,0.2)',
    line=dict(width=0),
    name='Illustrative error range'
)

st.plotly_chart(fig3, use_container_width=True)

st.subheader("Discharge Demand Forecast")

st.dataframe(
    forecast_data[['Step', 'Discharge Forecast']],
    use_container_width=True
)

st.caption(
    "Forecasts are persistence baselines over future observation steps. "
    "Error ranges use historical walk-forward MAE and are not calibrated "
    "statistical confidence intervals. Discharge forecasts are not "
    "independently validated."
)

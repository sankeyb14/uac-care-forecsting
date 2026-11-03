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

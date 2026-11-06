# UAC Care Forecasting

## Project Overview

UAC Care Forecasting is a data analytics and forecasting project designed to analyze care demand, discharge patterns, operational pressure, and future workload within the UAC program.

The project combines historical analysis, data-quality assessment, time-series forecasting, model validation, and scenario analysis to provide an operational view of care-load trends.

## Business Objective

The primary objective is to answer:

- How has UAC care demand changed over time?
- What relationship exists between care load, transfers, and discharges?
- Is there evidence of a structural change in demand?
- What does near-term care demand look like?
- How reliable are the forecasting approaches?
- What operational pressure scenarios can be identified from the available data?

## Key Analysis

The project includes:

- Historical UAC care-load analysis
- Transfers vs. discharge analysis
- Data-quality and reporting-coverage assessment
- Structural-break analysis
- Forecasting using multiple approaches
- Post-break model comparison
- Multi-horizon validation
- Discharge-demand forecasting
- Early-warning indicators
- Net-pressure analysis
- Scenario comparison
- Operational insights and recommendations

## Forecasting Approach

The analysis evaluates multiple forecasting approaches, including:

- Naive Persistence
- Random Forest
- Gradient Boosting
- Exponential Smoothing
- ARIMA
- Moving Average

Model performance is evaluated using metrics including MAE, RMSE, and MAPE where appropriate.

The analysis places particular emphasis on the post-break operating period because historical demand showed a substantial change in January 2025.

## Key Findings

The analysis identified a major structural change in UAC care demand around January 2025, with average HHS care declining substantially compared with the earlier period.

Post-break validation showed that **Naive Persistence performed strongly at the 1- and 7-observation horizons in the available post-break validation results**. Model comparisons at longer horizons should be interpreted cautiously because the validation samples were not identical across models.

Discharge forecasting showed substantially lower absolute errors than the care-load forecasting task, although MAPE is less informative for low-volume discharge counts and should therefore be interpreted cautiously.

The analysis also identified that actual facility capacity thresholds were not available in the supplied data. Therefore, a true capacity-breach probability could not be calculated.

## Operational Interpretation

The dashboard is intended to support operational decision-making by providing:

- Current and historical workload visibility
- Near-term demand estimates
- Forecast-performance transparency
- Identification of changes in operating regime
- Net-pressure scenario indicators
- Evidence for monitoring future demand and discharge patterns

Scenario values should be interpreted as **operational indicators rather than probability forecasts**.

## Dashboard

The project includes an interactive Streamlit dashboard containing the major analytical outputs and operational insights.

## Repository Structure

Key project components include:

- `app.py` — Streamlit dashboard
- CSV files — cleaned data, forecasts, validation results, model comparisons, and scenario outputs
- Jupyter notebooks — data preparation, modelling, validation, and analysis

## Limitations

The analysis is subject to several limitations:

1. The dataset contains missing calendar dates and reporting gaps.
2. Facility capacity thresholds were not available.
3. The January 2025 structural change limits the usefulness of models trained primarily on the earlier operating regime.
4. Multi-step forecasts show increasing error with longer horizons.
5. MAPE can be misleading for low-volume discharge counts.
6. Scenario analysis represents statistical operating scenarios and should not be interpreted as probabilistic forecasts.

## Technology

- Python
- Pandas
- NumPy
- Scikit-learn
- Statsmodels
- Plotly
- Streamlit
- Jupyter Notebook

## Live Dashboard

The interactive dashboard is available through the project's Streamlit deployment.

## Repository

[GitHub Repository](https://github.com/sankeyb14/uac-care-forecsting)

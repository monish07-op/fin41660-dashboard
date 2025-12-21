# FIN41660 Time Series Forecasting Dashboard

## Project Overview

An interactive dashboard for financial time series forecasting using econometric models. Built for FIN41660 Financial Econometrics at University College Dublin.

Live Dashboard: https://fin41660-dashboard-tyicmctuws9toztote45qj.streamlit.app

---

## Team Members

- Manoj Kumar Ranganatha (25204984)
- Vishnu Pramarajan (25205593)
- Monish Shiva Kumar (25206754)
- Naveen Rajagopal (25230070)

---

## Models Implemented

1. OLS Regression (CAPM) - Capital Asset Pricing Model for systematic risk analysis
2. ARIMA(1,1,1) - Autoregressive Integrated Moving Average for return forecasting
3. GARCH(1,1) - Generalized Autoregressive Conditional Heteroskedasticity for volatility modeling

---

## File Structure

```
forecasting_dashboard.py    # Main Streamlit dashboard
standalone_analysis.py      # Standalone script (reproduces all results)
FIN41660_Report.pdf         # Written report
requirements.txt            # Python dependencies
README.md                   # This file
```

---

## How to Run

### Option 1: Use the Live Dashboard
Visit: https://fin41660-dashboard-tyicmctuws9toztote45qj.streamlit.app

### Option 2: Run Locally

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the dashboard:
```bash
streamlit run forecasting_dashboard.py
```

Or run the standalone script:
```bash
python standalone_analysis.py
```

---

## Features

Dashboard allows users to:
- Select custom date ranges
- Configure ARIMA parameters (p, d, q)
- Configure GARCH parameters (p, q)
- Set forecast horizon (5-30 days)
- View real-time model estimation

Statistical tests included:
- ADF Test and KPSS Test for stationarity
- Ljung-Box Test at lags 5, 10, 15, 20
- ARCH-LM Test at lags 5, 10
- Diebold-Mariano Test for model comparison

Accuracy metrics: MAE, RMSE, MAPE

---

## Key Results

| Metric | Value |
|--------|-------|
| OLS Beta | 0.6746 |
| OLS R-squared | 0.1661 |
| ARIMA AIC | 3943.49 |
| GARCH Persistence | 0.9967 |
| DM Test p-value | 0.0233 |

OLS significantly outperforms ARIMA for XLE return forecasting (p < 0.05).

---

## Requirements

- Python 3.8 or higher
- 4 GB RAM minimum
- Internet connection for data download

Dependencies: streamlit, pandas, numpy, statsmodels, arch, scipy, yfinance, matplotlib

---

## Data

- Asset: XLE (Energy Select Sector SPDR Fund)
- Benchmark: S&P 500 (^GSPC)
- Source: Yahoo Finance
- Period: 5 years daily data
- Train/Test Split: 80/20

---

## References

Bollerslev, T. (1986). Generalized autoregressive conditional heteroskedasticity. Journal of Econometrics.

Box, G. E., Jenkins, G. M. (2015). Time Series Analysis: Forecasting and Control.

Diebold, F. X., & Mariano, R. S. (1995). Comparing predictive accuracy. Journal of Business and Economic Statistics.

Engle, R. F. (1982). Autoregressive conditional heteroscedasticity. Econometrica.

---

University College Dublin | Michael Smurfit Graduate Business School | December 2025

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FIN41660 Financial Econometrics - Group Project
Time Series Forecasting Dashboard
Asset: XLE (Energy Select Sector SPDR Fund)

This dashboard implements:
1. OLS Regression
2. ARIMA Model
3. GARCH Model

University College Dublin
Academic Year 2025/2026
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Statistical models
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

# Set page configuration
st.set_page_config(
    page_title="FIN41660 - Time Series Forecasting",
    page_icon="📈",
    layout="wide"
)

# =============================================================================
# TITLE AND INTRODUCTION
# =============================================================================
st.title("📈 Time Series Forecasting Dashboard")
st.markdown("### FIN41660 Financial Econometrics - Group Project")
st.markdown("**Asset:** XLE (Energy Select Sector SPDR Fund)")

st.markdown("""
---
This dashboard analyzes the **XLE Energy ETF** using three econometric models:
- **OLS Regression**: Linear relationship between returns and market factors
- **ARIMA**: Autoregressive Integrated Moving Average for price/return forecasting  
- **GARCH**: Generalized Autoregressive Conditional Heteroskedasticity for volatility forecasting

Use the sidebar to configure parameters and explore different analyses.
---
""")

# =============================================================================
# SIDEBAR CONFIGURATION
# =============================================================================
st.sidebar.header("⚙️ Dashboard Settings")

# Date range selection
st.sidebar.subheader("📅 Date Range")
default_start = datetime.now() - timedelta(days=5*365)  # 5 years of data
default_end = datetime.now()

start_date = st.sidebar.date_input(
    "Start Date",
    value=default_start,
    max_value=datetime.now()
)

end_date = st.sidebar.date_input(
    "End Date", 
    value=default_end,
    max_value=datetime.now()
)

# Model parameters
st.sidebar.subheader("🔧 Model Parameters")

# ARIMA parameters
st.sidebar.markdown("**ARIMA Parameters**")
arima_p = st.sidebar.slider("AR order (p)", 0, 5, 1)
arima_d = st.sidebar.slider("Differencing (d)", 0, 2, 1)
arima_q = st.sidebar.slider("MA order (q)", 0, 5, 1)

# GARCH parameters  
st.sidebar.markdown("**GARCH Parameters**")
garch_p = st.sidebar.slider("GARCH p", 1, 3, 1)
garch_q = st.sidebar.slider("GARCH q", 1, 3, 1)

# Forecast horizon
forecast_horizon = st.sidebar.slider("Forecast Horizon (days)", 5, 30, 10)

# =============================================================================
# DATA LOADING - FIXED VERSION
# =============================================================================
@st.cache_data
def load_data(ticker, start, end):
    """Download data from Yahoo Finance"""
    data = yf.download(ticker, start=start, end=end, progress=False)
    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

@st.cache_data  
def load_market_data(start, end):
    """Download market benchmark (S&P 500) for OLS"""
    data = yf.download("^GSPC", start=start, end=end, progress=False)
    # Flatten multi-index columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = [col[0] for col in data.columns]
    return data

# Load the data
with st.spinner("📊 Downloading data from Yahoo Finance..."):
    xle_data = load_data("XLE", start_date, end_date)
    market_data = load_market_data(start_date, end_date)

if xle_data.empty:
    st.error("❌ Failed to download XLE data. Please check your internet connection.")
    st.stop()

# =============================================================================
# DATA PREPROCESSING - FIXED VERSION
# =============================================================================
# Get the price column (try different possible names)
price_col = None
for col_name in ['Adj Close', 'Close', 'adj close', 'close']:
    if col_name in xle_data.columns:
        price_col = col_name
        break

if price_col is None:
    st.error(f"❌ Could not find price column. Available columns: {list(xle_data.columns)}")
    st.stop()

# Create a clean dataframe
df = pd.DataFrame()
df['Price'] = xle_data[price_col]

# Get market price
market_price_col = None
for col_name in ['Adj Close', 'Close', 'adj close', 'close']:
    if col_name in market_data.columns:
        market_price_col = col_name
        break

df['Returns'] = df['Price'].pct_change() * 100  # Percentage returns
df['Log_Returns'] = np.log(df['Price'] / df['Price'].shift(1)) * 100

if market_price_col:
    df['Market_Returns'] = market_data[market_price_col].pct_change() * 100
else:
    df['Market_Returns'] = np.nan

df = df.dropna()

# =============================================================================
# SECTION 1: DATA OVERVIEW
# =============================================================================
st.header("1️⃣ Data Overview")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Observations", len(df))
with col2:
    st.metric("Date Range", f"{df.index[0].strftime('%Y-%m-%d')}")
with col3:
    st.metric("Latest Price", f"${df['Price'].iloc[-1]:.2f}")
with col4:
    total_return = ((df['Price'].iloc[-1] / df['Price'].iloc[0]) - 1) * 100
    st.metric("Total Return", f"{total_return:.2f}%")

# Price chart
st.subheader("📈 XLE Price History")
fig_price, ax_price = plt.subplots(figsize=(12, 5))
ax_price.plot(df.index, df['Price'], color='#1f77b4', linewidth=1.5)
ax_price.set_xlabel('Date')
ax_price.set_ylabel('Price (USD)')
ax_price.set_title('XLE Energy ETF - Adjusted Close Price')
ax_price.grid(True, alpha=0.3)
st.pyplot(fig_price)

# Returns chart
st.subheader("📊 Daily Returns")
fig_ret, ax_ret = plt.subplots(figsize=(12, 4))
ax_ret.plot(df.index, df['Returns'], color='#2ca02c', linewidth=0.8, alpha=0.8)
ax_ret.axhline(y=0, color='red', linestyle='--', linewidth=0.8)
ax_ret.set_xlabel('Date')
ax_ret.set_ylabel('Returns (%)')
ax_ret.set_title('XLE Daily Returns')
ax_ret.grid(True, alpha=0.3)
st.pyplot(fig_ret)

# Summary statistics
st.subheader("📋 Summary Statistics")
stats_df = pd.DataFrame({
    'Statistic': ['Mean', 'Std Dev', 'Min', 'Max', 'Skewness', 'Kurtosis'],
    'Price': [
        f"${df['Price'].mean():.2f}",
        f"${df['Price'].std():.2f}",
        f"${df['Price'].min():.2f}",
        f"${df['Price'].max():.2f}",
        f"{df['Price'].skew():.4f}",
        f"{df['Price'].kurtosis():.4f}"
    ],
    'Returns (%)': [
        f"{df['Returns'].mean():.4f}",
        f"{df['Returns'].std():.4f}",
        f"{df['Returns'].min():.4f}",
        f"{df['Returns'].max():.4f}",
        f"{df['Returns'].skew():.4f}",
        f"{df['Returns'].kurtosis():.4f}"
    ]
})
st.table(stats_df)

# =============================================================================
# SECTION 2: OLS REGRESSION
# =============================================================================
st.header("2️⃣ OLS Regression Analysis")

st.markdown("""
We estimate the **Capital Asset Pricing Model (CAPM)** using OLS:

$$R_{XLE} - R_f = \\alpha + \\beta (R_{market} - R_f) + \\varepsilon$$

Where we assume $R_f \\approx 0$ for simplicity (daily risk-free rate is negligible).
""")

# Prepare OLS data
ols_df = df[['Returns', 'Market_Returns']].dropna()
y = ols_df['Returns']
X = sm.add_constant(ols_df['Market_Returns'])

# Fit OLS model
ols_model = sm.OLS(y, X).fit()

# Display results
col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Regression Results")
    st.write(f"**Alpha (α):** {ols_model.params['const']:.6f}")
    st.write(f"**Beta (β):** {ols_model.params['Market_Returns']:.4f}")
    st.write(f"**R-squared:** {ols_model.rsquared:.4f}")
    st.write(f"**Adjusted R-squared:** {ols_model.rsquared_adj:.4f}")
    
    # Coefficient table
    coef_df = pd.DataFrame({
        'Coefficient': ols_model.params,
        'Std Error': ols_model.bse,
        't-Statistic': ols_model.tvalues,
        'p-Value': ols_model.pvalues
    })
    st.table(coef_df.round(4))

with col2:
    st.subheader("📈 Scatter Plot with Regression Line")
    fig_ols, ax_ols = plt.subplots(figsize=(8, 6))
    ax_ols.scatter(ols_df['Market_Returns'], ols_df['Returns'], alpha=0.5, s=10)
    
    # Regression line
    x_line = np.linspace(ols_df['Market_Returns'].min(), ols_df['Market_Returns'].max(), 100)
    y_line = ols_model.params['const'] + ols_model.params['Market_Returns'] * x_line
    ax_ols.plot(x_line, y_line, color='red', linewidth=2, label=f'β = {ols_model.params["Market_Returns"]:.2f}')
    
    ax_ols.set_xlabel('Market Returns (S&P 500) %')
    ax_ols.set_ylabel('XLE Returns %')
    ax_ols.set_title('XLE vs Market Returns')
    ax_ols.legend()
    ax_ols.grid(True, alpha=0.3)
    st.pyplot(fig_ols)

# OLS Interpretation
st.subheader("📝 Interpretation")
beta_val = ols_model.params['Market_Returns']
if beta_val > 1:
    beta_interpret = "XLE is **more volatile** than the market (aggressive stock)"
elif beta_val < 1:
    beta_interpret = "XLE is **less volatile** than the market (defensive stock)"
else:
    beta_interpret = "XLE moves **in line** with the market"

st.markdown(f"""
- **Beta = {beta_val:.4f}**: {beta_interpret}
- **R² = {ols_model.rsquared:.4f}**: {ols_model.rsquared*100:.1f}% of XLE's return variation is explained by market movements
- The beta is {"statistically significant" if ols_model.pvalues['Market_Returns'] < 0.05 else "not statistically significant"} at the 5% level (p-value: {ols_model.pvalues['Market_Returns']:.4f})
""")

# =============================================================================
# SECTION 3: ARIMA MODEL
# =============================================================================
st.header("3️⃣ ARIMA Model")

st.markdown(f"""
We fit an **ARIMA({arima_p},{arima_d},{arima_q})** model to forecast XLE returns.

The ARIMA model combines:
- **AR({arima_p})**: Autoregressive component (past values)
- **I({arima_d})**: Integration/Differencing (for stationarity)
- **MA({arima_q})**: Moving Average component (past errors)
""")

# Stationarity test
st.subheader("🔍 Stationarity Test (Augmented Dickey-Fuller)")
adf_result = adfuller(df['Returns'].dropna())
adf_df = pd.DataFrame({
    'Metric': ['ADF Statistic', 'p-value', 'Critical Value (1%)', 'Critical Value (5%)', 'Critical Value (10%)'],
    'Value': [
        f"{adf_result[0]:.4f}",
        f"{adf_result[1]:.4f}",
        f"{adf_result[4]['1%']:.4f}",
        f"{adf_result[4]['5%']:.4f}",
        f"{adf_result[4]['10%']:.4f}"
    ]
})
st.table(adf_df)

if adf_result[1] < 0.05:
    st.success("✅ The returns series is **stationary** (p-value < 0.05). We can proceed with ARIMA.")
else:
    st.warning("⚠️ The returns series may be **non-stationary**. Consider increasing differencing (d).")

# ACF and PACF plots
st.subheader("📊 ACF and PACF Plots")
col1, col2 = st.columns(2)

with col1:
    fig_acf, ax_acf = plt.subplots(figsize=(8, 4))
    acf_vals = acf(df['Returns'].dropna(), nlags=20)
    ax_acf.bar(range(len(acf_vals)), acf_vals, color='steelblue')
    ax_acf.axhline(y=1.96/np.sqrt(len(df)), color='red', linestyle='--')
    ax_acf.axhline(y=-1.96/np.sqrt(len(df)), color='red', linestyle='--')
    ax_acf.set_xlabel('Lag')
    ax_acf.set_ylabel('ACF')
    ax_acf.set_title('Autocorrelation Function (ACF)')
    st.pyplot(fig_acf)

with col2:
    fig_pacf, ax_pacf = plt.subplots(figsize=(8, 4))
    pacf_vals = pacf(df['Returns'].dropna(), nlags=20)
    ax_pacf.bar(range(len(pacf_vals)), pacf_vals, color='darkorange')
    ax_pacf.axhline(y=1.96/np.sqrt(len(df)), color='red', linestyle='--')
    ax_pacf.axhline(y=-1.96/np.sqrt(len(df)), color='red', linestyle='--')
    ax_pacf.set_xlabel('Lag')
    ax_pacf.set_ylabel('PACF')
    ax_pacf.set_title('Partial Autocorrelation Function (PACF)')
    st.pyplot(fig_pacf)

# Fit ARIMA model
st.subheader(f"📈 ARIMA({arima_p},{arima_d},{arima_q}) Model Results")

try:
    arima_model = ARIMA(df['Returns'].dropna(), order=(arima_p, arima_d, arima_q)).fit()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Model Summary:**")
        st.write(f"- AIC: {arima_model.aic:.4f}")
        st.write(f"- BIC: {arima_model.bic:.4f}")
        st.write(f"- Log Likelihood: {arima_model.llf:.4f}")
        
        # Coefficients
        st.write("**Coefficients:**")
        arima_coef_df = pd.DataFrame({
            'Coefficient': arima_model.params,
            'Std Error': arima_model.bse,
            'p-Value': arima_model.pvalues
        })
        st.table(arima_coef_df.round(4))
    
    with col2:
        # Forecast
        st.write(f"**{forecast_horizon}-Day Forecast:**")
        forecast = arima_model.forecast(steps=forecast_horizon)
        forecast_df = pd.DataFrame({
            'Day': range(1, forecast_horizon + 1),
            'Forecasted Return (%)': forecast.values
        })
        st.table(forecast_df.round(4))
    
    # Residual diagnostics
    st.subheader("🔬 ARIMA Residual Diagnostics")
    residuals = arima_model.resid
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_res, ax_res = plt.subplots(figsize=(8, 4))
        ax_res.plot(residuals, color='purple', linewidth=0.5)
        ax_res.axhline(y=0, color='red', linestyle='--')
        ax_res.set_xlabel('Observation')
        ax_res.set_ylabel('Residual')
        ax_res.set_title('ARIMA Residuals')
        st.pyplot(fig_res)
    
    with col2:
        fig_hist, ax_hist = plt.subplots(figsize=(8, 4))
        ax_hist.hist(residuals, bins=50, color='purple', alpha=0.7, edgecolor='black')
        ax_hist.set_xlabel('Residual')
        ax_hist.set_ylabel('Frequency')
        ax_hist.set_title('Distribution of Residuals')
        st.pyplot(fig_hist)
    
    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
    st.write(f"**Ljung-Box Test (lag 10):** Q-statistic = {lb_test['lb_stat'].values[0]:.4f}, p-value = {lb_test['lb_pvalue'].values[0]:.4f}")
    
    if lb_test['lb_pvalue'].values[0] > 0.05:
        st.success("✅ Residuals show no significant autocorrelation (good model fit)")
    else:
        st.warning("⚠️ Residuals show significant autocorrelation (consider adjusting parameters)")

except Exception as e:
    st.error(f"❌ Error fitting ARIMA model: {str(e)}")
    st.info("Try adjusting the ARIMA parameters in the sidebar.")

# =============================================================================
# SECTION 4: GARCH MODEL
# =============================================================================
st.header("4️⃣ GARCH Model")

st.markdown(f"""
We fit a **GARCH({garch_p},{garch_q})** model to forecast **volatility** of XLE returns.

The GARCH model captures **volatility clustering** - periods of high volatility tend to be followed by high volatility.

$$\\sigma_t^2 = \\omega + \\alpha_1 \\varepsilon_{{t-1}}^2 + \\beta_1 \\sigma_{{t-1}}^2$$
""")

# Fit GARCH model
try:
    # Scale returns for numerical stability
    returns_scaled = df['Returns'].dropna() 
    
    garch_spec = arch_model(returns_scaled, vol='Garch', p=garch_p, q=garch_q, mean='Constant')
    garch_model = garch_spec.fit(disp='off')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 GARCH Model Results")
        st.write(f"**Model:** GARCH({garch_p},{garch_q})")
        st.write(f"**AIC:** {garch_model.aic:.4f}")
        st.write(f"**BIC:** {garch_model.bic:.4f}")
        st.write(f"**Log Likelihood:** {garch_model.loglikelihood:.4f}")
        
        # Parameters
        st.write("**Parameters:**")
        garch_params = pd.DataFrame({
            'Coefficient': garch_model.params,
            'Std Error': garch_model.std_err,
            'p-Value': garch_model.pvalues
        })
        st.table(garch_params.round(4))
    
    with col2:
        st.subheader("📈 Volatility Forecast")
        # Forecast volatility
        garch_forecast = garch_model.forecast(horizon=forecast_horizon)
        vol_forecast = np.sqrt(garch_forecast.variance.iloc[-1].values)
        
        forecast_vol_df = pd.DataFrame({
            'Day': range(1, forecast_horizon + 1),
            'Forecasted Volatility (%)': vol_forecast
        })
        st.table(forecast_vol_df.round(4))
    
    # Conditional volatility plot
    st.subheader("📈 Conditional Volatility Over Time")
    cond_vol = garch_model.conditional_volatility
    
    fig_vol, ax_vol = plt.subplots(figsize=(12, 5))
    ax_vol.plot(df.index[-len(cond_vol):], cond_vol, color='red', linewidth=1)
    ax_vol.set_xlabel('Date')
    ax_vol.set_ylabel('Conditional Volatility (%)')
    ax_vol.set_title('GARCH Conditional Volatility (XLE)')
    ax_vol.grid(True, alpha=0.3)
    st.pyplot(fig_vol)
    
    # Returns with volatility bands
    st.subheader("📊 Returns with Volatility Bands")
    fig_bands, ax_bands = plt.subplots(figsize=(12, 5))
    
    returns_plot = df['Returns'].iloc[-len(cond_vol):]
    ax_bands.plot(df.index[-len(cond_vol):], returns_plot, color='blue', linewidth=0.5, alpha=0.7, label='Returns')
    ax_bands.plot(df.index[-len(cond_vol):], 2*cond_vol, color='red', linewidth=1, label='+2σ')
    ax_bands.plot(df.index[-len(cond_vol):], -2*cond_vol, color='red', linewidth=1, label='-2σ')
    ax_bands.fill_between(df.index[-len(cond_vol):], -2*cond_vol, 2*cond_vol, color='red', alpha=0.1)
    ax_bands.set_xlabel('Date')
    ax_bands.set_ylabel('Returns / Volatility (%)')
    ax_bands.set_title('XLE Returns with GARCH Volatility Bands (±2σ)')
    ax_bands.legend()
    ax_bands.grid(True, alpha=0.3)
    st.pyplot(fig_bands)
    
    # GARCH Interpretation
    st.subheader("📝 GARCH Interpretation")
    alpha = garch_model.params.get('alpha[1]', 0)
    beta = garch_model.params.get('beta[1]', 0)
    persistence = alpha + beta
    
    st.markdown(f"""
    - **α (alpha) = {alpha:.4f}**: Reaction to market shocks (ARCH effect)
    - **β (beta) = {beta:.4f}**: Persistence of volatility (GARCH effect)  
    - **Persistence (α + β) = {persistence:.4f}**: {"High persistence - volatility shocks last long" if persistence > 0.9 else "Moderate persistence - volatility mean-reverts"}
    - {"⚠️ Note: α + β close to 1 suggests volatility is highly persistent (integrated GARCH)" if persistence > 0.95 else ""}
    """)

except Exception as e:
    st.error(f"❌ Error fitting GARCH model: {str(e)}")
    st.info("Try adjusting the GARCH parameters in the sidebar.")

# =============================================================================
# SECTION 5: MODEL COMPARISON & FORECAST ACCURACY
# =============================================================================
st.header("5️⃣ Model Comparison & Forecast Accuracy")

st.markdown("""
We compare model performance using standard forecast accuracy metrics:
- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Square Error
- **MAPE**: Mean Absolute Percentage Error
""")

# In-sample comparison
st.subheader("📊 In-Sample Performance Comparison")

# Calculate metrics for ARIMA (in-sample)
try:
    arima_fitted = arima_model.fittedvalues
    arima_actual = df['Returns'].iloc[-len(arima_fitted):]
    
    arima_mae = np.mean(np.abs(arima_actual - arima_fitted))
    arima_rmse = np.sqrt(np.mean((arima_actual - arima_fitted)**2))
    arima_mape = np.mean(np.abs((arima_actual - arima_fitted) / arima_actual.replace(0, np.nan))) * 100
    
    # OLS fitted values
    ols_fitted = ols_model.fittedvalues
    ols_actual = ols_df['Returns']
    
    ols_mae = np.mean(np.abs(ols_actual - ols_fitted))
    ols_rmse = np.sqrt(np.mean((ols_actual - ols_fitted)**2))
    ols_mape = np.mean(np.abs((ols_actual - ols_fitted) / ols_actual.replace(0, np.nan))) * 100
    
    comparison_df = pd.DataFrame({
        'Metric': ['MAE', 'RMSE', 'MAPE (%)'],
        'OLS': [f"{ols_mae:.4f}", f"{ols_rmse:.4f}", f"{ols_mape:.2f}"],
        'ARIMA': [f"{arima_mae:.4f}", f"{arima_rmse:.4f}", f"{arima_mape:.2f}"]
    })
    
    st.table(comparison_df)
    
    st.markdown("""
    **Note:** 
    - Lower values indicate better fit
    - OLS and ARIMA serve different purposes: OLS explains returns via market factors, ARIMA captures time-series patterns
    - GARCH forecasts volatility, not returns, so it's not directly comparable
    """)

except Exception as e:
    st.warning(f"Could not compute comparison metrics: {str(e)}")

# =============================================================================
# SECTION 6: CONCLUSIONS
# =============================================================================
st.header("6️⃣ Summary & Conclusions")

st.markdown(f"""
### Key Findings for XLE Energy ETF:

**1. OLS/CAPM Analysis:**
- XLE has a beta of **{ols_model.params['Market_Returns']:.4f}**, indicating it is {"more" if ols_model.params['Market_Returns'] > 1 else "less"} volatile than the overall market
- **{ols_model.rsquared*100:.1f}%** of XLE's return variation can be explained by market movements

**2. ARIMA Analysis:**
- The ARIMA({arima_p},{arima_d},{arima_q}) model captures the time-series dynamics of returns
- Returns series is {"stationary" if adf_result[1] < 0.05 else "non-stationary"} based on ADF test

**3. GARCH Analysis:**
- The GARCH model successfully captures volatility clustering in XLE returns
- Energy sector shows {"high" if persistence > 0.9 else "moderate"} volatility persistence, typical of commodity-linked assets

**4. Practical Implications:**
- Energy ETFs are sensitive to oil price movements, geopolitical events, and macroeconomic factors
- High volatility persistence suggests risk management is crucial for energy sector investments
- The models provide useful tools for both return and volatility forecasting
""")

# =============================================================================
# FOOTER
# =============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p><strong>FIN41660 Financial Econometrics - Group Project</strong></p>
    <p>University College Dublin | Academic Year 2025/2026</p>
</div>
""", unsafe_allow_html=True)

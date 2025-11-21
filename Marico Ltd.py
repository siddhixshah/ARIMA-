import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Marico ARIMA Forecasting", layout="wide")

# ------------------------------
# Header
# ------------------------------
st.title("📈 Marico Ltd — ARIMA Forecasting App")
st.markdown("Project 2: Monthly data from 2021 to 2025 + Forecast 2025–2026")

# ------------------------------
# Fetch Data
# ------------------------------
st.subheader("📥 Fetching Data")

ticker = "MARICO.NS"

with st.spinner("Downloading Marico monthly price data..."):
    data = yf.download(ticker, start="2021-01-01", end="2025-01-01")

if data.empty:
    st.error("Failed to download data. Try again later.")
    st.stop()

# Monthly resampling
monthly = data["Close"].resample("M").last()

st.write("### Preview of Monthly Data")
st.dataframe(monthly.to_frame())

# ------------------------------
# Graph 1 — Price Change
# ------------------------------
st.subheader("📊 Graph 1 — Monthly Price Change (2021–2025)")
fig1, ax1 = plt.subplots(figsize=(10,5))
ax1.plot(monthly, label="Monthly Close Price")
ax1.set_title("Monthly Price Change (2021–2025)")
ax1.set_xlabel("Year")
ax1.set_ylabel("Price (INR)")
ax1.grid(True)
ax1.legend()
st.pyplot(fig1)

# ------------------------------
# ARIMA Model
# ------------------------------
st.subheader("🤖 ARIMA Model Training (1,1,1)")
with st.spinner("Training ARIMA model..."):
    model = ARIMA(monthly, order=(1,1,1))
    model_fit = model.fit()

# In-sample forecast
fitted_vals = model_fit.predict(start=monthly.index[1], end=monthly.index[-1])

# ------------------------------
# Graph 2 — ARIMA Overlap
# ------------------------------
st.subheader("📈 Graph 2 — ARIMA Overlap on Original Prices")
fig2, ax2 = plt.subplots(figsize=(10,5))
ax2.plot(monthly, label="Actual")
ax2.plot(fitted_vals, label="ARIMA Fitted", linestyle="--")
ax2.set_title("ARIMA Forecast Overlap (2021–2025)")
ax2.set_xlabel("Year")
ax2.set_ylabel("Price (INR)")
ax2.grid(True)
ax2.legend()
st.pyplot(fig2)

# ------------------------------
# Forecast 2025–2026
# ------------------------------
st.subheader("🔮 Forecasting 2025–2026")

forecast_steps = 12  # 12 months
future_forecast = model_fit.forecast(steps=forecast_steps)

future_index = pd.date_range(start="2025-01-31", periods=forecast_steps, freq="M")
future_forecast.index = future_index

st.write("### 📅 Forecasted Prices (2025–2026)")
st.dataframe(future_forecast.to_frame("Forecasted Price"))

# ------------------------------
# Graph 3 — Future Forecast
# ------------------------------
st.subheader("📉 Graph 3 — Forecast (2025–2026)")
fig3, ax3 = plt.subplots(figsize=(10,5))
ax3.plot(monthly, label="Actual 2021–2025")
ax3.plot(future_forecast, label="Forecast 2025–2026", linestyle="--")
ax3.set_title("2025–2026 ARIMA Forecast")
ax3.set_xlabel("Year")
ax3.set_ylabel("Price (INR)")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

st.success("✨ Analysis Complete!")

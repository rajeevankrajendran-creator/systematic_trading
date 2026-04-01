# Systematic Trading
## Description
A regime-aware, ML-powered algorithmic trading system for BTC/USD. The system uses a LightGBM three-class classifier (BUY / SELL / FLAT) trained on OHLCV-derived technical features across three hypothesis groups: Trend Momentum, RSI Divergence + Volume, and Volatility Breakout. Signals are validated via walk-forward cross-validation and executed through a bar-by-bar engine with dual stop-loss logic.

Data used
Minute-level BTC/USDT OHLCV data sourced from Kaggle, resampled to 1-hour bars and filtered from 2017 onwards.

# API
Lime Demo accessible at:
https://systematictrading.streamlit.app/
Main endpoints:

GET /predict — returns trading signals and backtest performance metrics for a given date range
GET /health — returns API status

# Setup instructions
Clone the repository, create a virtual environment using pyenv, and install dependencies via pip install -r requirements.txt. Configure environment variables using direnv and a .envrc file. The trained model pickle is generated from the training notebooks and placed manually into raw_data/.

# Usage
Run the FastAPI backend locally with uvicorn and launch the Streamlit frontend to interact with the equity curve, candlestick chart, and action breakdown visualisations. Alternatively, query the deployed Cloud Run API directly with a cutoff_date, initial_capital, and position_size as parameters.

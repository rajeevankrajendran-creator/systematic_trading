# Systematic Trading Engine

A regime-aware, ML-powered algorithmic trading system for BTC/USD, built as the capstone project for Le Wagon Data Science & AI Bootcamp (Batch #2211).

The system generates long / flat / short signals using a LightGBM multiclass classifier trained on OHLCV-derived technical features. Signals are validated through a rigorous three-method feature validation pipeline and executed via a bar-by-bar engine with dual exit logic.

🔗 **Live demo:** [systematictrading.streamlit.app](https://systematictrading.streamlit.app/)

---

## Backtested Performance (2024–2026)

| Metric | Value |
|---|---|
| Total Return | +67.5% |
| Annualised Return | 27.0% |
| Sharpe Ratio | 2.94 |
| Max Drawdown | -0.6% |
| Win Rate | 84.6% |
| Profit Factor | 12.19 |
| Total Trades | 896 |
| BTC Buy-and-Hold Return | 58.8% |
| **Outperformance vs Benchmark** | **+8.7pp** |

> Backtested prototype only — not financial advice. Performance based on 20% risk per trade. Results are subject to the inherent limitations of backtesting including slippage, look-ahead bias, and overfitting.

---

## System Architecture

### Signal Generation
The core model is a LightGBM three-class classifier predicting BUY / FLAT / SELL at each 1-hour bar. Three hypothesis groups drive feature construction:

- **Trend Momentum** — EMA crossovers, rate of change, ADX
- **RSI Divergence + Volume** — RSI with volume confirmation signals
- **Volatility Breakout** — ATR-normalised price range, Bollinger Band width

### Feature Validation Pipeline
No single validation method is sufficient. Features are triangulated across three independent methods before inclusion:

- **Spearman Rank IC** — evaluated across 11 forward-return horizons to test linear predictive power
- **Mutual Information** — captures non-linear relationships that IC misses (e.g. natr shows high MI but weak IC)
- **SHAP Values** — identifies features the model actually uses at inference time

This triangulation exposed several artefacts: rsi_divergence showed high IC but near-zero MI and absent SHAP (linear artefact); time cyclical features appeared in SHAP despite weak IC/MI. Neither would have been caught by a single-method screen.

### Label Construction
Binary labels are derived from forward returns normalised against ATR, making the signal regime-aware rather than fixed-threshold. The label and execution horizon were both set to **N=12 bars (1 hour)** — confirmed as the optimal horizon via IC analysis across 11 candidates. Aligning these two was the single highest-leverage modelling decision in the project.

### Execution Engine
The bar-by-bar execution engine applies:
- **EMA-12 stop loss** — dynamic, volatility-sensitive exit
- **Time-based exit** — forced close at N_HOLD = 12 bars
- **Confidence threshold filtering** — signals below threshold are suppressed to control transaction cost drag; overtrading at low confidence thresholds wiped gross edge with transaction costs in earlier versions

---

## Data

- Source: Minute-level BTC/USDT OHLCV data from Kaggle
- Resampled to 1-hour bars
- Filtered from 2017 onwards
- Features computed using pandas-ta (v0.4.71b0)

---

## Stack

| Layer | Technology |
|---|---|
| Model | LightGBM, scikit-learn |
| Feature Engineering | pandas, pandas-ta |
| Backend API | FastAPI, deployed to Google Cloud Run |
| Frontend | Streamlit |
| Containerisation | Docker |
| Data | Binance historical OHLCV (pickle) |

---

## API

**Base URL:** `https://systematic-trading-api-469354767887.europe-west1.run.app`

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check — confirms API is running |
| `/backtest` | GET | Returns trading signals and backtest performance metrics for a given date range |

**Parameters for `/backtest`:**

| Parameter | Type | Description |
|---|---|---|
| `cutoff_date` | string | Start date for backtest window (YYYY-MM-DD) |
| `initial_capital` | float | Starting capital in USD |
| `position_size` | float | Risk per trade as a decimal (e.g. 0.2 for 20%) |

---

## Setup

### Prerequisites
- Python 3.10+
- pyenv
- direnv

### Installation

```bash
git clone https://github.com/rajeevankrajendran-creator/systematic_trading.git
cd systematic_trading
pyenv virtualenv 3.10.x systematic_trading
pyenv local systematic_trading
pip install -r requirements.txt
```

### Environment Variables

Configure a `.envrc` file in the project root using direnv:

```bash
export MODEL_PATH=raw_data/model.pkl
export DATA_PATH=raw_data/btc_data.pkl
```

The trained model pickle is generated from the training notebooks and placed manually into `raw_data/`.

---

## Usage

### Run locally

```bash
# Start FastAPI backend
uvicorn api.fast:app --reload

# Launch Streamlit frontend (separate terminal)
streamlit run app/app.py
```

The Streamlit frontend provides:
- Equity curve vs BTC benchmark
- Candlestick price chart
- Action breakdown (BUY / FLAT / SELL distribution)
- Full trade statistics and risk metrics

### Query the API directly

```bash
curl "https://systematic-trading-api-469354767887.europe-west1.run.app/backtest?cutoff_date=2024-01-01&initial_capital=10000&position_size=0.2"
```

---

## Project Structure

```
systematic_trading/
├── api/
│   └── fast.py               # FastAPI backend
├── app/
│   └── app.py                # Streamlit frontend
├── notebooks/
│   ├── feature_engineering/  # Feature construction and IC validation
│   ├── modelling/            # LightGBM training and evaluation
│   └── backtesting/          # Strategy simulation
├── raw_data/                 # Model pickle and OHLCV data (not committed)
├── requirements.txt
└── README.md
```

---

## Disclaimer

This project was created by students of Le Wagon Batch 2211 for educational and demonstration purposes only. Nothing herein constitutes financial, investment, or trading advice. All performance data is simulated via backtesting on historical data and is not indicative of future results. Any action taken based on this project is done strictly at the user's own risk.

---

*Collaborators: Isabella [Last Name] · Taylan [Last Name]*

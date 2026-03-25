import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.main import predict, load_data

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

app.state.df = load_data()

# Endpoint for https://systematic-trading-api-469354767887.europe-west1.run.app
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://systematic-trading-api-469354767887.europe-west1.run.app/backtest?cutoff_date=2024-01-01&initial_capital=10000&position_size=1.0
@app.get("/backtest")
def run_backtest(
    cutoff_date:     str   = '2025-01-01',
    initial_capital: float = 1000.0,
    position_size:   float = 1.0           # slider: 0.10 to 1.0
):
    summary = predict(cutoff_date, initial_capital, position_size, app.state.df)
    return summary

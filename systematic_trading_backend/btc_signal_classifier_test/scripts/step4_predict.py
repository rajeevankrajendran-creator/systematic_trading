"""
Step 4 — Predict
Loads saved model + features, generates predictions.
Output: outputs/predictions.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

from step3_train import MODEL_FEATURES, N_LAGS, create_lags, get_X_y


def load_model(path='models/model.txt'):
    return lgb.Booster(model_file=path)


def predict(model, X):
    proba = model.predict(X)          # shape: (n, 3)
    preds = proba.argmax(axis=1)      # 0=SELL, 1=FLAT, 2=BUY
    return preds, proba


if __name__ == '__main__':
    # Load data with features + labels
    df = pd.read_csv('data/labelled.csv', parse_dates=['open_time'], index_col='open_time')
    df = create_lags(df, MODEL_FEATURES, N_LAGS)
    X, y = get_X_y(df)

    model = load_model()
    preds, proba = predict(model, X)

    # Build output
    out = df[['close', 'sma_50', 'sma_200', 'adx', 'atr_14', 'regime', 'label']].copy()
    out['prediction'] = preds
    out['prob_sell'] = proba[:, 0]
    out['prob_flat'] = proba[:, 1]
    out['prob_buy']  = proba[:, 2]

    out.to_csv('outputs/predictions.csv')
    print(f"Saved → outputs/predictions.csv")
    print(f"Shape: {out.shape}")
    print(f"\nPrediction distribution:\n{pd.Series(preds).value_counts().sort_index()}")

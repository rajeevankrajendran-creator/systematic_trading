"""
Step 2 — Label Construction
Reads features.csv, computes forward return and 3-class labels (0=SELL, 1=FLAT, 2=BUY).
Output: data/labelled.csv
"""

import pandas as pd
import numpy as np

# ── Config ───────────────────────────────────────────────────────
N = 12      # lookahead horizon in bars (from IC validation)
K = 1.0     # ATR multiplier for threshold


def load_features(path='data/features.csv'):
    df = pd.read_csv(path, parse_dates=['open_time'], index_col='open_time')
    return df


def construct_labels(df, n=N, k=K):
    # Forward return
    df['forward_return'] = df['close'].pct_change(n).shift(-n)

    # ATR-based threshold
    threshold = k * df['atr_14'] / df['close']

    # 3-class labels: 0=SELL, 1=FLAT, 2=BUY
    conditions = [
        df['forward_return'] > threshold,     # BUY
        df['forward_return'] < -threshold,    # SELL
    ]
    df['label'] = np.select(conditions, [2, 0], default=1)

    # Drop rows with no forward return (last N bars)
    df = df.dropna(subset=['forward_return'])

    return df


if __name__ == '__main__':
    df = load_features()
    df = construct_labels(df)

    print(f"Shape: {df.shape}")
    print(f"\nLabel distribution:")
    print(f"  SELL (0): {(df['label']==0).sum():,}")
    print(f"  FLAT (1): {(df['label']==1).sum():,}")
    print(f"  BUY  (2): {(df['label']==2).sum():,}")

    df.to_csv('data/labelled.csv')
    print(f"\nSaved → data/labelled.csv")

"""
Step 1 — Feature Engineering
Reads raw OHLCV data, computes all 11 model features + regime gate columns.
Output: data/features.csv
"""

import pandas as pd
import pandas_ta as ta

def load_raw_data(path='data/btc_data_clean.csv'):
    df = pd.read_csv(path, parse_dates=['Open time'])
    df.columns = [c.strip().lower().replace(' ', '_') for c in df.columns]
    df = df.set_index('open_time').sort_index()
    return df

#  H1: Trend Momentu

def compute_roc(df):
    df['roc_10'] = df['close'].pct_change(periods=10)
    df['roc_21'] = df['close'].pct_change(periods=21)
    return df

def compute_macd(df):
    macd = ta.macd(df['close'], fast=12, slow=26, signal=9)
    df['macd_histogram'] = macd['MACDh_12_26_9']
    return df

def compute_adx(df):
    df['adx'] = ta.adx(df['high'], df['low'], df['close'], length=14)['ADX_14']
    return df

#  H2: RSI Divergence + Volume

def compute_rsi(df):
    df['rsi_14'] = ta.rsi(df['close'], length=14)
    return df

def compute_rsi_divergence(df):
    price_high = df['close'].rolling(14).max()
    price_low  = df['close'].rolling(14).min()
    rsi_high   = df['rsi_14'].rolling(14).max()
    rsi_low    = df['rsi_14'].rolling(14).min()

    bearish = (df['close'] == price_high) & (df['rsi_14'] < rsi_high)
    bullish = (df['close'] == price_low)  & (df['rsi_14'] > rsi_low)

    df['rsi_divergence'] = 0
    df.loc[bullish, 'rsi_divergence'] = 1
    df.loc[bearish, 'rsi_divergence'] = -1
    return df

def compute_obv_change(df):
    direction = df['close'].diff(1).apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    obv = (df['volume'] * direction).cumsum()
    df['obv_change'] = obv.diff(10)
    return df

def compute_volume_roc(df):
    df['volume_roc'] = df['volume'].pct_change(periods=10)
    return df

#  H3: Volatility Breakout

def compute_bb_width(df):
    bb = ta.bbands(df['close'], length=20, std=2)
    df['bb_width'] = bb[[c for c in bb.columns if c.startswith('BBB_')][0]]
    return df

def compute_atr(df):
    df['atr_14'] = ta.atr(df['high'], df['low'], df['close'], length=14)
    return df

def compute_natr(df):
    df['natr'] = ta.natr(df['high'], df['low'], df['close'], length=14)
    return df

#  Regime Gate (Step 6 only — not model features

def compute_regime(df):
    df['sma_50']  = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['regime']  = (df['sma_50'] > df['sma_200']).astype(int).replace(0, -1)
    return df

#  Pipeline

def feature_pipeline(df):
    steps = [
        compute_roc, compute_macd, compute_adx,          # H1
        compute_rsi, compute_rsi_divergence,              # H2
        compute_obv_change, compute_volume_roc,           # H2
        compute_bb_width, compute_atr, compute_natr,      # H3
        compute_regime,                                    # Regime gate
    ]
    for step in steps:
        df = step(df)
    return df


if __name__ == '__main__':
    df = load_raw_data()
    df = feature_pipeline(df)

    print(f"Shape: {df.shape}")
    print(f"\nNull counts:")
    feature_cols = [
        'roc_10', 'roc_21', 'macd_histogram', 'adx',
        'rsi_14', 'rsi_divergence', 'obv_change', 'volume_roc',
        'bb_width', 'atr_14', 'natr',
        'sma_50', 'sma_200', 'regime'
    ]
    print(df[feature_cols].isnull().sum())

    df.to_csv('data/features.csv')
    print(f"\nSaved → data/features.csv")

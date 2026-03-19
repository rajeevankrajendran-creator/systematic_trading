"""
Step 3 — Train LightGBM with Walk-Forward CV
Reads labelled.csv, trains model, saves to models/model.txt
Output: models/model.txt, outputs/cv_results.csv
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, f1_score
import time
import os

#  Config
N_LAGS   = 5       # lag features per base feature
N_SPLITS = 5       # walk-forward folds
GAP      = 24      # gap between train/val to prevent leakage

# The 11 model features (regime gate columns are excluded)
MODEL_FEATURES = [
    'roc_10', 'roc_21', 'macd_histogram', 'adx',
    'rsi_14', 'rsi_divergence', 'obv_change', 'volume_roc',
    'bb_width', 'atr_14', 'natr'
]


def load_labelled(path='data/labelled.csv'):
    df = pd.read_csv(path, parse_dates=['open_time'], index_col='open_time')
    return df


def create_lags(df, features, n_lags):
    for col in features:
        for lag in range(1, n_lags + 1):
            df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df = df.dropna()
    return df


def get_X_y(df):
    # base features + their lags
    feature_cols = MODEL_FEATURES.copy()
    for col in MODEL_FEATURES:
        for lag in range(1, N_LAGS + 1):
            feature_cols.append(f'{col}_lag{lag}')

    X = df[feature_cols]
    y = df['label']
    return X, y


def train_model(X, y):
    tscv = TimeSeriesSplit(n_splits=N_SPLITS, gap=GAP)
    fold_results = []

    print("Walk-Forward Cross-Validation")


    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        train_set = lgb.Dataset(X_train, label=y_train)
        val_set   = lgb.Dataset(X_val, label=y_val, reference=train_set)

        params = {
            'objective':        'multiclass',
            'num_class':        3,
            'metric':           'multi_logloss',
            'boosting_type':    'gbdt',
            'learning_rate':    0.05,
            'num_leaves':       31,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq':     5,
            'class_weight':     'balanced',
            'verbose':          -1,
            'seed':             42,
        }

        start = time.time()
        model = lgb.train(
            params,
            train_set,
            num_boost_round=300,
            valid_sets=[val_set],
            callbacks=[lgb.log_evaluation(period=-1)]
        )
        elapsed = time.time() - start

        # Predictions
        y_pred = model.predict(X_val).argmax(axis=1)
        f1 = f1_score(y_val, y_pred, average='weighted')

        fold_results.append({
            'fold': fold + 1,
            'train_size': len(X_train),
            'val_size': len(X_val),
            'f1_weighted': f1,
            'time_s': round(elapsed, 1)
        })

        print(f"  Fold {fold+1}: F1={f1:.3f}  "
              f"train={len(X_train):,} val={len(X_val):,}  "
              f"({elapsed:.1f}s)")



    # Final model on all data
    print("\nTraining final model on full dataset...")
    train_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, train_set, num_boost_round=300,
                            callbacks=[lgb.log_evaluation(period=-1)])

    return final_model, pd.DataFrame(fold_results)


if __name__ == '__main__':
    df = load_labelled()
    df = create_lags(df, MODEL_FEATURES, N_LAGS)
    X, y = get_X_y(df)

    print(f"X shape: {X.shape}")
    print(f"y distribution:\n{y.value_counts().sort_index()}\n")

    model, results = train_model(X, y)

    # Save
    os.makedirs('models', exist_ok=True)
    os.makedirs('outputs', exist_ok=True)

    model.save_model('models/model.txt')
    results.to_csv('outputs/cv_results.csv', index=False)

    print(f"\nSaved → models/model.txt")
    print(f"Saved → outputs/cv_results.csv")
    print(f"\nCV Summary:\n{results}")

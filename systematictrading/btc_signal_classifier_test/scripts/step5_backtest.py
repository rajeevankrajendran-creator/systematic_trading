"""
Step 5 — Execution Filter & Backtest
Applies regime gate to predictions, runs backtest.
Output: outputs/backtest_results.csv, outputs/trade_log.csv
"""

import pandas as pd
import numpy as np


#  Config
INITIAL_CAPITAL = 1000.0
ATR_MULTIPLIER  = 1.5
RISK_PCT        = 0.01
COST_PCT        = 0.001


def apply_regime_gate(df):
    """
    Regime gate from guide Step 6:
    - Bullish (sma_50 > sma_200): only BUY signals pass
    - Bearish (sma_50 < sma_200): only SELL signals pass
    - Otherwise: HOLD (prediction overridden to FLAT=1)
    """
    df = df.copy()
    bull = df['regime'] == 1
    bear = df['regime'] == -1

    # Override predictions that conflict with regime
    # BUY (2) only allowed in bull regime
    df.loc[~bull & (df['prediction'] == 2), 'prediction'] = 1
    # SELL (0) only allowed in bear regime
    df.loc[~bear & (df['prediction'] == 0), 'prediction'] = 1

    return df


def backtest(df, initial_capital=INITIAL_CAPITAL):
    capital = initial_capital
    position = 'flat'
    entry_price = 0.0
    pos_size = 0.0
    trade_log = []

    for timestamp, row in df.iterrows():
        close = row['close']
        atr   = row['atr_14']
        pred  = row['prediction']

        stop_distance = ATR_MULTIPLIER * atr
        new_pos_size  = (capital * RISK_PCT) / stop_distance if stop_distance > 0 else 0

        # Check stop loss if in position
        if position == 'long' and entry_price > 0:
            stop_price = entry_price - (ATR_MULTIPLIER * atr)
            if close <= stop_price:
                pnl = (close - entry_price) * pos_size
                cost = close * pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp, 'action': 'stop_loss',
                    'price': close, 'pnl': pnl, 'cost': cost, 'capital': capital
                })
                position = 'flat'
                entry_price = 0.0
                continue

        # BUY signal + not in position
        if pred == 2 and position == 'flat':
            pos_size = new_pos_size
            entry_price = close
            cost = close * pos_size * COST_PCT
            capital -= cost
            position = 'long'
            trade_log.append({
                'timestamp': timestamp, 'action': 'enter_long',
                'price': close, 'pnl': 0, 'cost': cost, 'capital': capital
            })

        # SELL/FLAT signal + in position → close
        elif pred != 2 and position == 'long':
            pnl = (close - entry_price) * pos_size
            cost = close * pos_size * COST_PCT
            capital += pnl - cost
            trade_log.append({
                'timestamp': timestamp, 'action': 'close_long',
                'price': close, 'pnl': pnl, 'cost': cost, 'capital': capital
            })
            position = 'flat'
            entry_price = 0.0

    return capital, pd.DataFrame(trade_log)


if __name__ == '__main__':
    df = pd.read_csv('outputs/predictions.csv', parse_dates=['open_time'], index_col='open_time')

    # Apply regime gate
    df = apply_regime_gate(df)

    print(f"Predictions after regime gate:\n{df['prediction'].value_counts().sort_index()}\n")

    # Run backtest
    final_capital, trade_log = backtest(df)

    # Results
    total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    n_trades = len(trade_log[trade_log['action'] == 'enter_long']) if len(trade_log) > 0 else 0

    print(f"Initial capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final capital:   ${final_capital:,.2f}")
    print(f"Total return:    {total_return:.2f}%")
    print(f"Total trades:    {n_trades}")

    if len(trade_log) > 0:
        exits = trade_log[trade_log['action'] != 'enter_long']
        if len(exits) > 0:
            wins = (exits['pnl'] > 0).sum()
            print(f"Win rate:        {wins/len(exits)*100:.1f}%")

    trade_log.to_csv('outputs/trade_log.csv', index=False)
    print(f"\nSaved → outputs/trade_log.csv")

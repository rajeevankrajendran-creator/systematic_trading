import pandas as pd
import numpy as np
import lightgbm as lgb

from backend.config import (
    PICKLE_PATH,
    FEATURE_COLS,
    LGBM_PARAMS,
    NUM_BOOST_ROUND,
    CONFIDENCE_THRESHOLD,
    COST_PCT,
    N_HOLD,
    RISK_FREE_RATE,
)

# ==============================================================================
# FUNCTION 1 — Load pickle
# ==============================================================================

def load_data():
    df = pd.read_pickle(PICKLE_PATH)
    return df


# ==============================================================================
# FUNCTION 2 — Split dataset according to user input
# ==============================================================================

def split_data(df, cutoff_date):
    X = df[FEATURE_COLS]
    y = df[['label']]

    # Creating the training set for X and y
    X_train   = X[X.index <  cutoff_date]
    y_train   = y[y.index <  cutoff_date]

    # Creating the testing set for X and y
    X_predict = X[X.index >= cutoff_date]
    y_predict = y[y.index >= cutoff_date]  # kept for evaluation only

    # Split diagnostics
    total_bars = len(X)
    train_bars = len(X_train)
    test_bars  = len(X_predict)
    train_pct  = train_bars / total_bars * 100
    test_pct   = test_bars  / total_bars * 100

    print(f"{'='*45}")
    print(f"  TRAIN / TEST SPLIT SUMMARY")
    print(f"{'='*45}")
    print(f"  Train:  {train_bars:>6,} bars  ({train_pct:>5.1f}%)  up to {cutoff_date}")
    print(f"  Test:   {test_bars:>6,} bars  ({test_pct:>5.1f}%)  from {cutoff_date} onwards")
    print(f"  Total:  {total_bars:>6,} bars")
    print(f"{'='*45}")
    split_ok = train_pct >= 70.0
    print(f"  Status: {'OK' if split_ok else 'WARNING — train below 70%'}")
    print(f"{'='*45}")

    return X_train, X_predict, y_train, y_predict


# ==============================================================================
# FUNCTION 3 — Train the model
# ==============================================================================

def train_model(X_train, y_train):
    # Multiclass — no scale_pos_weight for multiclass
    params = LGBM_PARAMS.copy()

    # Packaging the training data into LightGBM's required format
    train_set_lgb = lgb.Dataset(X_train, label=y_train)

    # Training the model
    final_model = lgb.train(
        params=params,
        train_set=train_set_lgb,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(period=-1)]
    )

    return final_model


# ==============================================================================
# FUNCTION 4 — Execution engine
# ==============================================================================

def run_execution_engine(df, final_model, X_predict, y_predict, initial_capital,
                         position_size):

    # Generate predictions — multiclass returns shape (n_bars, 3)
    # Column 0 = proba_sell, Column 1 = proba_flat, Column 2 = proba_buy
    pred_proba_final = final_model.predict(X_predict)

    # Building the signals DataFrame — input to the execution filter
    signals_df = df.loc[X_predict.index, ['Close', 'Low', 'sma_50', 'sma_200',
                                          'adx', 'atr_14', 'ema_12']].copy()
    signals_df['proba_sell'] = pred_proba_final[:, 0]
    signals_df['proba_flat'] = pred_proba_final[:, 1]
    signals_df['proba_buy']  = pred_proba_final[:, 2]
    signals_df['true_label'] = y_predict.values

    # State variables — maintained by the loop, not stored in DataFrame
    position       = 'flat'
    entry_price    = 0.0
    trade_pos_size = 0.0
    capital        = initial_capital
    N_HOLD_        = N_HOLD
    bars_in_trade  = 0

    # Storing the trade logs
    trade_log = []

    for timestamp, row in signals_df.iterrows():
        close      = row['Close']
        sma_50     = row['sma_50']
        sma_200    = row['sma_200']
        adx        = row['adx']
        atr        = row['atr_14']
        proba_buy  = row['proba_buy']
        proba_sell = row['proba_sell']

        # Skip bars where indicators are not yet available (NaN warmup period)
        if pd.isna(sma_50) or pd.isna(sma_200) or pd.isna(row['ema_12']):
            continue

        # Regime — SMA Cross only
        bull = (sma_50 > sma_200)
        bear = (sma_50 < sma_200)

        # ── LONG MANAGEMENT ───────────────────────────────────────────────────
        if position == 'long':

            # EXIT 1: Stop Loss — close below EMA 12
            if close < row['ema_12']:
                pnl  = (row['ema_12'] - entry_price) * trade_pos_size
                cost = row['ema_12'] * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp, 'action': 'stop_loss_long',
                    'close': row['ema_12'], 'pnl': pnl, 'cost': cost, 'capital': capital
                })
                position, entry_price, trade_pos_size, bars_in_trade = 'flat', 0.0, 0.0, 0
                continue

            # EXIT 2: Time Exit
            bars_in_trade += 1
            if bars_in_trade >= N_HOLD_:
                pnl  = (close - entry_price) * trade_pos_size
                cost = close * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp, 'action': 'close_long_time',
                    'close': close, 'pnl': pnl, 'cost': cost, 'capital': capital
                })
                position, entry_price, trade_pos_size, bars_in_trade = 'flat', 0.0, 0.0, 0
                continue

        # ── SHORT MANAGEMENT ──────────────────────────────────────────────────
        elif position == 'short':

            # EXIT 1: Stop Loss — close above EMA 12
            if close > row['ema_12']:
                pnl  = (entry_price - row['ema_12']) * trade_pos_size
                cost = row['ema_12'] * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp, 'action': 'stop_loss_short',
                    'close': row['ema_12'], 'pnl': pnl, 'cost': cost, 'capital': capital
                })
                position, entry_price, trade_pos_size, bars_in_trade = 'flat', 0.0, 0.0, 0
                continue

            # EXIT 2: Time Exit
            bars_in_trade += 1
            if bars_in_trade >= N_HOLD_:
                pnl  = (entry_price - close) * trade_pos_size
                cost = close * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp, 'action': 'close_short_time',
                    'close': close, 'pnl': pnl, 'cost': cost, 'capital': capital
                })
                position, entry_price, trade_pos_size, bars_in_trade = 'flat', 0.0, 0.0, 0
                continue


        # ── ENTRY BLOCK ───────────────────────────────────────────────────────
        else:
            if bull and proba_buy >= CONFIDENCE_THRESHOLD:
                trade_pos_size = (capital * position_size) / close
                cost = close * trade_pos_size * COST_PCT
                capital -= cost
                position, entry_price, bars_in_trade = 'long', close, 0
                trade_log.append({
                    'timestamp': timestamp, 'action': 'enter_long',
                    'close': close, 'pnl': 0, 'cost': cost, 'capital': capital
                })

            elif bear and proba_sell >= CONFIDENCE_THRESHOLD:
                trade_pos_size = (capital * position_size) / close
                cost = close * trade_pos_size * COST_PCT
                capital -= cost
                position, entry_price, bars_in_trade = 'short', close, 0
                trade_log.append({
                    'timestamp': timestamp, 'action': 'enter_short',
                    'close': close, 'pnl': 0, 'cost': cost, 'capital': capital
                })

    return trade_log, capital


# ==============================================================================
# FUNCTION 5 — Performance summary
# ==============================================================================

def compute_performance_summary(trade_log, capital, df, cutoff_date, initial_capital, position_size):

    # --- Setup ---
    trade_df       = pd.DataFrame(trade_log)
    entries        = trade_df[trade_df['action'].isin(['enter_long', 'enter_short'])]
    exits          = trade_df[~trade_df['action'].isin(['enter_long', 'enter_short'])]
    winning_trades = exits[exits['pnl'] > 0]
    losing_trades  = exits[exits['pnl'] < 0]
    total_closed   = len(exits)
    win_rate       = len(winning_trades) / total_closed * 100 if total_closed > 0 else 0
    loss_rate      = len(losing_trades)  / total_closed * 100 if total_closed > 0 else 0

    # --- Risk Metrics ---
    trade_df_sorted = trade_df.sort_values('timestamp')
    trade_df_sorted['capital_return'] = trade_df_sorted['capital'].pct_change()
    mean_return = trade_df_sorted['capital_return'].mean()
    std_return  = trade_df_sorted['capital_return'].std()

    # --- Backtest Window ---
    backtest_start   = trade_df_sorted['timestamp'].iloc[0].strftime('%d %b %Y')
    backtest_end     = trade_df_sorted['timestamp'].iloc[-1].strftime('%d %b %Y')
    days_in_backtest = (trade_df_sorted['timestamp'].iloc[-1] - trade_df_sorted['timestamp'].iloc[0]).days
    annualised_return = ((capital / initial_capital) ** (365 / days_in_backtest) - 1) * 100 if days_in_backtest > 0 else 0

    # --- Corrected Sharpe Ratio ---
    # Risk-free rate: 4.25% annualised
    # Annualisation based on actual average trade frequency
    entry_times = trade_df[trade_df['action'] == 'enter_long']['timestamp'].values
    exit_times  = trade_df[trade_df['action'] != 'enter_long']['timestamp'].values

    if len(entry_times) == len(exit_times) and len(entry_times) > 0:
        holding_hours   = np.mean([
            (pd.Timestamp(ex) - pd.Timestamp(en)).total_seconds() / 3600
            for en, ex in zip(entry_times, exit_times)
        ])
        trades_per_year = 8760 / holding_hours
    else:
        trades_per_year = total_closed / (days_in_backtest / 365) if days_in_backtest > 0 else 1
        holding_hours   = 8760 / trades_per_year

    rf_per_trade = RISK_FREE_RATE / trades_per_year
    sharpe       = ((mean_return - rf_per_trade) / std_return) * np.sqrt(trades_per_year) if std_return > 0 else 0

    # --- Drawdown ---
    trade_df_sorted['cummax']   = trade_df_sorted['capital'].cummax()
    trade_df_sorted['drawdown'] = (trade_df_sorted['capital'] - trade_df_sorted['cummax']) / trade_df_sorted['cummax']
    max_drawdown = trade_df_sorted['drawdown'].min() * 100

    # --- Profit Factor ---
    gross_profit  = winning_trades['pnl'].sum()
    gross_loss    = losing_trades['pnl'].abs().sum()
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0

    # --- Buy and Hold Benchmark ---
    buy_price   = df.loc[cutoff_date:, 'Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    btc_units   = initial_capital / buy_price
    bnh_value   = btc_units * final_price
    bnh_return  = ((bnh_value - initial_capital) / initial_capital) * 100

    # --- Transaction Costs ---
    total_transaction_costs = trade_df['cost'].sum()

    # --- Position Sizing Analysis ---
    avg_capital     = (initial_capital + capital) / 2
    avg_pos_usd     = avg_capital * position_size
    avg_pos_btc     = avg_pos_usd / entries['close'].mean() if len(entries) > 0 else 0

    # --- Action Breakdown ---
    action_breakdown = trade_df['action'].value_counts().to_dict()

    # --- Equity curve (for Streamlit chart) ---
    equity_curve = (
        trade_df_sorted[['timestamp', 'capital']]
        .rename(columns={'timestamp': 'date', 'capital': 'equity'})
        .to_dict(orient='records')
    )

    # --- Candlestick data — OHLCV for the entire backtest window ---
    # Returns every hourly bar from cutoff_date to end of data
    # Used by Streamlit to render a candlestick chart
    ohlcv_df = df.loc[cutoff_date:, ['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    ohlcv_df.index.name = 'timestamp'
    ohlcv_df = ohlcv_df.reset_index()
    ohlcv_df['timestamp'] = ohlcv_df['timestamp'].astype(str)
    candlestick_data = ohlcv_df.rename(columns={
        'timestamp': 'date',
        'Open':      'open',
        'High':      'high',
        'Low':       'low',
        'Close':     'close',
        'Volume':    'volume',
    }).to_dict(orient='records')

    return {
        # Capital
        'initial_capital':        round(initial_capital, 2),
        'final_capital':          round(capital, 2),
        'total_return_pct':       round((capital - initial_capital) / initial_capital * 100, 2),
        'annualised_return_pct':  round(annualised_return, 2),
        'backtest_start':         backtest_start,
        'backtest_end':           backtest_end,

        # Trade statistics
        'total_trades':           int(len(entries)),
        'winning_trades':         int(len(winning_trades)),
        'losing_trades':          int(len(losing_trades)),
        'win_rate_pct':           round(win_rate, 1),
        'loss_rate_pct':          round(loss_rate, 1),
        'avg_win_pnl':            round(winning_trades['pnl'].mean(), 2) if len(winning_trades) > 0 else 0,
        'avg_loss_pnl':           round(losing_trades['pnl'].mean(), 2)  if len(losing_trades)  > 0 else 0,
        'total_transaction_costs': round(total_transaction_costs, 2),
        'avg_position_usd':       round(avg_pos_usd, 2),
        'avg_position_btc':       round(avg_pos_btc, 6),

        # Risk metrics
        'sharpe_ratio':           round(sharpe, 2),
        'max_drawdown_pct':       round(max_drawdown, 2),
        'profit_factor':          round(profit_factor, 2),

        # Buy and Hold benchmark
        'bnh_buy_price':          round(buy_price, 2),
        'bnh_final_price':        round(final_price, 2),
        'bnh_final_value':        round(bnh_value, 2),
        'bnh_return_pct':         round(bnh_return, 2),
        'strategy_return_pct':    round((capital - initial_capital) / initial_capital * 100, 2),

        # Breakdown
        'action_breakdown':       action_breakdown,

        # Charts
        'equity_curve':           equity_curve,
        'candlestick_data':       candlestick_data,
    }


# ==============================================================================
# MASTER FUNCTION — called by fast.py
# ==============================================================================

def predict(cutoff_date, initial_capital, position_size):
    df                                     = load_data()
    X_train, X_predict, y_train, y_predict = split_data(df, cutoff_date)
    final_model                            = train_model(X_train, y_train)
    trade_log, capital                     = run_execution_engine(df, final_model, X_predict, y_predict,
                                                                  initial_capital, position_size)
    summary                                = compute_performance_summary(trade_log, capital, df, cutoff_date,
                                                                         initial_capital, position_size)
    return summary

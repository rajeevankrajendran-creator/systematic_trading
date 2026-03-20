import pandas as pd
import numpy as np
import lightgbm as lgb

from config import (
    PICKLE_PATH,
    FEATURE_COLS,
    LGBM_PARAMS,
    NUM_BOOST_ROUND,
    CONFIDENCE_THRESHOLD,
    ATR_MULTIPLIER,
    RISK_PCT,
    COST_PCT,
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
    X_train = X[X.index < cutoff_date]
    y_train = y[y.index < cutoff_date]

    # Creating the testing set for X and y
    X_predict = X[X.index >= cutoff_date]
    y_predict = y[y.index >= cutoff_date]  # kept for evaluation only

    print(f'Training on {len(X_train)} bars up to {cutoff_date}')
    print(f'Predicting on {len(X_predict)} bars from {cutoff_date} onwards')

    return X_train, X_predict, y_train, y_predict


# ==============================================================================
# FUNCTION 3 — Train the model
# ==============================================================================

def train_model(X_train, y_train):
    # Class imbalance correction
    scale_pos_weight = float((y_train['label'] == 0).sum() / (y_train['label'] == 1).sum())

    # Setting the parameters needed for the model
    params = LGBM_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    # Packaging the training data into LightGBM's required format
    train_set_lgb = lgb.Dataset(X_train, label=y_train)

    # Training the model
    final_model = lgb.train(
        params=params,
        train_set=train_set_lgb,
        num_boost_round=NUM_BOOST_ROUND,
        callbacks=[lgb.log_evaluation(period=-1)]
    )

    # Generate predictions on post-cutoff data
    return final_model


# ==============================================================================
# FUNCTION 4 — Execution engine
# ==============================================================================

def run_execution_engine(df, final_model, X_predict, y_predict, initial_capital):

    # Generate predictions
    pred_proba_final = final_model.predict(X_predict)

    # Building the signals DataFrame - this is the input to the execution filter
    signals_df = df.loc[X_predict.index, ['Close', 'sma_50', 'sma_200', 'adx', 'atr_14']].copy()
    signals_df['pred_proba'] = pred_proba_final
    signals_df['true_label'] = y_predict.values

    # Setting the parameters of the execution strategy
    # (these come from config but are used directly here)

    # State variables - maintained by the loop, not stored in Dataframe
    position        = 'flat'
    entry_price     = 0.0
    trade_pos_size  = 0.0    # locked at entry, never changes mid-trade
    trade_stop_dist = 0.0    # locked at entry, never changes mid-trade
    capital         = initial_capital

    # Storing the trade logs
    trade_log = []

    for timestamp, row in signals_df.iterrows():
        close   = row['Close']
        sma_50  = row['sma_50']
        sma_200 = row['sma_200']
        adx     = row['adx']
        atr     = row['atr_14']
        proba   = row['pred_proba']

        # Skip bars where indicators are not yet available (NaN warmup period)
        if pd.isna(atr) or pd.isna(adx) or pd.isna(sma_50) or pd.isna(sma_200):
            continue

        # Regime determination — checked every bar regardless of position
        # Bull requires price above both SMAs AND a strong trend (ADX > 25)
        # ADX < 25 = ranging/choppy market — signals unreliable
        bull = (close > sma_200) and (close > sma_50) and (adx > 25)

        # MANAGEMENT BLOCK — only runs when already in a trade (position == 'long')
        # Checks exit conditions in priority order:
        # Stop Loss → Regime Exit → Signal Exit

        if position == 'long':

            # EXIT 1: Stop Loss
            # Hard floor — exits immediately if price falls to locked stop level
            # stop_price is fixed at entry and never moves with ATR
            # This is the primary capital protection mechanism

            stop_price = entry_price - trade_stop_dist

            if close <= stop_price:
                pnl = (close - entry_price) * trade_pos_size
                cost = close * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp,
                    'action':    'stop_loss',
                    'close':     close,
                    'pnl':       pnl,
                    'capital':   capital
                })
                # Reset all trade state — trade_stop_dist also reset to avoid stale values
                position, entry_price, trade_pos_size, trade_stop_dist = 'flat', 0.0, 0.0, 0.0
                continue

            # EXIT 2: Regime Exit
            # If market structure shifts out of bull regime, close the long
            # regardless of model signal — regime gate overrides model
            # MVP is long-only so no point holding in bear/sideways

            if not bull:
                pnl = (close - entry_price) * trade_pos_size
                cost = close * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp,
                    'action':    'close_long_regime',
                    'close':     close,
                    'pnl':       pnl,
                    'capital':   capital
                })
                # Reset all trade state — trade_stop_dist also reset to avoid stale values
                position, entry_price, trade_pos_size, trade_stop_dist = 'flat', 0.0, 0.0, 0.0
                continue

            # EXIT 3: Signal Exit
            # Model confidence has flipped to strong SELL territory
            # Only exits if probability drops below (1 - threshold)
            # Probabilities near 0.5 do NOT trigger exit — only strong SELL signal does

            if proba <= (1 - CONFIDENCE_THRESHOLD):
                pnl     = (close - entry_price) * trade_pos_size
                cost    = close * trade_pos_size * COST_PCT
                capital += pnl - cost
                trade_log.append({
                    'timestamp': timestamp,
                    'action':    'close_long_signal',
                    'close':     close,
                    'pnl':       pnl,
                    'capital':   capital
                })
                position, entry_price, trade_pos_size, trade_stop_dist = 'flat', 0.0, 0.0, 0.0
                continue

        # ENTRY BLOCK - only runs when flat (position == 'flat')
        # All three conditions must be true simultaneously:
        #   1. Bull regime confirmed (regime gate)
        #   2. Model confidence above threshold (signal quality filter)
        # Entry metrics locked here — never recalculated during the trade

        else:
            if bull and proba >= CONFIDENCE_THRESHOLD:

                # Lock stop distance and position size at entry bar ATR
                # These values are frozen for the entire duration of this trade
                trade_stop_dist = ATR_MULTIPLIER * atr
                trade_pos_size  = (capital * RISK_PCT) / trade_stop_dist

                cost = close * trade_pos_size * COST_PCT
                capital -= cost
                position    = 'long'
                entry_price = close

                trade_log.append({
                    'timestamp':  timestamp,
                    'action':     'enter_long',
                    'close':      close,
                    'pnl':        0,
                    'capital':    capital
                })

    return trade_log, capital


# ==============================================================================
# FUNCTION 5 — Performance summary
# ==============================================================================

def compute_performance_summary(trade_log, capital, df, cutoff_date, initial_capital):

    # --- Setup ---
    trade_df       = pd.DataFrame(trade_log)
    entries        = trade_df[trade_df['action'] == 'enter_long']
    exits          = trade_df[trade_df['action'] != 'enter_long']
    winning_trades = exits[exits['pnl'] > 0]
    losing_trades  = exits[exits['pnl'] < 0]
    total_closed   = len(exits)
    win_rate       = len(winning_trades) / total_closed * 100 if total_closed > 0 else 0
    loss_rate      = len(losing_trades)  / total_closed * 100 if total_closed > 0 else 0

    # --- Risk Metrics ---
    trade_df_sorted = trade_df.sort_values('timestamp')
    trade_df_sorted['capital_return'] = trade_df_sorted['capital'].pct_change()
    mean_return      = trade_df_sorted['capital_return'].mean()
    std_return       = trade_df_sorted['capital_return'].std()
    sharpe           = (mean_return / std_return) * np.sqrt(8760)
    trade_df_sorted['cummax']   = trade_df_sorted['capital'].cummax()
    trade_df_sorted['drawdown'] = (trade_df_sorted['capital'] - trade_df_sorted['cummax']) / trade_df_sorted['cummax']
    max_drawdown     = trade_df_sorted['drawdown'].min() * 100
    gross_profit     = winning_trades['pnl'].sum()
    gross_loss       = losing_trades['pnl'].abs().sum()
    profit_factor    = gross_profit / gross_loss if gross_loss > 0 else 0
    days_in_backtest = (trade_df_sorted['timestamp'].iloc[-1] - trade_df_sorted['timestamp'].iloc[0]).days
    annualised_return = ((capital / initial_capital) ** (365 / days_in_backtest) - 1) * 100

    # --- Buy and Hold Benchmark ---
    buy_date    = cutoff_date
    buy_price   = df.loc[buy_date:, 'Close'].iloc[0]
    final_price = df['Close'].iloc[-1]
    btc_units   = initial_capital / buy_price
    bnh_value   = btc_units * final_price
    bnh_return  = ((bnh_value - initial_capital) / initial_capital) * 100

    # --- Transaction Costs ---
    implied_costs = trade_df['pnl'].sum() - (capital - initial_capital)

    # --- Action Breakdown ---
    action_breakdown = trade_df['action'].value_counts().to_dict()

    # --- Equity curve (for Streamlit chart) ---
    equity_curve = (
        trade_df_sorted[['timestamp', 'capital']]
        .rename(columns={'timestamp': 'date', 'capital': 'equity'})
        .to_dict(orient='records')
    )

    # Return all metrics as a dict for the API / Streamlit
    return {
        'initial_capital':       round(initial_capital, 2),
        'final_capital':         round(capital, 2),
        'total_return_pct':      round((capital - initial_capital) / initial_capital * 100, 2),
        'annualised_return_pct': round(annualised_return, 2),
        'total_trades':          int(len(entries)),
        'winning_trades':        int(len(winning_trades)),
        'losing_trades':         int(len(losing_trades)),
        'win_rate_pct':          round(win_rate, 1),
        'loss_rate_pct':         round(loss_rate, 1),
        'avg_win_pnl':           round(winning_trades['pnl'].mean(), 2) if len(winning_trades) > 0 else 0,
        'avg_loss_pnl':          round(losing_trades['pnl'].mean(), 2)  if len(losing_trades)  > 0 else 0,
        'implied_costs':         round(implied_costs, 2),
        'sharpe_ratio':          round(sharpe, 2),
        'max_drawdown_pct':      round(max_drawdown, 2),
        'profit_factor':         round(profit_factor, 2),
        'bnh_buy_price':         round(buy_price, 2),
        'bnh_final_price':       round(final_price, 2),
        'bnh_final_value':       round(bnh_value, 2),
        'bnh_return_pct':        round(bnh_return, 2),
        'strategy_return_pct':   round((capital - initial_capital) / initial_capital * 100, 2),
        'action_breakdown':      action_breakdown,
        'equity_curve':          equity_curve,
    }


# ==============================================================================
# MASTER FUNCTION — called by fast.py
# ==============================================================================

def predict(cutoff_date, initial_capital):
    df                                     = load_data()
    X_train, X_predict, y_train, y_predict = split_data(df, cutoff_date)
    final_model                            = train_model(X_train, y_train)
    trade_log, capital                     = run_execution_engine(df, final_model, X_predict, y_predict, initial_capital)
    summary                                = compute_performance_summary(trade_log, capital, df, cutoff_date, initial_capital)
    return summary

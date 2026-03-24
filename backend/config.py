# ============================================================
# CONFIG — V5.1
# ============================================================
# Edit this file between experiments.
# CUTOFF_DATE, INITIAL_CAPITAL, POSITION_SIZE are NOT here —
# they are user inputs sent to the API.
# ============================================================

# --- Path
PICKLE_PATH = 'raw_data/preprocessed.pkl'

# --- Lag configuration — variable depth per feature
# Matches exactly what the notebook builds
LAG_CONFIG = {
    'macd_histogram': 12,
    'rsi_14':          5,
    'roc_21':         12,
    'Volume':         12,
    'roc_10':          5,
    'atr_14':         12,
    'natr':           12,
    'bandwidth':      12,
    'obv_change':     12,
    'adx':             5,
}

# No-lag features — cyclical time features, computed from index
NO_LAG_COLS = ['month_cos', 'month_sin', 'dow_sin', 'dow_cos']

# Feature matrix X — generated from LAG_CONFIG + NO_LAG_COLS
# Do not edit manually — derived automatically below
_base_cols    = list(LAG_CONFIG.keys())
_all_lag_cols = [
    f'{col}_lag{lag}'
    for col, n_lags in LAG_CONFIG.items()
    for lag in range(1, n_lags + 1)
]
FEATURE_COLS = _base_cols + _all_lag_cols + NO_LAG_COLS

# --- LightGBM parameters — multiclass (SELL=0, FLAT=1, BUY=2)
LGBM_PARAMS = {
    'objective':        'multiclass',
    'metric':           'multi_logloss',
    'num_class':        3,
    'boosting_type':    'gbdt',
    'learning_rate':    0.05,
    'num_leaves':       31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq':     5,
    'verbose':          -1,
}

NUM_BOOST_ROUND = 300

# --- Execution engine parameters
CONFIDENCE_THRESHOLD = 0.55   # minimum proba_buy or proba_sell to enter
COST_PCT             = 0.001  # 0.1% transaction cost per side
N_HOLD               = 12     # exit after N bars (time exit / patience)

# --- Sharpe ratio
RISK_FREE_RATE = 0.0425       # 4.25% annualised

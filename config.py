'''
CONFIG
Edit this file between experiments.
CUTOFF_DATE and INITIAL_CAPITAL is NOT here — it is user input sent to the API.
'''

# --- Path
PICKLE_PATH = 'raw_data/preprocessed.pkl'

# --- Feature matrix X
# Exact columns built by the notebook — must match the pickle
FEATURE_COLS = [
    'Volume', 'roc_10', 'roc_21', 'macd_histogram', 'adx',
    'Volume_lag1', 'Volume_lag2', 'Volume_lag3', 'Volume_lag4', 'Volume_lag5',
    'roc_10_lag1', 'roc_10_lag2', 'roc_10_lag3', 'roc_10_lag4', 'roc_10_lag5',
    'roc_21_lag1', 'roc_21_lag2', 'roc_21_lag3', 'roc_21_lag4', 'roc_21_lag5',
    'macd_histogram_lag1', 'macd_histogram_lag2', 'macd_histogram_lag3', 'macd_histogram_lag4', 'macd_histogram_lag5',
    'adx_lag1', 'adx_lag2', 'adx_lag3', 'adx_lag4', 'adx_lag5',
]

# --- LightGBM parameters
# scale_pos_weight is computed at runtime from the data — not set here
LGBM_PARAMS = {
    'objective':         'binary',
    'metric':            'binary_logloss',
    'boosting_type':     'gbdt',
    'learning_rate':     0.05,
    'num_leaves':        31,
    'feature_fraction':  0.8,
    'bagging_fraction':  0.8,
    'bagging_freq':      5,
    'seed':              42,
    'verbose':           -1,
}

NUM_BOOST_ROUND = 300

# --- Execution engine parameters
CONFIDENCE_THRESHOLD = 0.55
ATR_MULTIPLIER       = 1.5
RISK_PCT             = 0.01   # risk 1% of capital per trade
COST_PCT             = 0.001  # 0.1% transaction cost per side

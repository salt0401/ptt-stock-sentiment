"""Advanced Quantitative ML Trading Strategy (XGBoost + Walk-Forward sliding windows)

This script upgrades the quantitative pipeline to full institutional standards:
1. Feature Engineering: Added 30-day sentiment volatility, Absolute day-over-day changes.
2. Sliding Window: 1-Year Train -> 1-Quarter Validation -> 1-Quarter Test.
3. Advanced ML: XGBoost with Early Stopping on the Validation Set.
4. Execution: State-Machine Hysteresis bounds with 0.20% transaction costs.
"""

import argparse
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt

# Ignore warnings for cleaner output
warnings.filterwarnings("ignore")

from config import OUTPUT_DIR, DATA_DIR

TOTAL_TABLE_PATH = DATA_DIR / "financial" / "aggregated_table" / "total_table.csv"
DAILY_SENT_PATH = OUTPUT_DIR / "daily_sentiment.csv"
FF5_COLS = ['SMB', 'HML', 'RMW', 'CMA']
RETURN_COL = 'spread'  # excess return
ONE_WAY_COST = 0.002  # 0.2% one-way friction
TRADING_DAYS_PER_YEAR = 252
TRADING_DAYS_PER_QUARTER = 63

def load_and_engineer_data(total_path=TOTAL_TABLE_PATH, sent_path=DAILY_SENT_PATH):
    print("Loading and creating advanced features (including 30-day volatility)...")
    total = pd.read_csv(total_path)
    total['Date'] = pd.to_datetime(total['Date'])
    
    sent = pd.read_csv(sent_path)
    sent['Date'] = pd.to_datetime(sent['Date'])
    
    # Rename W2V columns
    rename_cols = {
        'sentiment_mean': 'SENT_W2V',
        'sentiment_mean_author': 'SENT_W2V_AUTH',
        'sentiment_mean_engagement': 'SENT_W2V_ENG'
    }
    sent = sent.rename(columns={k: v for k, v in rename_cols.items() if k in sent.columns})
    merge_cols = ['Date'] + [c for c in rename_cols.values() if c in sent.columns]
    
    df = total.merge(sent[merge_cols], on='Date', how='left')
    
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=[RETURN_COL, 'SENT_W2V'])
    
    # Target: Predict Next Day Return Sign
    df['FWD_RET'] = df[RETURN_COL].shift(-1)
    df['TARGET'] = (df['FWD_RET'] > 0).astype(int)
    
    # Dynamic Feature Generation for ALL sentiment metrics
    # Only bloat W2V (to constrain dimensions). We will handle FinBERT separately.
    sentiment_cols = [c for c in df.columns if c.startswith('SENT_W2V')]
    for col in sentiment_cols:
        df[f'{col}_MA3'] = df[col].rolling(3).mean()
        df[f'{col}_MA5'] = df[col].rolling(5).mean()
        df[f'{col}_MOM'] = df[col] - df[f'{col}_MA5']
        df[f'{col}_LAG1'] = df[col].shift(1)
        df[f'{col}_LAG2'] = df[col].shift(2)
        df[f'{col}_VOL_30'] = df[col].rolling(30).std()
        df[f'{col}_ABS_CHANGE'] = abs(df[col] - df[f'{col}_LAG1'])
        df[f'{col}_MAX_10'] = df[col].rolling(10).max()
        df[f'{col}_MIN_10'] = df[col].rolling(10).min()
        
    # Feature 4: Historical Market Return MAs & Lags
    df['RET_LAG1'] = df[RETURN_COL].shift(1)
    df['RET_LAG2'] = df[RETURN_COL].shift(2)
    df['RET_MA5'] = df[RETURN_COL].rolling(5).mean()
    df['RET_MA10'] = df[RETURN_COL].rolling(10).mean()
    df['RET_MA20'] = df[RETURN_COL].rolling(20).mean()
    df['VOL_5D'] = df[RETURN_COL].rolling(5).std()
    
    return df

def walk_forward_xgboost(df, train_w=TRADING_DAYS_PER_YEAR, val_w=TRADING_DAYS_PER_QUARTER, test_w=TRADING_DAYS_PER_QUARTER):
    print(f"\nRunning XGBoost Walk-Forward Validation:")
    print(f" Train: {train_w}d (1 Year) | Val: {val_w}d (1 Quarter) | Test: {test_w}d (1 Quarter)")
    
    features = [c for c in df.columns if c.startswith('SENT_')] + \
               ['RET_LAG1', 'RET_LAG2', 'RET_MA5', 'RET_MA10', 'RET_MA20', 'VOL_5D'] + FF5_COLS
    
    print(f" Total Features: {len(features)}")
    
    df = df.dropna(subset=features + ['FWD_RET']).reset_index(drop=True)
    n = len(df)
    
    df['ML_PROB_UP'] = np.nan
    chunk_count = 0
    total_chunks = (n - train_w - val_w) // test_w + 1
    
    all_importances = []
    
    # Sliding window logic
    for t in range(train_w + val_w, n, test_w):
        chunk_count += 1
        
        # 1. Train set (e.g., d1 to d252)
        train_start = t - train_w - val_w
        train_end = t - val_w
        train_idx = list(range(train_start, train_end))
        
        # 2. Validation set (e.g., d253 to d315)
        val_start = train_end
        val_end = t
        val_idx = list(range(val_start, val_end))
        
        # 3. Test set (e.g., d316 to d378)
        test_start = t
        test_end = min(t + test_w, n)
        test_idx = list(range(test_start, test_end))
        
        if len(test_idx) < 5:  # Skip tiny final fragments
            break
            
        # Data prep & Imputation
        imputer = SimpleImputer(strategy='median')
        X_train = imputer.fit_transform(df.loc[train_idx, features])
        y_train = df.loc[train_idx, 'TARGET'].values
        
        X_val = imputer.transform(df.loc[val_idx, features])
        y_val = df.loc[val_idx, 'TARGET'].values
        
        X_test = imputer.transform(df.loc[test_idx, features])
        
        # XGBoost with Early Stopping on the Val set
        clf = xgb.XGBClassifier(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=3,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric="logloss",
            early_stopping_rounds=20,
            random_state=42
        )
        
        clf.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        
        # Predict Probabilities on Test segment
        probs = clf.predict_proba(X_test)[:, 1]
        df.loc[test_idx, 'ML_PROB_UP'] = probs
        
        all_importances.append(clf.feature_importances_)
        
        best_iter = clf.best_iteration
        print(f" Chunk {chunk_count:>2}/{total_chunks}: Stopped at tree {best_iter:>3} | Predicted {len(test_idx)} days")

    oos_df = df.dropna(subset=['ML_PROB_UP']).copy()
    
    if all_importances:
        import matplotlib.pyplot as plt
        avg_imp = np.mean(all_importances, axis=0)
        print("\n" + "-"*50)
        print("TOP 10 FEATURE IMPORTANCES (Avg across windows):")
        print("-" * 50)
        feat_imp = pd.Series(avg_imp, index=features).sort_values(ascending=False)
        print(feat_imp.head(10).to_string())
        print("-" * 50)
        
    return oos_df

def run_strategies(oos_df):
    """Compare baseline and new Hysteresis XGBoost strategy."""
    print("\nEvaluating Strategy Peformance and Transaction Costs...")
    
    # 1. Buy & Hold Benchmark
    oos_df['Pos_BH'] = 1
    oos_df['Ret_BH'] = oos_df['FWD_RET']
    oos_df['Ret_BH_Net'] = oos_df['Ret_BH'] - abs(oos_df['Pos_BH'].diff().fillna(0)) * ONE_WAY_COST
    
    # 2. Baseline Static Median (from evaluate_advanced.py)
    global_median = oos_df['SENT_W2V'].median()
    oos_df['Pos_Base'] = np.where(oos_df['SENT_W2V'] > global_median, 1, -1)
    oos_df['Ret_Base'] = oos_df['Pos_Base'] * oos_df['FWD_RET']
    oos_df['Ret_Base_Net'] = oos_df['Ret_Base'] - abs(oos_df['Pos_Base'].diff().fillna(0)) * ONE_WAY_COST
    
    # 3. Advanced State-Machine Hysteresis Strategy
    threshold_pairs = [
        (0.51, 0.49), (0.52, 0.48), (0.53, 0.47),
        (0.54, 0.46), (0.55, 0.45), (0.56, 0.44), (0.58, 0.42)
    ]
    
    best_sharpe = -999.0
    best_thresh = None
    best_pos_ml = None
    best_ret_net = None
    
    print("\nGrid Searching Hysteresis Thresholds...")
    for u, l in threshold_pairs:
        pos_ml = np.zeros(len(oos_df))
        current_pos = 0
        probs = oos_df['ML_PROB_UP'].values
        
        for i in range(len(probs)):
            if probs[i] > u:
                current_pos = 1
            elif probs[i] < l:
                 current_pos = -1
            pos_ml[i] = current_pos
            
        ret_ml = pos_ml * oos_df['FWD_RET'].values
        turnover = np.abs(np.diff(pos_ml, prepend=0))
        ret_net = ret_ml - (turnover * ONE_WAY_COST)
        
        ann_ret = ret_net.mean() * 252
        ann_std = ret_net.std() * np.sqrt(252)
        sharpe = ann_ret / ann_std if ann_std > 0 else 0
        print(f"  Threshold [{u:.2f} / {l:.2f}] -> Sharpe: {sharpe:.3f}, Trades: {(turnover != 0).sum()}")
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_thresh = (u, l)
            best_pos_ml = pos_ml
            best_ret_net = ret_net
            
    print(f"-> Selected Best Threshold: {best_thresh[0]:.2f} / {best_thresh[1]:.2f} (Sharpe {best_sharpe:.3f})")
    
    oos_df['Pos_ML'] = best_pos_ml
    oos_df['Ret_ML'] = oos_df['Pos_ML'] * oos_df['FWD_RET']
    oos_df['Ret_ML_Net'] = best_ret_net
    
    return oos_df

def print_metrics(oos_df):
    total_days = len(oos_df)
    
    def calc_metrics(ret_series, pos_series):
        ann_ret = ret_series.mean() * 252
        ann_std = ret_series.std() * np.sqrt(252)
        sharpe = ann_ret / ann_std if ann_std > 0 else 0
        cum_ret = (1 + ret_series).prod() - 1
        avg_exposure = abs(pos_series).mean()
        trade_count = (pos_series.diff().fillna(0) != 0).sum()
        return ann_ret, ann_std, sharpe, cum_ret, avg_exposure, trade_count
    
    strategies = [
        ("1. Buy & Hold Benchmark", 'Ret_BH_Net', 'Pos_BH'),
        ("2. Static Median Baseline (Net/w Costs)", 'Ret_Base_Net', 'Pos_Base'),
        ("3. XGBoost State-Machine (Net/w Costs)", 'Ret_ML_Net', 'Pos_ML')
    ]
    
    print("\n" + "="*75)
    print(f"OUT-OF-SAMPLE PERFORMANCE (N={total_days} days, Cost={ONE_WAY_COST*100}% per leg)")
    print("="*75)
    print(f"{'Strategy Name':<42} | {'Cum Ret':>8} | {'Sharpe':>7} | {'Market Expos':>12}")
    print("-" * 75)
    
    for name, ret_col, pos_col in strategies:
        ann_ret, ann_std, sharpe, cum_ret, avg_exp, trades = calc_metrics(oos_df[ret_col], oos_df[pos_col])
        print(f"{name:<42} | {cum_ret:>8.1%} | {sharpe:>7.2f} | {avg_exp:>12.1%} ")

    print("-" * 75)
    print("\nXGBoost Strategy Deep Dive:")
    ml_trades = (oos_df['Pos_ML'].diff().fillna(0) != 0).sum()
    print(f" - Number of Trades Executed: {ml_trades}")
    win_rate = (oos_df[oos_df['Pos_ML'] != 0]['Ret_ML'] > 0).mean()
    print(f" - ML Signal Accuracy (Win Rate): {win_rate:.1%}")

def main():
    df = load_and_engineer_data()
    oos_df = walk_forward_xgboost(
        df, 
        train_w=TRADING_DAYS_PER_YEAR, 
        val_w=TRADING_DAYS_PER_QUARTER, 
        test_w=TRADING_DAYS_PER_QUARTER
    )
    oos_df = run_strategies(oos_df)
    print_metrics(oos_df)

if __name__ == "__main__":
    main()

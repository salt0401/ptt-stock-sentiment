"""Advanced sentiment evaluation: predictive tests, trading, and non-linear analysis.

Six tests that go beyond contemporaneous FF5 regression:

1. Predictive regression (lag-1): Does today's sentiment predict tomorrow's return?
2. Granger causality: Does sentiment lead returns, or vice versa?
3. Out-of-sample R-sq: Rolling-window forecast vs historical mean benchmark.
4. Long-short portfolio backtest: Trade on sentiment signal -> Sharpe ratio.
5. Information Coefficient (IC): Rank correlation of SENT_t vs return_{t+1}.
6. Quantile regression: Does sentiment matter more in extreme markets?

Usage:
    python evaluate_advanced.py
    python evaluate_advanced.py --window 252 --lags 5
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import OUTPUT_DIR, DATA_DIR
from evaluate_ff5_regression import _nw_lag

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

TOTAL_TABLE_PATH = DATA_DIR / "financial" / "aggregated_table" / "total_table.csv"
DAILY_SENT_PATH = OUTPUT_DIR / "daily_sentiment.csv"

FF5_COLS = ['SMB', 'HML', 'RMW', 'CMA']
RETURN_COL = 'spread'  # excess return = return - RiskFreeRate


def load_merged_data(total_path=None, sent_path=None):
    """Load total_table and merge with daily sentiment.

    Returns
    -------
    pd.DataFrame : Sorted by date, with columns: Date, spread, SMB, HML, RMW,
        CMA, SENT, SENT_lag1, FWD_RET (next-day return).
    """
    if total_path is None:
        total_path = TOTAL_TABLE_PATH
    if sent_path is None:
        sent_path = DAILY_SENT_PATH

    total = pd.read_csv(total_path)
    total['Date'] = pd.to_datetime(total['Date'])

    sent = pd.read_csv(sent_path)
    sent['Date'] = pd.to_datetime(sent['Date'])

    df = total.merge(sent[['Date', 'sentiment_mean']], on='Date', how='left')
    df.rename(columns={'sentiment_mean': 'SENT'}, inplace=True)
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.dropna(subset=[RETURN_COL, 'SENT'] + FF5_COLS)

    # Lag-1 sentiment and forward return
    df['SENT_lag1'] = df['SENT'].shift(1)
    df['FWD_RET'] = df[RETURN_COL].shift(-1)

    return df


# ===================================================================
# TEST 1: Predictive regression (lag-1)
# ===================================================================

def test_predictive_regression(df, verbose=True):
    """Compare contemporaneous vs predictive (lag-1) FF5+SENT regression.

    Contemporaneous: return_t = FF5_t + SENT_t
    Predictive:      return_t = FF5_t + SENT_{t-1}

    Both use HAC (Newey-West) standard errors.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEST 1: PREDICTIVE REGRESSION (Lag-1)")
        print("=" * 70)

    results = {}

    for label, sent_col in [('Contemporaneous', 'SENT'), ('Predictive (lag-1)', 'SENT_lag1')]:
        subset = df.dropna(subset=[sent_col])
        y = subset[RETURN_COL]
        X = sm.add_constant(subset[FF5_COLS + [sent_col]])

        lag = _nw_lag(len(y))
        res = sm.OLS(y, X, missing='drop').fit(cov_type='HAC', cov_kwds={'maxlags': lag})

        coef = res.params.get(sent_col, np.nan)
        tstat = res.tvalues.get(sent_col, np.nan)
        pval = res.pvalues.get(sent_col, np.nan)

        results[label] = {
            'coef': coef, 'tstat': tstat, 'pval': pval,
            'r2': res.rsquared, 'adj_r2': res.rsquared_adj,
            'n': int(res.nobs),
        }

        if verbose:
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            print(f"\n  {label}:")
            print(f"    SENT coef:  {coef:.6f}")
            print(f"    t-stat:     {tstat:.4f}")
            print(f"    p-value:    {pval:.4e}  {sig}")
            print(f"    R-sq:         {res.rsquared:.6f}")
            print(f"    N:          {int(res.nobs)}")

    if verbose:
        print(f"\n  Key question: Does lag-1 sentiment still predict returns?")
        p_pred = results['Predictive (lag-1)']['pval']
        if p_pred < 0.05:
            print(f"  -> YES (p={p_pred:.4f}). Sentiment has genuine predictive power.")
        elif p_pred < 0.1:
            print(f"  -> Marginal (p={p_pred:.4f}). Weak evidence of predictive power.")
        else:
            print(f"  -> NO (p={p_pred:.4f}). Contemporaneous only — no forecasting power.")

    return results


# ===================================================================
# TEST 2: Granger causality
# ===================================================================

def test_granger_causality(df, max_lags=5, verbose=True):
    """Bidirectional Granger causality test: SENT <-> returns.

    Tests whether past SENT helps predict future returns (and vice versa),
    beyond what past returns alone can predict.
    """
    from statsmodels.tsa.stattools import grangercausalitytests

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 2: GRANGER CAUSALITY")
        print("=" * 70)

    subset = df[[RETURN_COL, 'SENT']].dropna()

    results = {}

    for direction, data_cols in [
        ('SENT -> Returns', [RETURN_COL, 'SENT']),
        ('Returns -> SENT', ['SENT', RETURN_COL]),
    ]:
        test_data = subset[data_cols].values
        gc = grangercausalitytests(test_data, maxlag=max_lags, verbose=False)

        if verbose:
            print(f"\n  {direction}:")
            print(f"    {'Lag':<6} {'F-stat':>10} {'p-value':>12} {'Sig':>6}")
            print(f"    {'-'*36}")

        lag_results = {}
        for lag in range(1, max_lags + 1):
            fstat = gc[lag][0]['ssr_ftest'][0]
            pval = gc[lag][0]['ssr_ftest'][1]
            sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''
            lag_results[lag] = {'fstat': fstat, 'pval': pval}

            if verbose:
                print(f"    {lag:<6} {fstat:>10.4f} {pval:>12.4e} {sig:>6}")

        results[direction] = lag_results

    if verbose:
        # Summarize
        best_sent_to_ret = min(results['SENT -> Returns'].values(), key=lambda x: x['pval'])
        best_ret_to_sent = min(results['Returns -> SENT'].values(), key=lambda x: x['pval'])
        print(f"\n  Best SENT -> Returns p-value: {best_sent_to_ret['pval']:.4e}")
        print(f"  Best Returns -> SENT p-value: {best_ret_to_sent['pval']:.4e}")

        if best_sent_to_ret['pval'] < 0.05 and best_ret_to_sent['pval'] >= 0.05:
            print("  -> Sentiment Granger-causes returns (unidirectional)")
        elif best_sent_to_ret['pval'] >= 0.05 and best_ret_to_sent['pval'] < 0.05:
            print("  -> Returns Granger-cause sentiment (reverse causality!)")
        elif best_sent_to_ret['pval'] < 0.05 and best_ret_to_sent['pval'] < 0.05:
            print("  -> Bidirectional Granger causality (feedback loop)")
        else:
            print("  -> No significant Granger causality in either direction")

    return results


# ===================================================================
# TEST 3: Out-of-sample R-sq
# ===================================================================

def test_oos_r2(df, window=252, verbose=True):
    """Rolling-window out-of-sample R-sq (Campbell & Thompson 2008).

    Benchmark: historical mean return (expanding window).
    Model: FF5 + SENT_{t-1} trained on rolling window.

    OOS R-sq = 1 - Σ(r_t - r̂_model)² / Σ(r_t - r̄_hist)²

    Positive OOS R-sq means the model beats the historical mean.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEST 3: OUT-OF-SAMPLE R-sq (Rolling Window)")
        print("=" * 70)

    subset = df.dropna(subset=['SENT_lag1', RETURN_COL]).reset_index(drop=True)
    n = len(subset)

    if n <= window + 10:
        if verbose:
            print(f"  [WARN] Not enough data (n={n}, window={window})")
        return {}

    y_all = subset[RETURN_COL].values
    X_cols = FF5_COLS + ['SENT_lag1']

    model_errors_sq = []
    bench_errors_sq = []
    ff5_only_errors_sq = []
    dates = []

    for t in range(window, n):
        # Training window
        train_y = y_all[t - window:t]
        train_X = subset[X_cols].iloc[t - window:t]
        train_X_ff5 = subset[FF5_COLS].iloc[t - window:t]

        # Test point
        test_y = y_all[t]
        test_X = subset[X_cols].iloc[t:t + 1]
        test_X_ff5 = subset[FF5_COLS].iloc[t:t + 1]

        # Benchmark: expanding mean
        hist_mean = y_all[:t].mean()

        # FF5+SENT model
        try:
            X_train_c = sm.add_constant(train_X)
            # Force has_constant='add' for single-row test data — default 'skip'
            # incorrectly detects zero variance in every column and omits the constant
            X_test_c = sm.add_constant(test_X, has_constant='add')
            res = sm.OLS(train_y, X_train_c, missing='drop').fit()
            pred = res.predict(X_test_c).values[0]
        except Exception:
            continue

        # FF5-only model
        try:
            X_train_ff5_c = sm.add_constant(train_X_ff5)
            X_test_ff5_c = sm.add_constant(test_X_ff5, has_constant='add')
            res_ff5 = sm.OLS(train_y, X_train_ff5_c, missing='drop').fit()
            pred_ff5 = res_ff5.predict(X_test_ff5_c).values[0]
        except Exception:
            continue

        model_errors_sq.append((test_y - pred) ** 2)
        ff5_only_errors_sq.append((test_y - pred_ff5) ** 2)
        bench_errors_sq.append((test_y - hist_mean) ** 2)
        dates.append(subset['Date'].iloc[t])

    model_errors_sq = np.array(model_errors_sq)
    ff5_only_errors_sq = np.array(ff5_only_errors_sq)
    bench_errors_sq = np.array(bench_errors_sq)

    if len(model_errors_sq) == 0:
        if verbose:
            print(f"  [ERROR] 0 successful predictions out of {n - window} windows.")
        return {}

    # OOS R-sq vs historical mean
    oos_r2_vs_mean = 1 - model_errors_sq.sum() / bench_errors_sq.sum()
    oos_r2_ff5_vs_mean = 1 - ff5_only_errors_sq.sum() / bench_errors_sq.sum()

    # OOS R-sq of FF5+SENT vs FF5-only
    oos_r2_incremental = 1 - model_errors_sq.sum() / ff5_only_errors_sq.sum()

    results = {
        'oos_r2_vs_mean': oos_r2_vs_mean,
        'oos_r2_ff5_vs_mean': oos_r2_ff5_vs_mean,
        'oos_r2_incremental': oos_r2_incremental,
        'n_predictions': len(model_errors_sq),
        'window': window,
        'rmse_model': np.sqrt(model_errors_sq.mean()),
        'rmse_ff5': np.sqrt(ff5_only_errors_sq.mean()),
        'rmse_bench': np.sqrt(bench_errors_sq.mean()),
    }

    if verbose:
        print(f"\n  Rolling window: {window} days")
        print(f"  Out-of-sample predictions: {len(model_errors_sq)}")
        print(f"  Period: {dates[0].strftime('%Y-%m-%d')} to {dates[-1].strftime('%Y-%m-%d')}")
        print(f"\n  RMSE:")
        print(f"    Historical mean (benchmark): {results['rmse_bench']:.6f}")
        print(f"    FF5-only model:              {results['rmse_ff5']:.6f}")
        print(f"    FF5+SENT model:              {results['rmse_model']:.6f}")
        print(f"\n  Out-of-sample R-sq:")
        print(f"    FF5-only vs hist mean:       {oos_r2_ff5_vs_mean:+.4f}")
        print(f"    FF5+SENT vs hist mean:       {oos_r2_vs_mean:+.4f}")
        print(f"    FF5+SENT vs FF5-only:        {oos_r2_incremental:+.6f}")
        print(f"\n  Interpretation:")
        if oos_r2_incremental > 0:
            print(f"  -> Adding SENT reduces OOS forecast error by {oos_r2_incremental:.4%}")
        else:
            print(f"  -> Adding SENT increases OOS forecast error (overfitting)")

    return results


# ===================================================================
# TEST 4: Long-short portfolio backtest
# ===================================================================

def test_long_short_backtest(df, verbose=True):
    """Backtest a long-short strategy based on sentiment signal.

    Strategy: Each day t, observe SENT_t.
    - If SENT_t > median -> go long next day
    - If SENT_t < median -> go short next day
    - Compute daily P&L, cumulative return, Sharpe ratio.

    Also tests quintile sorts.
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEST 4: LONG-SHORT PORTFOLIO BACKTEST")
        print("=" * 70)

    subset = df.dropna(subset=['SENT', 'FWD_RET']).copy()
    n = len(subset)

    # --- Median split ---
    median_sent = subset['SENT'].median()
    subset['signal'] = np.where(subset['SENT'] > median_sent, 1, -1)
    subset['strategy_ret'] = subset['signal'] * subset['FWD_RET']

    # Annualization factor (252 trading days)
    mean_daily = subset['strategy_ret'].mean()
    std_daily = subset['strategy_ret'].std()
    sharpe = (mean_daily / std_daily) * np.sqrt(252) if std_daily > 0 else 0

    cum_ret = (1 + subset['strategy_ret']).prod() - 1
    cum_bh = (1 + subset['FWD_RET']).prod() - 1  # buy-and-hold benchmark

    # --- Quintile sorts ---
    subset['quintile'] = pd.qcut(subset['SENT'], 5, labels=[1, 2, 3, 4, 5])
    quintile_rets = subset.groupby('quintile')['FWD_RET'].agg(['mean', 'std', 'count'])
    quintile_rets['annualized_mean'] = quintile_rets['mean'] * 252
    quintile_rets['annualized_std'] = quintile_rets['std'] * np.sqrt(252)

    # Long-short spread: Q5 (highest sentiment) minus Q1 (lowest)
    q5_mean = subset[subset['quintile'] == 5]['FWD_RET'].mean()
    q1_mean = subset[subset['quintile'] == 1]['FWD_RET'].mean()
    ls_spread = q5_mean - q1_mean
    ls_spread_annual = ls_spread * 252

    results = {
        'n_days': n,
        'sharpe_ratio': sharpe,
        'mean_daily_ret': mean_daily,
        'std_daily_ret': std_daily,
        'cumulative_return': cum_ret,
        'buy_hold_return': cum_bh,
        'ls_spread_daily': ls_spread,
        'ls_spread_annual': ls_spread_annual,
        'quintile_returns': quintile_rets,
    }

    if verbose:
        print(f"\n  Trading days: {n}")
        print(f"  Sentiment median: {median_sent:.6f}")

        print(f"\n  Median-split strategy (long high SENT, short low SENT):")
        print(f"    Mean daily return:   {mean_daily:+.6f} ({mean_daily * 252:+.4f} annual)")
        print(f"    Std daily return:    {std_daily:.6f}")
        print(f"    Sharpe ratio:        {sharpe:+.4f}")
        print(f"    Cumulative return:   {cum_ret:+.2%}")
        print(f"    Buy-and-hold return: {cum_bh:+.2%}")

        print(f"\n  Quintile sort (next-day returns by sentiment quintile):")
        print(f"    {'Q':<4} {'Mean(daily)':>14} {'Annualized':>12} {'Count':>8}")
        print(f"    {'-'*42}")
        for q in [1, 2, 3, 4, 5]:
            row = quintile_rets.loc[q]
            print(f"    Q{q:<3} {row['mean']:>+14.6f} {row['annualized_mean']:>+12.4f} {int(row['count']):>8}")

        print(f"\n  Long-short spread (Q5 - Q1):")
        print(f"    Daily:     {ls_spread:+.6f}")
        print(f"    Annual:    {ls_spread_annual:+.4f}")

        if sharpe > 0.5:
            print(f"\n  -> Sharpe {sharpe:.2f}: Economically significant signal")
        elif sharpe > 0:
            print(f"\n  -> Sharpe {sharpe:.2f}: Weak but positive signal")
        else:
            print(f"\n  -> Sharpe {sharpe:.2f}: Signal not tradeable")

    return results


# ===================================================================
# TEST 5: Information Coefficient (IC)
# ===================================================================

def test_information_coefficient(df, verbose=True):
    """Compute Information Coefficient: rank correlation of SENT_t vs return_{t+1}.

    IC is the standard metric in quantitative finance for evaluating signals.
    - IC > 0.02 is considered decent for a single signal
    - IC_IR (IC mean / IC std) > 0.5 is considered good
    """
    if verbose:
        print("\n" + "=" * 70)
        print("TEST 5: INFORMATION COEFFICIENT (IC)")
        print("=" * 70)

    subset = df.dropna(subset=['SENT', 'FWD_RET']).copy()

    # Overall IC
    ic_overall, ic_p = spearmanr(subset['SENT'], subset['FWD_RET'])

    # Monthly IC series
    subset['YearMonth'] = subset['Date'].dt.to_period('M')
    monthly_ics = []
    for _, group in subset.groupby('YearMonth'):
        if len(group) < 5:
            continue
        ic, _ = spearmanr(group['SENT'], group['FWD_RET'])
        monthly_ics.append(ic)

    monthly_ics = np.array(monthly_ics)
    ic_mean = monthly_ics.mean()
    ic_std = monthly_ics.std()
    ic_ir = ic_mean / ic_std if ic_std > 0 else 0
    ic_hit_rate = (monthly_ics > 0).mean()  # % of months with positive IC

    # t-test for IC mean ≠ 0
    ic_tstat = ic_mean / (ic_std / np.sqrt(len(monthly_ics))) if ic_std > 0 else 0

    results = {
        'ic_overall': ic_overall,
        'ic_overall_p': ic_p,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'ic_ir': ic_ir,
        'ic_tstat': ic_tstat,
        'ic_hit_rate': ic_hit_rate,
        'n_months': len(monthly_ics),
    }

    if verbose:
        print(f"\n  Overall IC (full period): {ic_overall:+.4f} (p={ic_p:.4e})")
        print(f"\n  Monthly IC series ({len(monthly_ics)} months):")
        print(f"    IC mean:      {ic_mean:+.4f}")
        print(f"    IC std:       {ic_std:.4f}")
        print(f"    IC IR:        {ic_ir:+.4f}  (IC mean / IC std)")
        print(f"    IC t-stat:    {ic_tstat:+.4f}")
        print(f"    IC hit rate:  {ic_hit_rate:.1%}  (% months with IC > 0)")

        print(f"\n  Benchmarks:")
        print(f"    IC > 0.02 is decent for a single signal")
        print(f"    IC_IR > 0.5 is considered strong")
        if abs(ic_mean) > 0.02:
            print(f"  -> IC mean {ic_mean:+.4f}: Meaningful signal strength")
        else:
            print(f"  -> IC mean {ic_mean:+.4f}: Weak signal")

    return results


# ===================================================================
# TEST 6: Quantile regression
# ===================================================================

def test_quantile_regression(df, quantiles=None, verbose=True):
    """Quantile regression: does sentiment matter more in extreme markets?

    Like the volatility smile — tests if the SENT coefficient varies across
    the return distribution (tails vs center).

    return_q = α_q + β₁SMB + β₂HML + β₃RMW + β₄CMA + β₅SENT + ε
    for q ∈ {0.10, 0.25, 0.50, 0.75, 0.90}
    """
    if quantiles is None:
        quantiles = [0.10, 0.25, 0.50, 0.75, 0.90]

    if verbose:
        print("\n" + "=" * 70)
        print("TEST 6: QUANTILE REGRESSION (Non-linear effects)")
        print("=" * 70)
        print(f"\n  Like a volatility smile: does sentiment's effect vary")
        print(f"  across the return distribution?")

    subset = df.dropna(subset=['SENT']).copy()
    y = subset[RETURN_COL]
    X = sm.add_constant(subset[FF5_COLS + ['SENT']])

    results = {}

    if verbose:
        print(f"\n  {'Quantile':<10} {'SENT coef':>12} {'t-stat':>10} {'p-value':>12} {'Sig':>6}")
        print(f"  {'-'*52}")

    # OLS for comparison
    ols_res = sm.OLS(y, X, missing='drop').fit()
    ols_coef = ols_res.params.get('SENT', np.nan)

    for q in quantiles:
        model = sm.QuantReg(y, X)
        res = model.fit(q=q)

        coef = res.params.get('SENT', np.nan)
        tstat = res.tvalues.get('SENT', np.nan)
        pval = res.pvalues.get('SENT', np.nan)
        sig = '***' if pval < 0.01 else '**' if pval < 0.05 else '*' if pval < 0.1 else ''

        results[q] = {'coef': coef, 'tstat': tstat, 'pval': pval}

        if verbose:
            print(f"  {q:<10.2f} {coef:>12.6f} {tstat:>10.4f} {pval:>12.4e} {sig:>6}")

    if verbose:
        print(f"  {'OLS mean':<10} {ols_coef:>12.6f}")

        # Check for "smile" pattern
        coef_10 = results[0.10]['coef']
        coef_50 = results[0.50]['coef']
        coef_90 = results[0.90]['coef']

        print(f"\n  Pattern analysis:")
        print(f"    Left tail (Q10):  {coef_10:+.6f}")
        print(f"    Center (Q50):     {coef_50:+.6f}")
        print(f"    Right tail (Q90): {coef_90:+.6f}")

        if abs(coef_10) > abs(coef_50) and abs(coef_90) > abs(coef_50):
            print(f"  -> 'Smile' pattern: sentiment matters more in extreme markets")
        elif coef_10 > coef_90:
            print(f"  -> Asymmetric: sentiment effect stronger in bearish markets")
        elif coef_90 > coef_10:
            print(f"  -> Asymmetric: sentiment effect stronger in bullish markets")
        else:
            print(f"  -> Relatively flat: sentiment effect is similar across quantiles")

    return results


# ===================================================================
# Master runner
# ===================================================================

def run_all_tests(df, window=252, max_lags=5):
    """Run all 6 advanced evaluation tests."""
    all_results = {}

    all_results['predictive_regression'] = test_predictive_regression(df)
    all_results['granger_causality'] = test_granger_causality(df, max_lags=max_lags)
    all_results['oos_r2'] = test_oos_r2(df, window=window)
    all_results['long_short'] = test_long_short_backtest(df)
    all_results['information_coefficient'] = test_information_coefficient(df)
    all_results['quantile_regression'] = test_quantile_regression(df)

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: ADVANCED EVALUATION")
    print("=" * 70)

    pred = all_results['predictive_regression']
    gc = all_results['granger_causality']
    oos = all_results['oos_r2']
    ls = all_results['long_short']
    ic = all_results['information_coefficient']
    qr = all_results['quantile_regression']

    print(f"\n  {'Test':<35} {'Key metric':>20} {'Verdict':>15}")
    print(f"  {'-'*70}")

    # 1. Predictive
    p_pred = pred.get('Predictive (lag-1)', {}).get('pval', 1)
    v = 'YES' if p_pred < 0.05 else 'Marginal' if p_pred < 0.1 else 'NO'
    print(f"  {'1. Predictive power':<35} {'p=' + f'{p_pred:.4f}':>20} {v:>15}")

    # 2. Granger
    best_gc_p = min(v['pval'] for v in gc.get('SENT -> Returns', {0: {'pval': 1}}).values())
    v = 'YES' if best_gc_p < 0.05 else 'Marginal' if best_gc_p < 0.1 else 'NO'
    print(f"  {'2. Granger causality (SENT->RET)':<35} {'p=' + f'{best_gc_p:.4f}':>20} {v:>15}")

    # 3. OOS R-sq
    oos_val = oos.get('oos_r2_incremental', 0)
    v = 'YES' if oos_val > 0 else 'NO'
    print(f"  {'3. OOS R-sq improvement':<35} {f'{oos_val:+.6f}':>20} {v:>15}")

    # 4. Long-short
    sharpe = ls.get('sharpe_ratio', 0)
    v = 'Strong' if sharpe > 0.5 else 'Weak' if sharpe > 0 else 'None'
    print(f"  {'4. Long-short Sharpe':<35} {f'{sharpe:+.4f}':>20} {v:>15}")

    # 5. IC
    ic_mean = ic.get('ic_mean', 0)
    v = 'Decent' if abs(ic_mean) > 0.02 else 'Weak'
    print(f"  {'5. IC mean':<35} {f'{ic_mean:+.4f}':>20} {v:>15}")

    # 6. Quantile
    coef_10 = qr.get(0.10, {}).get('coef', 0)
    coef_90 = qr.get(0.90, {}).get('coef', 0)
    if abs(coef_10) > abs(coef_90) * 1.5 or abs(coef_90) > abs(coef_10) * 1.5:
        v = 'Asymmetric'
    else:
        v = 'Flat'
    print(f"  {'6. Quantile asymmetry':<35} {'Q10/Q90 ratio':>20} {v:>15}")

    return all_results


# ===================================================================
# CLI
# ===================================================================

def main():
    parser = argparse.ArgumentParser(description="Advanced sentiment evaluation.")
    parser.add_argument("--window", type=int, default=252,
                        help="Rolling window size for OOS R-sq (default: 252)")
    parser.add_argument("--lags", type=int, default=5,
                        help="Max lags for Granger causality (default: 5)")
    parser.add_argument("--total-table", type=str, default=None,
                        help="Path to total_table.csv")
    parser.add_argument("--sentiment-csv", type=str, default=None,
                        help="Path to daily_sentiment.csv")
    args = parser.parse_args()

    print("Loading data...")
    df = load_merged_data(
        total_path=args.total_table,
        sent_path=args.sentiment_csv,
    )
    print(f"  Merged data: {len(df)} trading days")
    print(f"  Date range: {df['Date'].min().strftime('%Y-%m-%d')} to "
          f"{df['Date'].max().strftime('%Y-%m-%d')}")
    print(f"  SENT coverage: {df['SENT'].notna().mean():.1%}")

    run_all_tests(df, window=args.window, max_lags=args.lags)


if __name__ == "__main__":
    main()

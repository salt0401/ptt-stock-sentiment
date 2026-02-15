"""FF5 + Sentiment OLS regression comparison.

Uses the existing total_table.csv (which has FF5 factors + old sentiment scores
aligned by trading date) as the baseline template.

Compares two OLS models per dictionary:
  Baseline:  excess_return ~ SMB + HML + RMW + CMA
  Augmented: excess_return ~ SMB + HML + RMW + CMA + SENT

where excess_return = return - RiskFreeRate  (= 'spread' column in total_table.csv).

Reports: R-squared, adj-R-squared, AIC/BIC, SENT coefficient/t-stat/p-value.
Uses HAC (Newey-West) standard errors for time-series robustness.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import DATA_DIR, OUTPUT_DIR


TOTAL_TABLE_PATH = DATA_DIR / "financial" / "aggregated_table" / "total_table.csv"

# FF5 factor columns (without spread — spread IS the dependent variable)
FF5_COLS = ['SMB', 'HML', 'RMW', 'CMA']
RETURN_COL = 'spread'  # excess return = return - RiskFreeRate


# ---------------------------------------------------------------------------
# Sentiment aggregation
# ---------------------------------------------------------------------------

def aggregate_daily_sentiment(push_data, sentiment_dict, date_col='ipdatetime'):
    """Aggregate comment-level sentiment to daily mean.

    Parameters
    ----------
    push_data : list[dict]
        Push data with 'content', 'ipdatetime', 'tag' keys.
    sentiment_dict : dict
        {word: sentiment_score}.
    date_col : str
        Key in push_data for datetime string.

    Returns
    -------
    pd.DataFrame : Columns ['Date', 'sentiment_mean', 'sentiment_count'].
    """
    from text_processing import segment_with_jieba, load_stopwords, filter_tokens
    import re

    stopwords = load_stopwords()
    records = []

    for item in push_data:
        dt_str = item.get(date_col, '')
        content = item.get('content', '')
        if not dt_str or not content:
            continue

        # Parse date from ipdatetime (format: "MM/DD HH:MM" or "IP MM/DD HH:MM")
        date_match = re.search(r'(\d{1,2})/(\d{1,2})', dt_str)
        if not date_match:
            continue

        words = list(segment_with_jieba([content])[0])
        words = filter_tokens(words, stopwords)
        scores = [sentiment_dict[w] for w in words if w in sentiment_dict]
        if not scores:
            continue

        records.append({
            'month': int(date_match.group(1)),
            'day': int(date_match.group(2)),
            'score': np.mean(scores),
        })

    if not records:
        return pd.DataFrame(columns=['Date', 'sentiment_mean', 'sentiment_count'])

    df = pd.DataFrame(records)

    # Group by month/day and take mean
    daily = df.groupby(['month', 'day']).agg(
        sentiment_mean=('score', 'mean'),
        sentiment_count=('score', 'count'),
    ).reset_index()

    return daily


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

def _nw_lag(T):
    """Compute Newey-West lag using Andrews' rule: int(4*(T/100)^(2/9))."""
    return int(4 * (T / 100) ** (2 / 9))


def run_ols(y, X, cov_type='HAC'):
    """Run OLS regression with HAC standard errors.

    Parameters
    ----------
    y : array-like
        Dependent variable.
    X : pd.DataFrame
        Independent variables (constant will be added).
    cov_type : str
        'HAC' for Newey-West, 'HC1' for White robust, 'nonrobust' for OLS.

    Returns
    -------
    statsmodels RegressionResultsWrapper
    """
    X_const = sm.add_constant(X)
    model = sm.OLS(y, X_const, missing='drop')

    T = len(y)
    if cov_type == 'HAC':
        lag = _nw_lag(T)
        results = model.fit(cov_type='HAC', cov_kwds={'maxlags': lag})
    elif cov_type in ('HC0', 'HC1', 'HC2', 'HC3'):
        results = model.fit(cov_type=cov_type)
    else:
        results = model.fit()

    return results


def run_regression_comparison(total_table_df, sentiment_series,
                              sentiment_name='SENT', cov_type='HAC',
                              verbose=True):
    """Run baseline vs augmented FF5 regression.

    Parameters
    ----------
    total_table_df : pd.DataFrame
        Must contain columns: 'return', 'spread', 'SMB', 'HML', 'RMW', 'CMA'.
    sentiment_series : pd.Series
        Sentiment values aligned with total_table_df index.
    sentiment_name : str
        Label for the sentiment variable.
    cov_type : str
        Standard error type.
    verbose : bool
        Print results.

    Returns
    -------
    dict : Regression comparison results.
    """
    # Drop rows with missing data
    df = total_table_df[FF5_COLS + [RETURN_COL]].copy()
    df[sentiment_name] = sentiment_series.values
    df = df.dropna()

    y = df[RETURN_COL]

    # Baseline: FF5 only
    X_base = df[FF5_COLS]
    res_base = run_ols(y, X_base, cov_type=cov_type)

    # Augmented: FF5 + sentiment
    X_aug = df[FF5_COLS + [sentiment_name]]
    res_aug = run_ols(y, X_aug, cov_type=cov_type)

    results = {
        'n_obs': int(len(df)),
        # Baseline
        'base_r2': res_base.rsquared,
        'base_adj_r2': res_base.rsquared_adj,
        'base_aic': res_base.aic,
        'base_bic': res_base.bic,
        # Augmented
        'aug_r2': res_aug.rsquared,
        'aug_adj_r2': res_aug.rsquared_adj,
        'aug_aic': res_aug.aic,
        'aug_bic': res_aug.bic,
        # Sentiment coefficient
        'sent_coef': res_aug.params.get(sentiment_name, np.nan),
        'sent_tstat': res_aug.tvalues.get(sentiment_name, np.nan),
        'sent_pvalue': res_aug.pvalues.get(sentiment_name, np.nan),
        # Delta
        'delta_r2': res_aug.rsquared - res_base.rsquared,
        'delta_adj_r2': res_aug.rsquared_adj - res_base.rsquared_adj,
        'delta_aic': res_aug.aic - res_base.aic,
        'delta_bic': res_aug.bic - res_base.bic,
    }

    if verbose:
        print(f"\n  {'Metric':<20} {'Baseline':>12} {'+ {}'.format(sentiment_name):>12} {'Delta':>12}")
        print(f"  {'-'*56}")
        print(f"  {'R-squared':<20} {results['base_r2']:>12.6f} {results['aug_r2']:>12.6f} {results['delta_r2']:>+12.6f}")
        print(f"  {'Adj R-squared':<20} {results['base_adj_r2']:>12.6f} {results['aug_adj_r2']:>12.6f} {results['delta_adj_r2']:>+12.6f}")
        print(f"  {'AIC':<20} {results['base_aic']:>12.2f} {results['aug_aic']:>12.2f} {results['delta_aic']:>+12.2f}")
        print(f"  {'BIC':<20} {results['base_bic']:>12.2f} {results['aug_bic']:>12.2f} {results['delta_bic']:>+12.2f}")
        print(f"\n  {sentiment_name} coefficient: {results['sent_coef']:.6f}")
        print(f"  {sentiment_name} t-stat:      {results['sent_tstat']:.4f}")
        print(f"  {sentiment_name} p-value:     {results['sent_pvalue']:.4e}")
        sig = '***' if results['sent_pvalue'] < 0.01 else '**' if results['sent_pvalue'] < 0.05 else '*' if results['sent_pvalue'] < 0.1 else ''
        print(f"  Significance:      {sig}")
        print(f"\n  N observations: {results['n_obs']}")
        nw_lag = _nw_lag(results['n_obs'])
        print(f"  HAC lags (Newey-West): {nw_lag}")

    return results


# ---------------------------------------------------------------------------
# Load total_table and run with existing sentiment columns
# ---------------------------------------------------------------------------

def load_total_table(path=None):
    """Load the existing total_table.csv.

    Parameters
    ----------
    path : str or Path, optional

    Returns
    -------
    pd.DataFrame
    """
    if path is None:
        path = TOTAL_TABLE_PATH
    df = pd.read_csv(path)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df


def run_existing_sentiment_regressions(total_df, verbose=True):
    """Run FF5 regressions for all existing sentiment columns in total_table.

    Parameters
    ----------
    total_df : pd.DataFrame
        total_table.csv loaded as DataFrame.
    verbose : bool

    Returns
    -------
    dict : {column_name: regression_results}
    """
    sentiment_cols = [c for c in total_df.columns
                      if '分數' in c and '_norm' in c]

    all_results = {}
    for col in sentiment_cols:
        if verbose:
            print(f"\n{'='*60}")
            print(f"Regression: {col}")
            print(f"{'='*60}")
        res = run_regression_comparison(
            total_df, total_df[col], sentiment_name=col, verbose=verbose
        )
        all_results[col] = res

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    import argparse

    parser = argparse.ArgumentParser(description="FF5 + Sentiment regression comparison.")
    parser.add_argument("--total-table", type=str, default=None,
                        help="Path to total_table.csv")
    parser.add_argument("--sentiment-csv", type=str, default=None,
                        help="Path to new sentiment CSV to add as column")
    parser.add_argument("--sentiment-name", type=str, default="SENT",
                        help="Name for the new sentiment column")
    args = parser.parse_args()

    print("Loading total_table.csv...")
    total_df = load_total_table(args.total_table)
    print(f"  Shape: {total_df.shape}")

    if args.sentiment_csv:
        print(f"\nLoading new sentiment from {args.sentiment_csv}...")
        new_sent = pd.read_csv(args.sentiment_csv)
        # Expect columns: Date, sentiment_mean
        if 'Date' in new_sent.columns:
            new_sent['Date'] = pd.to_datetime(new_sent['Date'])
            total_df = total_df.merge(new_sent[['Date', 'sentiment_mean']],
                                      on='Date', how='left')
            total_df.rename(columns={'sentiment_mean': args.sentiment_name},
                            inplace=True)

        print(f"\nRunning regression with {args.sentiment_name}...")
        run_regression_comparison(
            total_df, total_df[args.sentiment_name],
            sentiment_name=args.sentiment_name,
        )
    else:
        print("\nRunning regressions for existing sentiment columns...")
        run_existing_sentiment_regressions(total_df)


if __name__ == "__main__":
    main()

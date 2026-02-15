"""Fama-French 5-Factor Model — extracted from five_factor_notebook.ipynb.

Computes the five factors: R_market, SMB, HML, RMW, CMA from
Taiwanese stock market data.
"""

import pandas as pd
import numpy as np

from config import FINANCIAL_DIR
from column_map import STOCK_COLUMNS, ROE_COLUMNS, ASSET_COLUMNS


# ---------------------------------------------------------------------------
# Data loading (from cells 2-4)
# ---------------------------------------------------------------------------

def load_stock_price_data(data_dir=None, year_start=2018, year_end=2023):
    """Load monthly stock price CSV files and clean numeric columns.

    Parameters
    ----------
    data_dir : str or Path, optional
        Directory containing yearly CSV files like '2018-1.csv'.
        Defaults to config.FINANCIAL_DIR.
    year_start : int
        First year to load.
    year_end : int
        Last year to load (inclusive).

    Returns
    -------
    pd.DataFrame : Combined stock price data with cleaned numeric columns.
    """
    if data_dir is None:
        data_dir = FINANCIAL_DIR

    df = pd.DataFrame()
    for year in range(year_start, year_end + 1):
        for month in range(1, 13):
            file_path = data_dir / f"{year}-{month}.csv"
            try:
                data = pd.read_csv(file_path, thousands=",")
                df = pd.concat([df, data], ignore_index=True)
            except FileNotFoundError:
                print(f"File {file_path} not found")

    # Rename Chinese columns to English
    df.rename(columns=STOCK_COLUMNS, inplace=True)

    # Drop rows with missing essential columns
    for col in ['close_price', 'shares_outstanding_k', 'pb_ratio', 'daily_return_pct']:
        df.dropna(subset=[col], inplace=True)

    # Convert to numeric
    df['close_price'] = pd.to_numeric(df['close_price'], errors='coerce')
    df['shares_outstanding_k'] = pd.to_numeric(df['shares_outstanding_k'], errors='coerce')
    df['pb_ratio'] = pd.to_numeric(df['pb_ratio'], errors='coerce')
    df['daily_return_pct'] = pd.to_numeric(df['daily_return_pct'], errors='coerce')

    # Parse dates and sort
    df['date'] = pd.to_datetime(df['date'], format='%Y/%m/%d')
    df.sort_values(['stock_id', 'date'], inplace=True)

    return df


def load_roe_data(path):
    """Load ROE.csv and compute ROE = (gross_profit - expenses) / (revenue - expenses).

    Parameters
    ----------
    path : str or Path
        Path to ROE.csv.

    Returns
    -------
    pd.DataFrame : ROE data with columns ['stock_id', 'year', 'month', 'ROE', ...].
    """
    data_roe = pd.read_csv(path, thousands=",")
    data_roe.rename(columns=ROE_COLUMNS, inplace=True)

    data_roe['date'] = pd.to_datetime(data_roe['date_str'], format='%Y/%m/%d', errors='coerce')
    data_roe.sort_values(['stock_id', 'date'], inplace=True)

    data_roe['year'] = data_roe['date'].dt.year
    data_roe['month'] = data_roe['date'].dt.month

    # ROE = (gross_profit - operating_expenses) / (net_revenue - operating_expenses)
    data_roe['ROE'] = (
        (data_roe['gross_profit'] - data_roe['operating_expenses'])
        / (data_roe['net_revenue'] - data_roe['operating_expenses'])
    )

    return data_roe


def load_asset_data(path):
    """Load total_assets.csv for asset growth rate calculation.

    Parameters
    ----------
    path : str or Path
        Path to total_assets.csv.

    Returns
    -------
    pd.DataFrame : Asset data with columns ['stock_id', 'date_str', 'month', 'total_assets', 'year'].
    """
    data_asset = pd.read_csv(path, thousands=",")
    data_asset.rename(columns=ASSET_COLUMNS, inplace=True)
    data_asset['date_str'] = pd.to_datetime(data_asset['date_str'], format='%Y/%m/%d', errors='coerce')
    data_asset['year'] = data_asset['date_str'].dt.year
    data_asset.sort_values(['stock_id', 'date_str'], inplace=True)
    return data_asset


# ---------------------------------------------------------------------------
# Grouping functions (from cells 5-7, 12-22)
# ---------------------------------------------------------------------------

def _quarter_group(month):
    """Map month to quarterly group: 3, 6, 9, or 12."""
    if month <= 3:
        return 3
    elif month <= 6:
        return 6
    elif month <= 9:
        return 9
    else:
        return 12


def add_market_cap(df):
    """Compute market cap = close_price * shares_outstanding_k.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with price and shares columns.

    Returns
    -------
    pd.DataFrame : Same DataFrame with 'market_cap' column added.
    """
    df['market_cap'] = df['close_price'] * df['shares_outstanding_k']
    return df


def add_size_groups(df):
    """Split market cap into S/B groups (50th percentile) per date.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'date' and 'market_cap' columns.

    Returns
    -------
    pd.DataFrame : With 'size_group' column added ('S' or 'B').
    """
    df['size_group'] = df.groupby('date', group_keys=False)['market_cap'].apply(
        lambda x: pd.qcut(x, q=[0, 0.5, 1], labels=['S', 'B'])
    )
    return df


def add_pb_groups(df):
    """Split PB ratio into L/M/H groups (33rd/67th percentile) per date.

    Parameters
    ----------
    df : pd.DataFrame
        Must have 'date', 'pb_ratio', 'size_group' columns.

    Returns
    -------
    pd.DataFrame : With 'pb_group' and 'group' columns added.
    """
    df.dropna(subset=['market_cap'], inplace=True)
    df['pb_group'] = df.groupby('date', group_keys=False)['pb_ratio'].apply(
        lambda x: pd.qcut(x, q=[0, 1/3, 2/3, 1], labels=['L', 'M', 'H'])
    )
    df['group'] = df['size_group'].astype(str) + df['pb_group'].astype(str)
    return df


def add_roe_groups(df, roe_data):
    """Merge ROE data and split into W/F/R groups (33rd/67th percentile).

    Parameters
    ----------
    df : pd.DataFrame
        Stock data with 'year', 'month' columns.
    roe_data : pd.DataFrame
        ROE data from load_roe_data().

    Returns
    -------
    pd.DataFrame : With ROE, roe_group, size_roe_group columns added.
    """
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['quarter_group'] = df['month'].apply(_quarter_group)

    merged = pd.merge(
        df,
        roe_data[['stock_id', 'year', 'month', 'ROE']],
        left_on=['stock_id', 'year', 'quarter_group'],
        right_on=['stock_id', 'year', 'month'],
        suffixes=['', '_roe'],
        how='left',
    )

    merged['ROE'] = pd.to_numeric(merged['ROE'], errors='coerce')
    merged.dropna(subset=['ROE'], inplace=True)
    merged = merged.reset_index(drop=True)

    merged['roe_group'] = merged.groupby('date', group_keys=False)['ROE'].apply(
        lambda x: pd.qcut(x, q=[0, 1/3, 2/3, 1], labels=['W', 'F', 'R'])
    )
    merged['size_roe_group'] = merged['size_group'].astype(str) + merged['roe_group'].astype(str)

    return merged


def add_asset_growth_groups(df, asset_data):
    """Merge asset data, compute growth rate, and split into C/I/A groups.

    Parameters
    ----------
    df : pd.DataFrame
        Stock data (after ROE merge) with 'quarter_group' column.
    asset_data : pd.DataFrame
        Asset data from load_asset_data().

    Returns
    -------
    pd.DataFrame : With 'asset_growth_rate', 'asset_growth_group', 'size_asset_growth_group' columns.
    """
    # Merge current period assets
    merged_asset = pd.merge(
        df[['stock_id', 'date', 'year', 'month', 'quarter_group']],
        asset_data[['stock_id', 'year', 'month', 'total_assets']],
        left_on=['stock_id', 'year', 'quarter_group'],
        right_on=['stock_id', 'year', 'month'],
        suffixes=["", "_asset"],
        how='left',
    )
    merged_asset['total_assets'] = pd.to_numeric(merged_asset['total_assets'], errors='coerce')
    merged_asset.dropna(subset=['total_assets'], inplace=True)

    # Compute previous period
    merged_asset['date_ym'] = pd.to_datetime(
        merged_asset['year'].astype(str) + '-' + merged_asset['month'].astype(str)
    )
    merged_asset['prev_date_ym'] = merged_asset['date_ym'] - pd.DateOffset(months=1)
    merged_asset['prev_year'] = merged_asset['prev_date_ym'].dt.year
    merged_asset['prev_month'] = merged_asset['prev_date_ym'].dt.month
    merged_asset['prev_quarter_group'] = merged_asset['prev_month'].apply(_quarter_group)

    # Merge previous period assets
    merged_asset = pd.merge(
        merged_asset,
        asset_data[['stock_id', 'year', 'month', 'total_assets']],
        left_on=['stock_id', 'prev_year', 'prev_quarter_group'],
        right_on=['stock_id', 'year', 'month'],
        suffixes=["", "_prev"],
        how='left',
    )

    # Compute growth rate
    merged_asset['asset_growth_rate'] = (
        (merged_asset['total_assets'] - merged_asset['total_assets_prev'])
        / merged_asset['total_assets_prev']
    )

    # Assign growth rate back to main df
    df['asset_growth_rate'] = merged_asset['asset_growth_rate'].values

    # Group by asset growth rate
    df['asset_growth_group'] = df.groupby('date', group_keys=False)['asset_growth_rate'].apply(
        lambda x: pd.qcut(x.rank(method="first"), q=[0, 1/3, 2/3, 1], labels=['C', 'I', 'A'])
    )
    df['size_asset_growth_group'] = df['size_group'].astype(str) + df['asset_growth_group'].astype(str)

    return df


# ---------------------------------------------------------------------------
# Factor calculation (from cells 24-27)
# ---------------------------------------------------------------------------

def calculate_weighted_return(group_df):
    """Calculate value-weighted return for a group.

    Parameters
    ----------
    group_df : pd.DataFrame
        Subset of data for one group, must have 'daily_return_pct' and 'market_cap'.

    Returns
    -------
    float : Value-weighted return.
    """
    total_market_cap = group_df['market_cap'].sum()
    if total_market_cap == 0:
        return 0.0
    return (group_df['daily_return_pct'] * group_df['market_cap']).sum() / total_market_cap


def calculate_factors_for_date(df, date):
    """Compute R_market, SMB, HML, RMW, CMA for one date.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all grouping columns.
    date : datetime-like
        Target date.

    Returns
    -------
    dict : {'date', 'R_market', 'SMB', 'HML', 'RMW', 'CMA'}
    """
    target = df[df['date'] == date]

    if target.empty:
        return None

    R_market = calculate_weighted_return(target)

    # --- SMB and HML (Size x PB groups) ---
    grouped_pb = target.groupby('group')
    group_returns_pb = {}
    for label in ['SL', 'SM', 'SH', 'BL', 'BM', 'BH']:
        if label in grouped_pb.groups:
            group_returns_pb[label] = calculate_weighted_return(grouped_pb.get_group(label))
        else:
            group_returns_pb[label] = 0.0

    SL = group_returns_pb['SL']
    SM = group_returns_pb['SM']
    SH = group_returns_pb['SH']
    BL = group_returns_pb['BL']
    BM = group_returns_pb['BM']
    BH = group_returns_pb['BH']

    SMB = (SL + SM + SH) / 3 - (BL + BM + BH) / 3
    HML = (BH + SH) / 2 - (BL + SL) / 2

    # --- RMW (Size x ROE groups) ---
    grouped_roe = target.groupby('size_roe_group')
    group_returns_roe = {}
    for label in ['SR', 'BR', 'SW', 'BW']:
        if label in grouped_roe.groups:
            group_returns_roe[label] = calculate_weighted_return(grouped_roe.get_group(label))
        else:
            group_returns_roe[label] = 0.0

    SR = group_returns_roe['SR']
    BR = group_returns_roe['BR']
    SW = group_returns_roe['SW']
    BW = group_returns_roe['BW']

    RMW = (SR + BR) / 2 - (SW + BW) / 2

    # --- CMA (Size x Asset Growth groups) ---
    grouped_cma = target.groupby('size_asset_growth_group')
    group_returns_cma = {}
    for label in ['SC', 'BC', 'SA', 'BA']:
        if label in grouped_cma.groups:
            group_returns_cma[label] = calculate_weighted_return(grouped_cma.get_group(label))
        else:
            group_returns_cma[label] = 0.0

    SC = group_returns_cma['SC']
    BC = group_returns_cma['BC']
    SA = group_returns_cma['SA']
    BA = group_returns_cma['BA']

    CMA = (SC + BC) / 2 - (SA + BA) / 2

    return {
        'date': date,
        'R_market': R_market,
        'SMB': SMB,
        'HML': HML,
        'RMW': RMW,
        'CMA': CMA,
    }


def calculate_all_factors(df):
    """Loop over all unique dates and compute the 5 factors.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset with all grouping columns.

    Returns
    -------
    pd.DataFrame : Factor table with columns ['date', 'R_market', 'SMB', 'HML', 'RMW', 'CMA'].
    """
    dates = sorted(df['date'].unique())
    results = []

    for i, date in enumerate(dates):
        result = calculate_factors_for_date(df, date)
        if result is not None:
            results.append(result)
        if (i + 1) % 100 == 0:
            print(f"Processed {i + 1}/{len(dates)} dates...")

    print(f"Done. Processed {len(dates)} dates total.")
    return pd.DataFrame(results)

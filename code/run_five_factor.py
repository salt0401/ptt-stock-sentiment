"""Entry point: Calculate Fama-French 5 factors for all dates.

Usage:
    python run_five_factor.py [--year-start 2018] [--year-end 2023]
                              [--stock-dir PATH] [--roe-path PATH] [--asset-path PATH]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import FINANCIAL_DIR, FAMA_FRENCH_OUTPUT_DIR
from five_factor import (
    load_stock_price_data,
    load_roe_data,
    load_asset_data,
    add_market_cap,
    add_size_groups,
    add_pb_groups,
    add_roe_groups,
    add_asset_growth_groups,
    calculate_all_factors,
)


def main():
    parser = argparse.ArgumentParser(description="Calculate Fama-French 5 factors.")
    parser.add_argument("--year-start", type=int, default=2018,
                        help="First year to load (default: 2018)")
    parser.add_argument("--year-end", type=int, default=2023,
                        help="Last year to load (default: 2023)")
    parser.add_argument("--stock-dir", type=str, default=None,
                        help="Directory with monthly stock price CSVs (default: config)")
    parser.add_argument("--roe-path", type=str, default=None,
                        help="Path to ROE.csv (default: FINANCIAL_DIR/ROE.csv)")
    parser.add_argument("--asset-path", type=str, default=None,
                        help="Path to total_assets.csv (default: FINANCIAL_DIR/total_assets.csv)")
    args = parser.parse_args()

    stock_dir = Path(args.stock_dir) if args.stock_dir else FINANCIAL_DIR
    roe_path = Path(args.roe_path) if args.roe_path else FINANCIAL_DIR / "ROE.csv"
    asset_path = Path(args.asset_path) if args.asset_path else FINANCIAL_DIR / "total_assets.csv"

    FAMA_FRENCH_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Step 1: Load stock price data ---
    print(f"Step 1: Loading stock price data ({args.year_start}-{args.year_end})...")
    df = load_stock_price_data(stock_dir, args.year_start, args.year_end)
    print(f"  Loaded {len(df)} rows")

    # --- Step 2: Compute market cap and size groups ---
    print("Step 2: Computing market cap and size groups...")
    df = add_market_cap(df)
    df = add_size_groups(df)

    # --- Step 3: Add PB groups ---
    print("Step 3: Adding PB ratio groups (L/M/H)...")
    df = add_pb_groups(df)

    # --- Step 4: Load and merge ROE data ---
    print(f"Step 4: Loading ROE data from {roe_path}...")
    roe_data = load_roe_data(roe_path)
    df = add_roe_groups(df, roe_data)
    print(f"  After ROE merge: {len(df)} rows")

    # --- Step 5: Load and merge asset growth data ---
    print(f"Step 5: Loading asset data from {asset_path}...")
    asset_data = load_asset_data(asset_path)
    df = add_asset_growth_groups(df, asset_data)
    print(f"  After asset merge: {len(df)} rows")

    # --- Step 6: Calculate 5 factors for each date ---
    print("Step 6: Calculating 5 factors for each trading date...")
    factor_df = calculate_all_factors(df)

    # --- Step 7: Save results ---
    output_path = FAMA_FRENCH_OUTPUT_DIR / "five_factors.csv"
    factor_df.to_csv(output_path, index=False)
    print(f"\nSaved 5-factor results to {output_path}")
    print(f"Total dates: {len(factor_df)}")
    print("\nSample output:")
    print(factor_df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

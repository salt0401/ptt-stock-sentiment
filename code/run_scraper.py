"""Entry point: Scrape PTT Stock board -> filter -> save raw data.

Usage:
    python run_scraper.py [--days 350] [--max-posts 1000]
"""

import argparse
import pickle
import time
import sys
from pathlib import Path

import pandas as pd

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import COMMENTS_RAW_DIR
from scraper import PTTScraper
from text_processing import filter_posts_by_title


def main():
    parser = argparse.ArgumentParser(description="Scrape PTT Stock board posts.")
    parser.add_argument("--days", type=int, default=350,
                        help="Number of days to look back (default: 350)")
    parser.add_argument("--max-posts", type=int, default=1000,
                        help="Maximum posts to scrape per category (default: 1000)")
    parser.add_argument("--board", type=str, default="Stock",
                        help="PTT board name (default: Stock)")
    args = parser.parse_args()

    # Ensure output directory exists
    COMMENTS_RAW_DIR.mkdir(parents=True, exist_ok=True)

    board = args.board
    days = args.days
    max_posts = args.max_posts

    # --- Scrape after-market posts ---
    print(f"Scraping after-market posts from PTT {board} board (last {days} days, max {max_posts})...")
    scraper_after = PTTScraper(board)
    start = time.time()
    data_after = scraper_after.get_title_and_before_days(
        "盤後", "[閒聊]", delta_days=days, max_posts=max_posts
    )
    elapsed = time.time() - start
    print(f"After-market: scraped {len(data_after)} posts in {elapsed:.1f}s")

    if data_after:
        df_after = pd.DataFrame(data_after)
        # Filter: remove rows containing wrong category
        df_after = filter_posts_by_title(df_after, exclude_words=['盤中閒聊'])
        print(f"After-market after filtering: {len(df_after)} posts")
    else:
        df_after = pd.DataFrame()
        print("After-market: no data scraped")

    # --- Scrape during-market posts ---
    print(f"\nScraping during-market posts from PTT {board} board (last {days} days, max {max_posts})...")
    scraper_during = PTTScraper(board)
    start = time.time()
    data_during = scraper_during.get_title_and_before_days(
        "盤中", "[閒聊]", delta_days=days, max_posts=max_posts
    )
    elapsed = time.time() - start
    print(f"During-market: scraped {len(data_during)} posts in {elapsed:.1f}s")

    if data_during:
        df_during = pd.DataFrame(data_during)
        # Filter: remove rows containing wrong category
        df_during = filter_posts_by_title(df_during, exclude_words=['盤後閒聊'])
        print(f"During-market after filtering: {len(df_during)} posts")
    else:
        df_during = pd.DataFrame()
        print("During-market: no data scraped")

    # --- Save results ---
    after_path = COMMENTS_RAW_DIR / "after_market.pkl"
    during_path = COMMENTS_RAW_DIR / "during_market.pkl"

    with open(after_path, 'wb') as f:
        pickle.dump(df_after.to_dict('records'), f)
    print(f"\nSaved after-market data to {after_path}")

    with open(during_path, 'wb') as f:
        pickle.dump(df_during.to_dict('records'), f)
    print(f"Saved during-market data to {during_path}")

    # Also save as CSV for easy inspection
    if not df_after.empty:
        csv_path = COMMENTS_RAW_DIR / "after_market.csv"
        df_after.drop(columns=['Pushes'], errors='ignore').to_csv(csv_path, index=False)
        print(f"Saved after-market CSV to {csv_path}")

    if not df_during.empty:
        csv_path = COMMENTS_RAW_DIR / "during_market.csv"
        df_during.drop(columns=['Pushes'], errors='ignore').to_csv(csv_path, index=False)
        print(f"Saved during-market CSV to {csv_path}")


if __name__ == "__main__":
    main()

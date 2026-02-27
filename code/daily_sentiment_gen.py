"""Generate daily_sentiment.csv from any sentiment dictionary.

Scores every push comment using the given dictionary, then aggregates
to daily mean sentiment using the post-level Date column (which has the
full date, unlike ipdatetime which lacks the year).

Supports pre-segmented cache: segment once, then score with multiple
dictionaries without re-running jieba (saves ~30 min per method).

Output format matches existing daily_sentiment.csv:
    Date,sentiment_mean,sentiment_count
    2015-07-17,0.012171,1243.0

Usage:
    python daily_sentiment_gen.py --dict-csv output/total_dictionary.csv
    python daily_sentiment_gen.py --dict-csv output/fasttext_total_dictionary.csv \
           --output output/fasttext_daily_sentiment.csv
"""

import argparse
import json
import pickle  # nosec - reading trusted local PTT data files only
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import PICKLE_AFTER_PATH, PICKLE_DURING_PATH, OUTPUT_DIR
from text_processing import (
    load_stopwords, extract_push_contents_with_tags,
    segment_with_jieba, filter_tokens,
)

# Cache file for pre-segmented comments with dates
SEGMENTED_DATED_CACHE = OUTPUT_DIR / "segmented_dated_cache.json"


def _score_words(words, sentiment_dict, stopwords):
    """Score a list of tokens using a sentiment dictionary."""
    words = filter_tokens(words, stopwords)
    scores = [sentiment_dict[w] for w in words if w in sentiment_dict]
    if not scores:
        return None
    return np.mean(scores)


def build_segmented_cache(after_pkl=None, during_pkl=None,
                          cache_path=None, verbose=True):
    """Segment all comments once and cache with dates.

    This is the expensive step (~30 min for 3.6M comments). The cache
    is reused across all dictionary scoring runs.

    Parameters
    ----------
    after_pkl, during_pkl : Path, optional
    cache_path : Path, optional
    verbose : bool

    Returns
    -------
    tuple : (dates_list, segmented_list)
        dates_list: list[str] — ISO date strings
        segmented_list: list[list[str]] — tokenized comments
    """
    if cache_path is None:
        cache_path = SEGMENTED_DATED_CACHE
    cache_path = Path(cache_path)

    if after_pkl is None:
        after_pkl = PICKLE_AFTER_PATH
    if during_pkl is None:
        during_pkl = PICKLE_DURING_PATH

    # Try loading from cache
    if cache_path.exists():
        if verbose:
            print(f"  Loading segmented cache from {cache_path}...")
        with open(cache_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        dates = data['dates']
        segmented = data['tokens']
        if verbose:
            print(f"  Cache hit: {len(dates)} comments")
        return dates, segmented

    if verbose:
        print("  Building segmented cache (this runs once, ~30 min)...")

    # Collect all comments with their post dates
    comment_dates = []
    comment_texts = []

    for label, pkl_path in [('after', after_pkl), ('during', during_pkl)]:
        with open(pkl_path, 'rb') as f:
            raw_data = pickle.load(f)  # nosec - trusted local PTT data

        if isinstance(raw_data, pd.DataFrame):
            df = raw_data
        else:
            df = pd.DataFrame(raw_data)

        if verbose:
            print(f"  Processing {label}: {len(df)} posts")

        for _, row in df.iterrows():
            post_date = row.get('Date', None)
            pushes = row.get('Pushes', [])

            if post_date is None or not isinstance(pushes, list):
                continue

            # Normalize date to string
            date_str = str(pd.Timestamp(post_date).strftime('%Y-%m-%d'))

            tagged = extract_push_contents_with_tags(pushes)
            for item in tagged:
                comment_dates.append(date_str)
                comment_texts.append(item['content'])

    if verbose:
        print(f"  Total comments to segment: {len(comment_texts)}")

    # Batch segmentation
    segmented = segment_with_jieba(comment_texts)

    # Save cache
    cache_data = {
        'dates': comment_dates,
        'tokens': segmented,
    }
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(cache_data, f, ensure_ascii=False)
    if verbose:
        print(f"  Cached to {cache_path}")

    return comment_dates, segmented


def generate_daily_sentiment(dict_df=None, after_pkl=None, during_pkl=None,
                             output_path=None, verbose=True,
                             precomputed_cache=None, use_finbert=False):
    """Score push comments and aggregate to daily sentiment.

    Uses the post-level Date column (full date) rather than the
    comment-level ipdatetime (which lacks year).

    Parameters
    ----------
    dict_df : pd.DataFrame, optional
        Sentiment dictionary with columns ['word', 'sentiment_score'].
        Required unless use_finbert is True.
    after_pkl, during_pkl : Path, optional
    output_path : Path, optional
    verbose : bool
    precomputed_cache : tuple, optional
        (dates_list, segmented_list) from build_segmented_cache().
        If provided, skips segmentation entirely.
    use_finbert : bool
        If True, ignores dict_df and uses FinBERT to score raw, unsegmented strings.

    Returns
    -------
    pd.DataFrame : Columns ['Date', 'sentiment_mean', 'sentiment_count'].
    """
    if not use_finbert:
        if dict_df is None:
            raise ValueError("dict_df is required when use_finbert=False")
        sentiment_dict = dict_df.set_index('word')['sentiment_score'].to_dict()
        stopwords = load_stopwords()

    # Get pre-segmented data
    if precomputed_cache is not None:
        dates, segmented = precomputed_cache
    else:
        dates, segmented = build_segmented_cache(
            after_pkl=after_pkl, during_pkl=during_pkl, verbose=verbose,
        )

    if verbose:
        print(f"  Scoring {len(segmented)} comments...")

    # Load weights
    weights_path = OUTPUT_DIR / "sentiment_weights.npz"
    weight_author = None
    weight_engagement = None
    if weights_path.exists():
        ws = np.load(weights_path)
        if len(ws['author']) == len(segmented):
            if verbose:
                print("  Loaded Author and Engagement weights.")
            weight_author = ws['author']
            weight_engagement = ws['engagement']
        else:
            print("  [WARN] Weight mismatch, ignoring.")

    all_records = []
    
    if use_finbert:
        # Instead of word-level dictionary scoring, we score full strings contextually.
        # We need the original strings. Since 'segmented' might be token lists depending 
        # on the cache, we reconstruct the strings without spaces.
        print("  Initializing FinBERT model...")
        from finbert_scorer import FinBERTScorer
        scorer = FinBERTScorer()
        
        # Combine segmented tokens back to continuous strings for BERT inference
        raw_texts = ["".join(tokens) for tokens in segmented]
        
        scores = scorer.score_sentences(raw_texts, verbose=verbose)
        for i, score in enumerate(scores):
            wa = weight_author[i] if weight_author is not None else 1.0
            we = weight_engagement[i] if weight_engagement is not None else 1.0
            all_records.append({
                'Date': dates[i],
                'score': score,
                'weight_author': wa,
                'weight_engagement': we
            })
    else:
        # Score each comment (fast — just dictionary lookups)
        for i, words in enumerate(segmented):
            score = _score_words(list(words), sentiment_dict, stopwords)
            if score is not None:
                wa = weight_author[i] if weight_author is not None else 1.0
                we = weight_engagement[i] if weight_engagement is not None else 1.0
                all_records.append({
                    'Date': dates[i],
                    'score': score,
                    'weight_author': wa,
                    'weight_engagement': we
                })

    if not all_records:
        print("  [WARN] No scoreable comments found!")
        return pd.DataFrame(columns=['Date', 'sentiment_mean', 'sentiment_count', 'sentiment_mean_author', 'sentiment_mean_engagement'])

    records_df = pd.DataFrame(all_records)
    records_df['Date'] = pd.to_datetime(records_df['Date'])
    
    # Calculate weighted elements
    records_df['sc_auth'] = records_df['score'] * records_df['weight_author']
    records_df['sc_eng'] = records_df['score'] * records_df['weight_engagement']

    daily = records_df.groupby('Date').agg(
        sentiment_mean=('score', 'mean'),
        sentiment_count=('score', 'count'),
        sum_sc_auth=('sc_auth', 'sum'),
        sum_w_auth=('weight_author', 'sum'),
        sum_sc_eng=('sc_eng', 'sum'),
        sum_w_eng=('weight_engagement', 'sum'),
    ).reset_index()

    # Reconstruct weighted means globally for the day
    daily['sentiment_mean_author'] = daily['sum_sc_auth'] / daily['sum_w_auth']
    daily['sentiment_mean_engagement'] = daily['sum_sc_eng'] / daily['sum_w_eng']
    
    # Drop intermediate sums
    daily = daily.drop(columns=['sum_sc_auth', 'sum_w_auth', 'sum_sc_eng', 'sum_w_eng'])
    
    # Handle NaN resulting from division by zero
    daily = daily.fillna({'sentiment_mean_author': 0.0, 'sentiment_mean_engagement': 0.0})

    daily = daily.sort_values('Date').reset_index(drop=True)
    daily['Date'] = daily['Date'].dt.strftime('%Y-%m-%d')

    if verbose:
        print(f"  Total comments scored: {len(records_df)}")
        print(f"  Trading days with sentiment: {len(daily)}")
        print(f"  Date range: {daily['Date'].iloc[0]} to {daily['Date'].iloc[-1]}")
        print(f"  Mean sentiment: {daily['sentiment_mean'].mean():.6f}")

    if output_path is not None:
        daily.to_csv(output_path, index=False)
        if verbose:
            print(f"  Saved to {output_path}")

    return daily


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate daily_sentiment.csv from a sentiment dictionary."
    )
    parser.add_argument("--dict-csv", type=str, default=None,
                        help="Path to sentiment dictionary CSV (word, sentiment_score)")
    parser.add_argument("--after-pkl", type=str, default=None)
    parser.add_argument("--during-pkl", type=str, default=None)
    parser.add_argument("--output", type=str, default=None,
                        help="Output CSV path (default: output/daily_sentiment.csv)")
    parser.add_argument("--use-finbert", action="store_true",
                        help="Use FinBERT for contextual scoring instead of dictionary")
    args = parser.parse_args()

    dict_df = None
    if not args.use_finbert:
        if args.dict_csv is None:
            parser.error("--dict-csv is required unless --use-finbert is specified")
            
        dict_df = pd.read_csv(args.dict_csv)
        if 'word' not in dict_df.columns:
            from column_map import SENTIMENT_COLUMNS
            dict_df.rename(columns=SENTIMENT_COLUMNS, inplace=True)
        print(f"Loaded dictionary: {len(dict_df)} words")

    output_path = Path(args.output) if args.output else OUTPUT_DIR / "daily_sentiment.csv"

    after_pkl = Path(args.after_pkl) if args.after_pkl else None
    during_pkl = Path(args.during_pkl) if args.during_pkl else None

    generate_daily_sentiment(
        dict_df, after_pkl=after_pkl, during_pkl=during_pkl,
        output_path=output_path, use_finbert=args.use_finbert
    )


if __name__ == "__main__":
    main()

"""Master comparison script: old pipeline vs improved pipeline.

Usage:
    python run_comparison.py                    # Full run (trains both, evaluates)
    python run_comparison.py --skip-training    # Load existing dicts, evaluate only
    python run_comparison.py --eval-only        # Same as --skip-training
    python run_comparison.py --k 20 --vector-size 150

Steps:
1. Load raw data
2. Load/build old pipeline dictionary
3. Load/build new pipeline dictionary
4. Run push-tag evaluation for both
5. Run FF5 regression for both
6. Print formatted side-by-side comparison
7. Save results to output/comparison/

Note: Raw PTT data is in .pkl format (created by existing scraper).
The pickle module is used only for reading these trusted local data files.
"""

import argparse
import json
import pickle  # nosec - reading trusted local PTT data files
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PICKLE_AFTER_PATH, PICKLE_DURING_PATH,
    OPINION_DICT_PATH, OUTPUT_DIR,
)
from sentiment import load_opinion_dictionary
from evaluate_sentiment import (
    load_push_data, evaluate_dictionary, find_best_threshold,
)
from evaluate_ff5_regression import (
    load_total_table, run_regression_comparison, FF5_COLS, RETURN_COL,
)
from text_processing_v2 import load_stopwords


# Paths
OLD_DICT_PATH = OUTPUT_DIR / "total_dictionary.csv"
COMPUTED_OLD_PATH = OUTPUT_DIR / "computed_sentiment_dict.csv"
NEW_DICT_PATH = OUTPUT_DIR / "v2" / "total_dictionary_v2.csv"
COMPARISON_DIR = OUTPUT_DIR / "comparison"


def load_or_build_old_dict():
    """Load the old pipeline dictionary (or the merged one)."""
    if OLD_DICT_PATH.exists():
        print(f"  Loading old dictionary from {OLD_DICT_PATH}")
        return pd.read_csv(OLD_DICT_PATH)
    elif COMPUTED_OLD_PATH.exists():
        print(f"  Loading computed dictionary and merging with opinion dict...")
        computed = pd.read_csv(COMPUTED_OLD_PATH)
        opinion = load_opinion_dictionary(OPINION_DICT_PATH)
        return pd.concat([computed, opinion[['word', 'sentiment_score']]],
                         ignore_index=True)
    else:
        print("  [ERROR] Old dictionary not found. Run run_sentiment.py first.")
        return None


def load_or_build_new_dict():
    """Load the new v2 pipeline dictionary."""
    if NEW_DICT_PATH.exists():
        print(f"  Loading v2 dictionary from {NEW_DICT_PATH}")
        return pd.read_csv(NEW_DICT_PATH)
    else:
        print("  [ERROR] V2 dictionary not found. Run run_sentiment_v2.py first.")
        return None


def format_comparison_table(old_results, new_results, label_old="v1 (old)",
                            label_new="v2 (new)"):
    """Format a side-by-side comparison table."""
    metrics = [
        ('n_comments', 'Comments scored', 'd'),
        ('coverage', 'Coverage', '.1%'),
        ('acc_3class', '3-class accuracy', '.4f'),
        ('f1_3class_macro', '3-class macro F1', '.4f'),
        ('kappa', "Cohen's kappa", '.4f'),
        ('acc_binary', 'Binary accuracy', '.4f'),
        ('f1_binary', 'Binary F1', '.4f'),
        ('pearson_r', 'Pearson r', '.4f'),
        ('spearman_r', 'Spearman r', '.4f'),
        ('mean_score_push', 'Mean score (push)', '+.4f'),
        ('mean_score_boo', 'Mean score (boo)', '+.4f'),
        ('mean_score_arrow', 'Mean score (arrow)', '+.4f'),
    ]

    lines = []
    header = f"  {'Metric':<22} {label_old:>14} {label_new:>14} {'Delta':>14}"
    lines.append(header)
    lines.append(f"  {'-' * 64}")

    for key, name, fmt in metrics:
        v_old = old_results.get(key, 0)
        v_new = new_results.get(key, 0)
        delta = v_new - v_old if isinstance(v_new, (int, float)) else 0

        # Format values as strings first, then right-align
        s_old = format(v_old, fmt)
        s_new = format(v_new, fmt)
        s_delta = format(delta, '+' + fmt.lstrip('+'))
        lines.append(f"  {name:<22} {s_old:>14} {s_new:>14} {s_delta:>14}")

    return '\n'.join(lines)


def format_regression_table(old_reg, new_reg, label_old="v1 (old)",
                            label_new="v2 (new)"):
    """Format regression comparison table."""
    metrics = [
        ('base_r2', 'FF5 R-squared', '.6f'),
        ('aug_r2', 'FF5+SENT R-squared', '.6f'),
        ('delta_r2', 'Delta R-squared', '+.6f'),
        ('base_adj_r2', 'FF5 Adj R-sq', '.6f'),
        ('aug_adj_r2', 'FF5+SENT Adj R-sq', '.6f'),
        ('sent_coef', 'SENT coefficient', '.6f'),
        ('sent_tstat', 'SENT t-statistic', '.4f'),
        ('sent_pvalue', 'SENT p-value', '.4e'),
        ('delta_aic', 'Delta AIC', '+.2f'),
        ('delta_bic', 'Delta BIC', '+.2f'),
    ]

    lines = []
    header = f"  {'Metric':<22} {label_old:>14} {label_new:>14}"
    lines.append(header)
    lines.append(f"  {'-' * 50}")

    for key, name, fmt in metrics:
        v_old = old_reg.get(key, np.nan)
        v_new = new_reg.get(key, np.nan)
        s_old = format(v_old, fmt)
        s_new = format(v_new, fmt)
        lines.append(f"  {name:<22} {s_old:>14} {s_new:>14}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description="Compare old vs new sentiment pipeline.")
    parser.add_argument("--skip-training", action="store_true",
                        help="Skip training; load existing dictionaries")
    parser.add_argument("--eval-only", action="store_true",
                        help="Same as --skip-training")
    parser.add_argument("--max-posts", type=int, default=None,
                        help="Limit posts for push-tag eval (faster testing)")
    parser.add_argument("--k", type=int, default=20,
                        help="k-NN neighbors (ignored with --skip-training)")
    parser.add_argument("--vector-size", type=int, default=150,
                        help="W2V vector size (ignored with --skip-training)")
    parser.add_argument("--no-push-eval", action="store_true",
                        help="Skip push-tag evaluation")
    parser.add_argument("--no-ff5-eval", action="store_true",
                        help="Skip FF5 regression evaluation")
    args = parser.parse_args()

    COMPARISON_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("SENTIMENT PIPELINE COMPARISON: v1 (old) vs v2 (improved)")
    print("=" * 70)

    # --- Load dictionaries ---
    print("\n[1/6] Loading dictionaries...")
    old_dict = load_or_build_old_dict()
    new_dict = load_or_build_new_dict()

    if old_dict is None or new_dict is None:
        print("\n[ERROR] Cannot proceed without both dictionaries.")
        print("  Run run_sentiment.py and run_sentiment_v2.py first,")
        print("  or use --skip-training if dictionaries already exist.")
        sys.exit(1)

    print(f"  Old dictionary: {len(old_dict)} words")
    print(f"  New dictionary: {len(new_dict)} words")

    stopwords = load_stopwords()

    # --- Push-tag evaluation ---
    push_results_old = {}
    push_results_new = {}

    if not args.no_push_eval:
        print(f"\n[2/6] Loading push data for evaluation...")
        push_data = load_push_data(PICKLE_AFTER_PATH, max_posts=args.max_posts)
        print(f"  Total pushes: {len(push_data)}")

        print(f"\n[3/6] Evaluating OLD dictionary (push-tag)...")
        print("-" * 50)
        _, _, old_grid = find_best_threshold(push_data, old_dict, stopwords=stopwords)

        # Get best result
        best_old_t = max(old_grid, key=lambda t: old_grid[t].get('f1_3class_macro', 0))
        push_results_old = old_grid[best_old_t]
        push_results_old['best_threshold'] = best_old_t

        print(f"\n  Detailed results at best threshold ({best_old_t}):")
        evaluate_dictionary(push_data, old_dict, threshold=best_old_t,
                            stopwords=stopwords)

        print(f"\n[4/6] Evaluating NEW dictionary (push-tag)...")
        print("-" * 50)
        _, _, new_grid = find_best_threshold(push_data, new_dict, stopwords=stopwords)

        best_new_t = max(new_grid, key=lambda t: new_grid[t].get('f1_3class_macro', 0))
        push_results_new = new_grid[best_new_t]
        push_results_new['best_threshold'] = best_new_t

        print(f"\n  Detailed results at best threshold ({best_new_t}):")
        evaluate_dictionary(push_data, new_dict, threshold=best_new_t,
                            stopwords=stopwords)
    else:
        print("\n[2-4/6] Skipping push-tag evaluation (--no-push-eval)")

    # --- FF5 regression ---
    reg_results_old = {}
    reg_results_new = {}

    if not args.no_ff5_eval:
        print(f"\n[5/6] Running FF5 regressions...")
        print("-" * 50)

        total_df = load_total_table()
        print(f"  Total table shape: {total_df.shape}")

        # Old: use existing normalized sentiment columns
        old_sent_cols = [c for c in total_df.columns if '分數_norm' in c]
        if old_sent_cols:
            old_sent_col = old_sent_cols[0]
            print(f"\n  Old sentiment column: {old_sent_col}")
            reg_results_old = run_regression_comparison(
                total_df, total_df[old_sent_col],
                sentiment_name='SENT_v1',
            )
        else:
            print("  [WARN] No normalized sentiment columns found in total_table.csv")

        # New: load v2 daily sentiment if available
        new_daily_path = OUTPUT_DIR / "v2" / "daily_sentiment_v2.csv"
        if new_daily_path.exists():
            print(f"\n  Loading v2 daily sentiment from {new_daily_path}")
            new_daily = pd.read_csv(new_daily_path)
            new_daily['Date'] = pd.to_datetime(new_daily['Date'])
            merged = total_df.merge(new_daily[['Date', 'sentiment_mean']],
                                    on='Date', how='left')
            print(f"\n  New sentiment column: SENT_v2")
            reg_results_new = run_regression_comparison(
                merged, merged['sentiment_mean'],
                sentiment_name='SENT_v2',
            )
        else:
            if len(old_sent_cols) > 1:
                alt_col = old_sent_cols[1]
                print(f"\n  V2 daily sentiment not found; comparing with {alt_col}")
                reg_results_new = run_regression_comparison(
                    total_df, total_df[alt_col],
                    sentiment_name='SENT_alt',
                )
            else:
                print("  [INFO] V2 daily sentiment not yet aggregated.")
                print("  Run run_sentiment_v2.py first, then aggregate daily scores.")
    else:
        print("\n[5/6] Skipping FF5 regression (--no-ff5-eval)")

    # --- Comparison summary ---
    print(f"\n{'=' * 70}")
    print("COMPARISON SUMMARY")
    print(f"{'=' * 70}")

    if push_results_old and push_results_new:
        print("\n--- Push-Tag Classification ---")
        print(format_comparison_table(push_results_old, push_results_new))

    if reg_results_old and reg_results_new:
        print("\n--- FF5 Regression ---")
        print(format_regression_table(reg_results_old, reg_results_new))

    # --- Save results ---
    print(f"\n[6/6] Saving comparison results to {COMPARISON_DIR}...")

    if push_results_old and push_results_new:
        push_summary = {
            'v1': {k: v for k, v in push_results_old.items()
                   if k != 'confusion_matrix'},
            'v2': {k: v for k, v in push_results_new.items()
                   if k != 'confusion_matrix'},
        }
        with open(COMPARISON_DIR / "push_tag_comparison.json", "w") as f:
            json.dump(push_summary, f, indent=2, default=str)

    if reg_results_old and reg_results_new:
        reg_summary = {'v1': reg_results_old, 'v2': reg_results_new}
        with open(COMPARISON_DIR / "ff5_regression_comparison.json", "w") as f:
            json.dump(reg_summary, f, indent=2, default=str)

    if push_results_old and push_results_new:
        summary_rows = []
        for name, res in [('v1', push_results_old), ('v2', push_results_new)]:
            row = {'pipeline': name}
            row.update({k: v for k, v in res.items() if k != 'confusion_matrix'})
            summary_rows.append(row)
        pd.DataFrame(summary_rows).to_csv(
            COMPARISON_DIR / "comparison_summary.csv", index=False
        )

    print("\nDone! Comparison complete.")


if __name__ == "__main__":
    main()

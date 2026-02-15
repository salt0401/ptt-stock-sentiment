"""Push-tag sentiment evaluation.

Uses PTT push tags (推/噓/→) as ground-truth sentiment labels to evaluate
how well a sentiment dictionary captures comment polarity.

Metrics:
- 3-class accuracy / macro-F1 / Cohen's kappa (push vs boo vs arrow)
- Binary accuracy / F1 (positive vs negative, excluding arrows)
- Pearson/Spearman correlation (continuous score vs numeric label)
- Mean score per class (sanity check)
- Confusion matrix
"""

import pickle  # Required: raw PTT data stored in .pkl by existing scraper
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, cohen_kappa_score, confusion_matrix,
    classification_report,
)
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))

from text_processing_v2 import (
    extract_push_contents_with_tags, segment_with_jieba_v2,
    load_stopwords, filter_tokens,
)
from config import PICKLE_AFTER_PATH, PICKLE_DURING_PATH


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_push_data(pkl_path, max_posts=None):
    """Load push data from a pickle file, extracting tags as ground truth.

    Handles both formats:
    - DataFrame with ['Date', 'Pushes'] columns (current format)
    - List of dicts with 'Pushes' key (legacy format)

    Parameters
    ----------
    pkl_path : str or Path
        Path to .pkl file (after_market or during_market).
    max_posts : int, optional
        Limit number of posts to load (for faster testing).

    Returns
    -------
    list[dict] : Each dict has keys: tag, content, userid, ipdatetime.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)

    # Handle DataFrame format (columns: Date, Pushes)
    if isinstance(data, pd.DataFrame):
        push_series = data['Pushes'].iloc[:max_posts]
        all_pushes = []
        for pushes in push_series:
            if isinstance(pushes, list):
                tagged = extract_push_contents_with_tags(pushes)
                all_pushes.extend(tagged)
        return all_pushes

    # Handle list-of-dicts format
    all_pushes = []
    for post in data[:max_posts]:
        if isinstance(post, dict):
            pushes = post.get('Pushes', [])
            tagged = extract_push_contents_with_tags(pushes)
            all_pushes.extend(tagged)

    return all_pushes


# ---------------------------------------------------------------------------
# Comment scoring
# ---------------------------------------------------------------------------

def score_comment(words, sentiment_dict, stopwords=None):
    """Score a comment by averaging word-level sentiment scores.

    Parameters
    ----------
    words : list[str]
        Segmented tokens from a comment.
    sentiment_dict : dict
        {word: sentiment_score}.
    stopwords : set, optional
        Stopwords to filter before scoring.

    Returns
    -------
    float or None : Mean sentiment score, or None if no words found.
    """
    if stopwords:
        words = filter_tokens(words, stopwords)

    scores = [sentiment_dict[w] for w in words if w in sentiment_dict]
    if not scores:
        return None
    return np.mean(scores)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

TAG_MAP = {'推': 1, '噓': -1, '→': 0}
TAG_NAMES = {1: 'push', -1: 'boo', 0: 'arrow'}


def evaluate_dictionary(push_data, dict_df, threshold=0.0, stopwords=None,
                        verbose=True):
    """Evaluate a sentiment dictionary against push-tag ground truth.

    Parameters
    ----------
    push_data : list[dict]
        Push data with 'tag' and 'content' keys.
    dict_df : pd.DataFrame
        Sentiment dictionary with ['word', 'sentiment_score'].
    threshold : float
        Score threshold for positive/negative classification.
        score > threshold -> positive, score < -threshold -> negative, else neutral.
    stopwords : set, optional
        Stopwords for filtering.
    verbose : bool
        Print detailed results.

    Returns
    -------
    dict : Evaluation metrics.
    """
    sentiment_dict = dict_df.set_index('word')['sentiment_score'].to_dict()

    # Score each comment
    records = []
    for item in push_data:
        tag_str = item['tag']
        if tag_str not in TAG_MAP:
            continue
        true_label = TAG_MAP[tag_str]

        words = list(segment_with_jieba_v2([item['content']])[0])
        score = score_comment(words, sentiment_dict, stopwords)
        if score is None:
            continue

        # Predicted class based on threshold
        if score > threshold:
            pred_label = 1
        elif score < -threshold:
            pred_label = -1
        else:
            pred_label = 0

        records.append({
            'true_label': true_label,
            'pred_label': pred_label,
            'score': score,
        })

    if not records:
        print("  [WARN] No scoreable comments found.")
        return {}

    df = pd.DataFrame(records)
    y_true = df['true_label'].values
    y_pred = df['pred_label'].values
    scores = df['score'].values

    results = {}

    # --- 3-class metrics ---
    results['n_comments'] = len(df)
    results['coverage'] = len(df) / len(push_data)
    results['acc_3class'] = accuracy_score(y_true, y_pred)
    results['f1_3class_macro'] = f1_score(y_true, y_pred, average='macro',
                                          zero_division=0)
    results['kappa'] = cohen_kappa_score(y_true, y_pred)

    # --- Binary metrics (push vs boo only) ---
    binary_mask = (y_true != 0)
    if binary_mask.sum() > 0:
        y_true_bin = y_true[binary_mask]
        y_pred_bin = y_pred[binary_mask]
        # Map to 0/1 for binary metrics
        yt_bin = (y_true_bin > 0).astype(int)
        yp_bin = (y_pred_bin > 0).astype(int)
        results['acc_binary'] = accuracy_score(yt_bin, yp_bin)
        results['f1_binary'] = f1_score(yt_bin, yp_bin, zero_division=0)
        results['n_binary'] = int(binary_mask.sum())
    else:
        results['acc_binary'] = 0.0
        results['f1_binary'] = 0.0
        results['n_binary'] = 0

    # --- Correlation ---
    numeric_true = y_true.astype(float)
    r_pearson, p_pearson = pearsonr(scores, numeric_true)
    r_spearman, p_spearman = spearmanr(scores, numeric_true)
    results['pearson_r'] = r_pearson
    results['pearson_p'] = p_pearson
    results['spearman_r'] = r_spearman
    results['spearman_p'] = p_spearman

    # --- Mean score per class ---
    for label, name in TAG_NAMES.items():
        mask = y_true == label
        if mask.sum() > 0:
            results[f'mean_score_{name}'] = float(scores[mask].mean())
            results[f'count_{name}'] = int(mask.sum())
        else:
            results[f'mean_score_{name}'] = 0.0
            results[f'count_{name}'] = 0

    # --- Confusion matrix ---
    labels = [-1, 0, 1]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    results['confusion_matrix'] = cm

    if verbose:
        print(f"\n  Comments scored: {results['n_comments']} "
              f"(coverage: {results['coverage']:.1%})")
        print(f"  Threshold: {threshold}")
        print(f"\n  3-class accuracy:  {results['acc_3class']:.4f}")
        print(f"  3-class macro F1:  {results['f1_3class_macro']:.4f}")
        print(f"  Cohen's kappa:     {results['kappa']:.4f}")
        print(f"\n  Binary accuracy:   {results['acc_binary']:.4f} "
              f"(n={results['n_binary']})")
        print(f"  Binary F1:         {results['f1_binary']:.4f}")
        print(f"\n  Pearson r:  {results['pearson_r']:.4f} (p={results['pearson_p']:.2e})")
        print(f"  Spearman r: {results['spearman_r']:.4f} (p={results['spearman_p']:.2e})")
        print(f"\n  Mean score by class:")
        for label, name in TAG_NAMES.items():
            print(f"    {name:>6}: {results[f'mean_score_{name}']:+.4f} "
                  f"(n={results[f'count_{name}']})")
        print(f"\n  Confusion matrix (rows=true, cols=pred):")
        print(f"          boo  arrow  push")
        for i, lbl in enumerate(['boo', 'arrow', 'push']):
            print(f"  {lbl:>6} {cm[i]}")

    return results


# ---------------------------------------------------------------------------
# Threshold grid search
# ---------------------------------------------------------------------------

def find_best_threshold(push_data, dict_df, stopwords=None,
                        thresholds=None, metric='f1_3class_macro'):
    """Grid search for the best classification threshold.

    Parameters
    ----------
    push_data : list[dict]
    dict_df : pd.DataFrame
    stopwords : set, optional
    thresholds : list[float], optional
        Defaults to [-0.1, -0.05, 0, 0.05, 0.1].
    metric : str
        Metric to optimize.

    Returns
    -------
    tuple : (best_threshold, best_score, all_results)
    """
    if thresholds is None:
        thresholds = [-0.1, -0.05, 0.0, 0.05, 0.1]

    best_threshold = 0.0
    best_score = -1.0
    all_results = {}

    for t in thresholds:
        print(f"\n  --- Threshold = {t} ---")
        res = evaluate_dictionary(push_data, dict_df, threshold=t,
                                  stopwords=stopwords, verbose=False)
        score = res.get(metric, 0.0)
        all_results[t] = res
        print(f"    {metric}: {score:.4f}")

        if score > best_score:
            best_score = score
            best_threshold = t

    print(f"\n  Best threshold: {best_threshold} ({metric}={best_score:.4f})")
    return best_threshold, best_score, all_results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Standalone evaluation of a sentiment dictionary."""
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate sentiment dictionary via push tags.")
    parser.add_argument("--dict-csv", type=str, required=True,
                        help="Path to sentiment dictionary CSV (word, sentiment_score)")
    parser.add_argument("--pkl", type=str, default=None,
                        help="Path to .pkl file (default: after_market.pkl)")
    parser.add_argument("--max-posts", type=int, default=None,
                        help="Limit number of posts for faster testing")
    parser.add_argument("--threshold", type=float, default=0.0,
                        help="Classification threshold")
    parser.add_argument("--grid-search", action="store_true",
                        help="Run threshold grid search")
    args = parser.parse_args()

    pkl_path = Path(args.pkl) if args.pkl else PICKLE_AFTER_PATH
    print(f"Loading push data from {pkl_path}...")
    push_data = load_push_data(pkl_path, max_posts=args.max_posts)
    print(f"  Total pushes: {len(push_data)}")

    print(f"Loading dictionary from {args.dict_csv}...")
    dict_df = pd.read_csv(args.dict_csv)
    if 'word' not in dict_df.columns:
        dict_df.rename(columns={'情緒字詞': 'word', '情緒分數': 'sentiment_score'},
                       inplace=True)
    print(f"  Dictionary size: {len(dict_df)}")

    stopwords = load_stopwords()

    if args.grid_search:
        print("\nRunning threshold grid search...")
        find_best_threshold(push_data, dict_df, stopwords=stopwords)
    else:
        print(f"\nEvaluating with threshold={args.threshold}...")
        evaluate_dictionary(push_data, dict_df, threshold=args.threshold,
                            stopwords=stopwords)


if __name__ == "__main__":
    main()

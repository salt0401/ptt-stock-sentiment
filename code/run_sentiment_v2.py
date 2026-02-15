"""Entry point: Improved sentiment analysis pipeline (v2).

Usage:
    python run_sentiment_v2.py [--skip-w2v] [--vector-size 150] [--k 20]

Outputs to output/v2/ (separate from original output/).

Improvements over run_sentiment.py:
1. Expanded stopwords (~300 words) + custom jieba dictionary
2. Token filtering (min_len=2, no pure numbers)
3. Skip-gram W2V with higher dimensions (150D) and min_count=5
4. No PCA — uses raw vectors for k-NN
5. Cosine k-NN sentiment propagation with confidence weighting
"""

import argparse
import json
import pickle  # Required: existing raw data is in .pkl format from scraper
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PICKLE_AFTER_PATH, PICKLE_DURING_PATH,
    OPINION_DICT_PATH, OUTPUT_DIR,
)
from text_processing_v2 import (
    load_stopwords, extract_push_contents, segment_with_jieba_v2,
    build_word_frequency_v2,
)
from sentiment_v2 import (
    train_word2vec_v2, build_sentiment_dictionary_v2,
)
from sentiment import load_opinion_dictionary


# Output directory for v2 results
OUTPUT_V2 = OUTPUT_DIR / "v2"

# v2-specific file paths
VECTORS_V2_PATH = OUTPUT_V2 / "w2v_vectors_v2.npy"
VOCAB_V2_PATH = OUTPUT_V2 / "vocab_v2.json"
SEGMENTED_V2_PATH = OUTPUT_V2 / "segmented_sentences_v2.json"
COMPUTED_DICT_V2_PATH = OUTPUT_V2 / "computed_sentiment_dict_v2.csv"
TOTAL_DICT_V2_PATH = OUTPUT_V2 / "total_dictionary_v2.csv"
WORD_FREQ_V2_PATH = OUTPUT_V2 / "word_frequency_v2.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run improved sentiment analysis pipeline (v2)."
    )
    parser.add_argument("--skip-w2v", action="store_true",
                        help="Skip W2V training; load existing model/vectors")
    parser.add_argument("--after-pkl", type=str, default=None,
                        help="Path to after_market.pkl")
    parser.add_argument("--during-pkl", type=str, default=None,
                        help="Path to during_market.pkl")
    parser.add_argument("--vector-size", type=int, default=150,
                        help="W2V vector dimensionality (default: 150)")
    parser.add_argument("--min-count", type=int, default=5,
                        help="W2V minimum word frequency (default: 5)")
    parser.add_argument("--k", type=int, default=20,
                        help="Number of nearest neighbors for propagation (default: 20)")
    parser.add_argument("--batch-size", type=int, default=5000,
                        help="Batch size for k-NN propagation (default: 5000)")
    args = parser.parse_args()

    OUTPUT_V2.mkdir(parents=True, exist_ok=True)

    after_pkl = Path(args.after_pkl) if args.after_pkl else PICKLE_AFTER_PATH
    during_pkl = Path(args.during_pkl) if args.during_pkl else PICKLE_DURING_PATH

    # --- Step 1: Load raw data ---
    print("=" * 60)
    print("IMPROVED SENTIMENT PIPELINE (v2)")
    print("=" * 60)
    print("\nStep 1: Loading raw data...")
    with open(after_pkl, 'rb') as f:
        data_after = pickle.load(f)
    dfA = pd.DataFrame(data_after)

    with open(during_pkl, 'rb') as f:
        data_during = pickle.load(f)
    dfD = pd.DataFrame(data_during)

    print(f"  After-market: {len(dfA)} posts, During-market: {len(dfD)} posts")

    # --- Step 2: Extract push contents (v2 compatible) ---
    print("\nStep 2: Extracting push contents...")
    all_contents_after = []
    for pushes in dfA['Pushes']:
        contents = extract_push_contents(pushes)
        all_contents_after.append(contents)

    all_contents_during = []
    for pushes in dfD['Pushes']:
        contents = extract_push_contents(pushes)
        all_contents_during.append(contents)

    flat_after = [c for day_contents in all_contents_after for c in day_contents]
    flat_during = [c for day_contents in all_contents_during for c in day_contents]
    print(f"  After-market comments: {len(flat_after)}")
    print(f"  During-market comments: {len(flat_during)}")

    # --- Step 3: Segment with jieba (custom dict loaded) ---
    print("\nStep 3: Segmenting with jieba (v2 + custom dict)...")
    seg_after = segment_with_jieba_v2(flat_after)
    seg_during = segment_with_jieba_v2(flat_during)
    sentence_cutted = seg_after + seg_during
    print(f"  Total segmented sentences: {len(sentence_cutted)}")

    # Save segmented data
    with open(SEGMENTED_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(sentence_cutted, f, ensure_ascii=False)
    print(f"  Saved to {SEGMENTED_V2_PATH}")

    # --- Step 4: Word frequency (v2 with expanded stopwords) ---
    print("\nStep 4: Building word frequency (v2)...")
    stopwords = load_stopwords()
    print(f"  Loaded {len(stopwords)} stopwords")

    word_freq = build_word_frequency_v2(sentence_cutted, stopwords=stopwords)
    print(f"  Unique tokens after filtering: {len(word_freq)}")

    # Save word frequency
    with open(WORD_FREQ_V2_PATH, "w", encoding="utf-8") as f:
        json.dump(word_freq, f, ensure_ascii=False, indent=0)

    top_20 = list(word_freq.items())[:20]
    for i, (word, count) in enumerate(top_20, 1):
        print(f"  {i}. {word}: {count}")

    # --- Step 5: Train Word2Vec (v2: skip-gram, 150D, min_count=5) ---
    if args.skip_w2v and VECTORS_V2_PATH.exists() and VOCAB_V2_PATH.exists():
        print("\nStep 5: Loading existing v2 vectors...")
        vectors = np.load(VECTORS_V2_PATH)
        with open(VOCAB_V2_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"  Loaded {len(vocab)} words, vectors shape: {vectors.shape}")

        model = None
    else:
        print(f"\nStep 5: Training Word2Vec v2 (sg=1, dim={args.vector_size}, "
              f"min_count={args.min_count})...")
        model = train_word2vec_v2(
            sentence_cutted,
            vector_size=args.vector_size,
            min_count=args.min_count,
        )
        vocab = model.wv.index_to_key
        vectors = model.wv[vocab]
        np.save(VECTORS_V2_PATH, vectors)
        with open(VOCAB_V2_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Vector shape: {vectors.shape}")
        print(f"  Saved to {VECTORS_V2_PATH}")

    # --- Step 6: Load opinion dictionary ---
    print("\nStep 6: Loading opinion dictionary...")
    opinion_dict = load_opinion_dictionary(OPINION_DICT_PATH)
    print(f"  Opinion dictionary size: {len(opinion_dict)}")

    # --- Step 7-10: Build sentiment dictionary (map, split, propagate, merge) ---
    print("\nStep 7-10: Building sentiment dictionary v2...")
    if model is not None:
        total_dict, computed_dict = build_sentiment_dictionary_v2(
            model, opinion_dict, word_freq=word_freq,
            k=args.k, batch_size=args.batch_size,
        )
    else:
        # Manual pipeline when loading from saved vectors
        from sentiment_v2 import (
            map_sentiment_scores_v2, split_scored_unscored_v2,
            propagate_sentiment_knn,
        )
        mapped_df = map_sentiment_scores_v2(vocab, opinion_dict, word_freq)
        mask_scored = mapped_df['sentiment_score'] != 0.0

        df_unscored = mapped_df[~mask_scored].reset_index(drop=True)
        df_scored = mapped_df[mask_scored].reset_index(drop=True)
        vecs_unscored = vectors[~mask_scored.values]
        vecs_scored = vectors[mask_scored.values]

        print(f"  Scored: {len(df_scored)}, Unscored: {len(df_unscored)}")
        print(f"  Propagating sentiment (k={args.k})...")

        propagated = propagate_sentiment_knn(
            vecs_unscored, vecs_scored,
            df_scored['sentiment_score'].to_numpy(),
            confidence_scored=df_scored['confidence'].to_numpy(),
            k=args.k, batch_size=args.batch_size,
        )
        df_unscored = df_unscored.copy()
        df_unscored['sentiment_score'] = propagated

        computed_dict = df_unscored[['word', 'sentiment_score']]
        total_dict = pd.concat(
            [computed_dict, opinion_dict[['word', 'sentiment_score']]],
            ignore_index=True,
        )

    # --- Step 11: Save results ---
    print("\nStep 11: Saving results...")
    computed_dict.to_csv(COMPUTED_DICT_V2_PATH, index=False)
    print(f"  Computed dictionary: {COMPUTED_DICT_V2_PATH} ({len(computed_dict)} words)")

    total_dict.to_csv(TOTAL_DICT_V2_PATH, index=False)
    print(f"  Total dictionary: {TOTAL_DICT_V2_PATH} ({len(total_dict)} words)")

    # --- Summary statistics ---
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Total words in dictionary: {len(total_dict)}")
    print(f"  Mean sentiment score: {total_dict['sentiment_score'].mean():.6f}")
    print(f"  Std sentiment score:  {total_dict['sentiment_score'].std():.6f}")
    print(f"  Min: {total_dict['sentiment_score'].min():.6f}")
    print(f"  Max: {total_dict['sentiment_score'].max():.6f}")

    pos_count = (total_dict['sentiment_score'] > 0).sum()
    neg_count = (total_dict['sentiment_score'] < 0).sum()
    zero_count = (total_dict['sentiment_score'] == 0).sum()
    print(f"  Positive: {pos_count}, Negative: {neg_count}, Zero: {zero_count}")
    print("\nDone! v2 pipeline complete.")


if __name__ == "__main__":
    main()

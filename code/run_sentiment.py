"""Entry point: Sentiment analysis pipeline.

Usage:
    python run_sentiment.py [--skip-w2v] [--vector-size 150] [--k 20]

Outputs to output/ directory.

Pipeline:
1. Expanded stopwords (~300 words) + custom jieba dictionary
2. Token filtering (min_len=2, no pure numbers)
3. Skip-gram W2V with higher dimensions (150D) and min_count=5
4. No PCA — uses raw vectors for k-NN
5. Cosine k-NN sentiment propagation with confidence weighting

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

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PICKLE_AFTER_PATH, PICKLE_DURING_PATH,
    OPINION_DICT_PATH, OUTPUT_DIR,
)
from text_processing import (
    load_stopwords, extract_push_contents, segment_with_jieba,
    build_word_frequency,
)
from sentiment import (
    load_opinion_dictionary, train_word2vec, build_sentiment_dictionary,
)


# Output file paths
VECTORS_PATH = OUTPUT_DIR / "w2v_vectors.npy"
VOCAB_PATH = OUTPUT_DIR / "vocab.json"
SEGMENTED_PATH = OUTPUT_DIR / "segmented_sentences.json"
COMPUTED_DICT_PATH = OUTPUT_DIR / "computed_sentiment_dict.csv"
TOTAL_DICT_PATH = OUTPUT_DIR / "total_dictionary.csv"
WORD_FREQ_PATH = OUTPUT_DIR / "word_frequency.json"


def main():
    parser = argparse.ArgumentParser(
        description="Run sentiment analysis pipeline."
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

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    after_pkl = Path(args.after_pkl) if args.after_pkl else PICKLE_AFTER_PATH
    during_pkl = Path(args.during_pkl) if args.during_pkl else PICKLE_DURING_PATH

    # --- Step 1: Load raw data ---
    print("=" * 60)
    print("SENTIMENT ANALYSIS PIPELINE")
    print("=" * 60)
    print("\nStep 1: Loading raw data...")
    with open(after_pkl, 'rb') as f:
        data_after = pickle.load(f)  # nosec - trusted local file
    dfA = pd.DataFrame(data_after)

    with open(during_pkl, 'rb') as f:
        data_during = pickle.load(f)  # nosec - trusted local file
    dfD = pd.DataFrame(data_during)

    print(f"  After-market: {len(dfA)} posts, During-market: {len(dfD)} posts")

    # --- Step 2: Extract push contents ---
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
    print("\nStep 3: Segmenting with jieba + custom dict...")
    seg_after = segment_with_jieba(flat_after)
    seg_during = segment_with_jieba(flat_during)
    sentence_cutted = seg_after + seg_during
    print(f"  Total segmented sentences: {len(sentence_cutted)}")

    # Save segmented data
    with open(SEGMENTED_PATH, "w", encoding="utf-8") as f:
        json.dump(sentence_cutted, f, ensure_ascii=False)
    print(f"  Saved to {SEGMENTED_PATH}")

    # --- Step 4: Word frequency ---
    print("\nStep 4: Building word frequency...")
    stopwords = load_stopwords()
    print(f"  Loaded {len(stopwords)} stopwords")

    word_freq = build_word_frequency(sentence_cutted, stopwords=stopwords)
    print(f"  Unique tokens after filtering: {len(word_freq)}")

    # Save word frequency
    with open(WORD_FREQ_PATH, "w", encoding="utf-8") as f:
        json.dump(word_freq, f, ensure_ascii=False, indent=0)

    top_20 = list(word_freq.items())[:20]
    for i, (word, count) in enumerate(top_20, 1):
        print(f"  {i}. {word}: {count}")

    # --- Step 5: Train Word2Vec (skip-gram, 150D, min_count=5) ---
    if args.skip_w2v and VECTORS_PATH.exists() and VOCAB_PATH.exists():
        print("\nStep 5: Loading existing vectors...")
        vectors = np.load(VECTORS_PATH)
        with open(VOCAB_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)
        print(f"  Loaded {len(vocab)} words, vectors shape: {vectors.shape}")

        model = None
    else:
        print(f"\nStep 5: Training Word2Vec (sg=1, dim={args.vector_size}, "
              f"min_count={args.min_count})...")
        model = train_word2vec(
            sentence_cutted,
            vector_size=args.vector_size,
            min_count=args.min_count,
        )
        vocab = model.wv.index_to_key
        vectors = model.wv[vocab]
        np.save(VECTORS_PATH, vectors)
        with open(VOCAB_PATH, "w", encoding="utf-8") as f:
            json.dump(vocab, f, ensure_ascii=False)
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Vector shape: {vectors.shape}")
        print(f"  Saved to {VECTORS_PATH}")

    # --- Step 6: Load opinion dictionary ---
    print("\nStep 6: Loading opinion dictionary...")
    opinion_dict = load_opinion_dictionary(OPINION_DICT_PATH)
    print(f"  Opinion dictionary size: {len(opinion_dict)}")

    # --- Step 7-10: Build sentiment dictionary (map, split, propagate, merge) ---
    print("\nStep 7-10: Building sentiment dictionary...")
    if model is not None:
        total_dict, computed_dict = build_sentiment_dictionary(
            model, opinion_dict, word_freq=word_freq,
            k=args.k, batch_size=args.batch_size,
        )
    else:
        # Manual pipeline when loading from saved vectors
        from sentiment import (
            map_sentiment_scores, split_scored_unscored,
            propagate_sentiment_knn,
        )
        mapped_df = map_sentiment_scores(vocab, opinion_dict, word_freq)
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
    computed_dict.to_csv(COMPUTED_DICT_PATH, index=False)
    print(f"  Computed dictionary: {COMPUTED_DICT_PATH} ({len(computed_dict)} words)")

    total_dict.to_csv(TOTAL_DICT_PATH, index=False)
    print(f"  Total dictionary: {TOTAL_DICT_PATH} ({len(total_dict)} words)")

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
    print("\nDone! Pipeline complete.")


if __name__ == "__main__":
    main()

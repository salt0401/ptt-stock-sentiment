"""Entry point: Full sentiment analysis pipeline.

Usage:
    python run_sentiment.py [--skip-w2v] [--skip-distances]

Steps:
    1. Load raw comment data from pickles
    2. Extract push contents, filter URLs
    3. Segment with jieba
    4. Train Word2Vec model
    5. PCA reduce to 4D
    6. Map existing sentiment scores
    7. Calculate distance-weighted sentiment for unscored words
    8. Save computed_sentiment_dict.csv
    9. Merge with original opinion dictionary
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add code directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import (
    PICKLE_AFTER_PATH, PICKLE_DURING_PATH,
    OPINION_DICT_PATH, NO_SCORE_DICT_PATH, OUTPUT_DIR,
    VECTORS_PATH, DISTANCES_PATH, WEIGHTED_DISTANCES_PATH,
    SEGMENTED_JSON_PATH,
)
from text_processing import extract_push_contents, segment_with_jieba, build_word_frequency
from sentiment import (
    load_opinion_dictionary,
    train_word2vec,
    reduce_dimensions,
    map_sentiment_scores,
    split_scored_unscored,
    calculate_distances,
    calculate_weighted_distances,
    compute_final_sentiment,
    merge_dictionaries,
)


def main():
    parser = argparse.ArgumentParser(description="Run full sentiment analysis pipeline.")
    parser.add_argument("--skip-w2v", action="store_true",
                        help="Skip W2V training; load existing vectors from disk")
    parser.add_argument("--skip-distances", action="store_true",
                        help="Skip distance calculations; load existing distances from disk")
    parser.add_argument("--after-pkl", type=str, default=None,
                        help="Path to after_market.pkl (default: config path)")
    parser.add_argument("--during-pkl", type=str, default=None,
                        help="Path to during_market.pkl (default: config path)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    after_pkl = Path(args.after_pkl) if args.after_pkl else PICKLE_AFTER_PATH
    during_pkl = Path(args.during_pkl) if args.during_pkl else PICKLE_DURING_PATH

    # --- Step 1: Load raw data ---
    print("Step 1: Loading raw data...")
    with open(after_pkl, 'rb') as f:
        data_after = pickle.load(f)
    dfA = pd.DataFrame(data_after)

    with open(during_pkl, 'rb') as f:
        data_during = pickle.load(f)
    dfD = pd.DataFrame(data_during)

    print(f"  After-market: {len(dfA)} posts, During-market: {len(dfD)} posts")

    # --- Step 2: Extract push contents ---
    print("Step 2: Extracting push contents...")
    all_contents_after = []
    for pushes in dfA['Pushes']:
        contents = extract_push_contents(pushes)
        all_contents_after.append(contents)

    all_contents_during = []
    for pushes in dfD['Pushes']:
        contents = extract_push_contents(pushes)
        all_contents_during.append(contents)

    # Flatten to 1D lists
    flat_after = [c for day_contents in all_contents_after for c in day_contents]
    flat_during = [c for day_contents in all_contents_during for c in day_contents]
    print(f"  After-market comments: {len(flat_after)}, During-market comments: {len(flat_during)}")

    # --- Step 3: Segment with jieba ---
    print("Step 3: Segmenting with jieba...")
    seg_after = segment_with_jieba(flat_after)
    seg_during = segment_with_jieba(flat_during)
    sentence_cutted = seg_after + seg_during
    print(f"  Total segmented sentences: {len(sentence_cutted)}")

    # Save segmented data
    with open(SEGMENTED_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(sentence_cutted, f, ensure_ascii=False)
    print(f"  Saved segmented data to {SEGMENTED_JSON_PATH}")

    # --- Step 4: Word frequency ---
    print("Step 4: Building word frequency...")
    word_freq = build_word_frequency(sentence_cutted)
    top_20 = list(word_freq.items())[:20]
    for i, (word, count) in enumerate(top_20, 1):
        print(f"  {i}. {word}: {count}")

    # --- Step 5: Train Word2Vec ---
    if args.skip_w2v:
        print("Step 5: Loading existing vectors...")
        vectors = np.load(VECTORS_PATH)
        # We need vocab too — reload model or use saved data
        # For now, load from segmented json and rebuild vocab
        from gensim.models import Word2Vec as W2V
        print("  (Re-training W2V to get vocab ordering...)")
        model = W2V(sentence_cutted, vector_size=50, window=5, min_count=1, workers=4)
        vocab = model.wv.index_to_key
        vectors = model.wv[vocab]
    else:
        print("Step 5: Training Word2Vec model...")
        model = train_word2vec(sentence_cutted, vector_size=50, window=5, min_count=1)
        vocab = model.wv.index_to_key
        vectors = model.wv[vocab]
        np.save(VECTORS_PATH, vectors)
        print(f"  Vocabulary size: {len(vocab)}")
        print(f"  Saved vectors to {VECTORS_PATH}")

    # --- Step 6: PCA reduce to 4D ---
    print("Step 6: PCA dimensionality reduction to 4D...")
    vectors_reduced = reduce_dimensions(vectors, n_components=4)
    print(f"  Reduced shape: {vectors_reduced.shape}")

    # --- Step 7: Map sentiment scores ---
    print("Step 7: Mapping sentiment scores...")
    opinion_dict = load_opinion_dictionary(OPINION_DICT_PATH)
    mapped_df = map_sentiment_scores(vocab, opinion_dict)

    scored_count = (mapped_df['sentiment_score'] != 0).sum()
    print(f"  Words with existing scores: {scored_count}")
    print(f"  Words without scores: {len(mapped_df) - scored_count}")

    # --- Step 8: Split scored / unscored ---
    df_unscored, df_scored = split_scored_unscored(mapped_df, vectors_reduced)

    unscored_coords = df_unscored[['X', 'Y', 'Z', 'A']].to_numpy()
    scored_coords = df_scored[['X', 'Y', 'Z', 'A']].to_numpy()
    scored_values = df_scored['sentiment_score'].to_numpy()

    # --- Step 9: Distance-weighted sentiment propagation ---
    if args.skip_distances:
        print("Step 9: Loading existing distances...")
        distances = np.load(DISTANCES_PATH)
        weighted_distances = np.load(WEIGHTED_DISTANCES_PATH)
    else:
        print("Step 9a: Calculating inverse Minkowski distances (this may take a while)...")
        distances = calculate_distances(unscored_coords, scored_coords, p=4, n_chunks=10)
        np.save(DISTANCES_PATH, distances)
        print(f"  Saved distances to {DISTANCES_PATH}")

        print("Step 9b: Calculating weighted distances...")
        weighted_distances = calculate_weighted_distances(
            unscored_coords, scored_coords, scored_values, p=4, n_chunks=5
        )
        np.save(WEIGHTED_DISTANCES_PATH, weighted_distances)
        print(f"  Saved weighted distances to {WEIGHTED_DISTANCES_PATH}")

    # --- Step 10: Compute final sentiment ---
    print("Step 10: Computing final sentiment scores...")
    final_sentiment = compute_final_sentiment(weighted_distances, distances)
    df_unscored['sentiment_score'] = final_sentiment

    # --- Step 11: Save computed_sentiment_dict.csv ---
    print("Step 11: Saving computed dictionary...")
    df_unscored[['word', 'sentiment_score']].to_csv(NO_SCORE_DICT_PATH, index=False)
    print(f"  Saved to {NO_SCORE_DICT_PATH}")

    # --- Step 12: Merge dictionaries ---
    print("Step 12: Merging dictionaries...")
    computed_dict = df_unscored[['word', 'sentiment_score']]
    total_dict = merge_dictionaries(opinion_dict, computed_dict)
    print(f"  Total dictionary size: {len(total_dict)}")

    total_dict_path = OUTPUT_DIR / "total_dictionary.csv"
    total_dict.to_csv(total_dict_path, index=False)
    print(f"  Saved merged dictionary to {total_dict_path}")

    print("\nDone! Sentiment analysis pipeline complete.")


if __name__ == "__main__":
    main()

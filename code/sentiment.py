"""Sentiment analysis module — extracted from FinalSentimentScore.ipynb and PttStock_v2.ipynb.

Provides:
- Sentiment dictionary loading and merging
- Word2Vec training pipeline
- PCA dimensionality reduction
- Minkowski distance-weighted sentiment propagation
- CKIPTagger-based sentiment scoring
"""

import numpy as np
import pandas as pd
from collections import Counter
from itertools import chain
from multiprocessing import Pool

from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from gensim.models import Word2Vec

from config import OPINION_DICT_PATH, NO_SCORE_DICT_PATH
from column_map import SENTIMENT_COLUMNS


# ---------------------------------------------------------------------------
# Dictionary functions
# ---------------------------------------------------------------------------

def load_opinion_dictionary(path=None):
    """Load opinion_dict.xlsx sentiment dictionary.

    Parameters
    ----------
    path : str or Path, optional
        Path to opinion_dict.xlsx. Defaults to config.OPINION_DICT_PATH.

    Returns
    -------
    pd.DataFrame : Columns ['word', 'sentiment_score'].
    """
    if path is None:
        path = OPINION_DICT_PATH
    df = pd.read_excel(path)
    df.rename(columns=SENTIMENT_COLUMNS, inplace=True)
    return df


def load_computed_dictionary(path=None):
    """Load the computed sentiment dictionary CSV.

    Parameters
    ----------
    path : str or Path, optional
        Path to computed_sentiment_dict.csv. Defaults to config.NO_SCORE_DICT_PATH.

    Returns
    -------
    pd.DataFrame : Columns ['word', 'sentiment_score'].
    """
    if path is None:
        path = NO_SCORE_DICT_PATH
    df = pd.read_csv(path)
    df.rename(columns=SENTIMENT_COLUMNS, inplace=True)
    return df


def merge_dictionaries(opinion_df, computed_df):
    """Combine original opinion dictionary with computed sentiment scores.

    Parameters
    ----------
    opinion_df : pd.DataFrame
        Original opinion dictionary.
    computed_df : pd.DataFrame
        Computed dictionary for previously-unscored words.

    Returns
    -------
    pd.DataFrame : Merged dictionary with columns ['word', 'sentiment_score'].
    """
    return pd.concat([computed_df, opinion_df], ignore_index=True)


# ---------------------------------------------------------------------------
# Word2Vec pipeline (from FinalSentimentScore cells 16-19)
# ---------------------------------------------------------------------------

def train_word2vec(sentences, vector_size=50, window=5, min_count=1, workers=4):
    """Train a Word2Vec model on segmented sentences.

    Parameters
    ----------
    sentences : list[list[str]]
        Segmented sentences.
    vector_size : int
        Dimensionality of word vectors.
    window : int
        Context window size.
    min_count : int
        Minimum word frequency.
    workers : int
        Number of training threads.

    Returns
    -------
    Word2Vec : Trained model.
    """
    model = Word2Vec(
        sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=workers,
    )
    return model


def reduce_dimensions(vectors, n_components=4):
    """Reduce word vectors to lower dimensions using PCA.

    Parameters
    ----------
    vectors : np.ndarray
        Word vectors of shape (n_words, vector_size).
    n_components : int
        Target dimensionality.

    Returns
    -------
    np.ndarray : Reduced vectors of shape (n_words, n_components).
    """
    pca = PCA(n_components=n_components)
    return pca.fit_transform(vectors)


# ---------------------------------------------------------------------------
# Sentiment score mapping (from FinalSentimentScore cells 20-21)
# ---------------------------------------------------------------------------

def map_sentiment_scores(vocab, opinion_dict):
    """Map existing sentiment scores to vocabulary, 0 for unknown words.

    Parameters
    ----------
    vocab : list[str]
        Word2Vec vocabulary (ordered).
    opinion_dict : pd.DataFrame
        DataFrame with columns ['word', 'sentiment_score'].

    Returns
    -------
    pd.DataFrame : DataFrame with columns ['word', 'sentiment_score'].
    """
    emotion_map = opinion_dict.set_index('word')['sentiment_score'].to_dict()
    df = pd.DataFrame({'word': vocab})
    df['sentiment_score'] = df['word'].map(emotion_map).fillna(0)
    return df


def split_scored_unscored(merged_df, vectors_reduced):
    """Split into scored and unscored DataFrames with PCA coordinates.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with ['word', 'sentiment_score'].
    vectors_reduced : np.ndarray
        PCA-reduced vectors, same length as merged_df.

    Returns
    -------
    tuple : (df_with_score_zero, df_without_score_zero)
        Both DataFrames have columns ['word', 'sentiment_score', 'X', 'Y', 'Z', 'A'].
    """
    coord_cols = ['X', 'Y', 'Z', 'A']
    vr = pd.DataFrame(vectors_reduced, columns=coord_cols)
    full_df = pd.concat([merged_df, vr], axis=1)

    df_scored_zero = full_df[full_df['sentiment_score'] == 0.0].reset_index(drop=True)
    df_not_zero = full_df[full_df['sentiment_score'] != 0.0].reset_index(drop=True)

    return df_scored_zero, df_not_zero


# ---------------------------------------------------------------------------
# Distance-based sentiment propagation (from FinalSentimentScore cells 23-29)
# ---------------------------------------------------------------------------

# Module-level variables used by multiprocessing workers
_points2_global = None
_value_global = None


def _init_distance_worker(points2):
    """Initializer for distance calculation pool workers."""
    global _points2_global
    _points2_global = points2


def _calculate_distance_chunk(chunk):
    """Calculate sum of inverse Minkowski distances for a chunk of points."""
    distances = []
    for point1 in chunk:
        distance_row = 1 / cdist([point1], _points2_global, metric='minkowski', p=4)
        distances.append(np.sum(distance_row))
    return distances


def calculate_distances(unscored_coords, scored_coords, p=4, n_chunks=10):
    """Calculate parallel Minkowski distance sums from unscored to scored points.

    For each unscored point, computes the sum of 1/d(p, q) for all scored points q,
    where d is the Minkowski distance with parameter p.

    Parameters
    ----------
    unscored_coords : np.ndarray
        Coordinates of unscored words, shape (N, D).
    scored_coords : np.ndarray
        Coordinates of scored words, shape (M, D).
    p : int
        Minkowski distance parameter.
    n_chunks : int
        Number of chunks for parallel processing.

    Returns
    -------
    np.ndarray : Distance sums for each unscored word, shape (N,).
    """
    chunks = np.array_split(unscored_coords, n_chunks)

    with Pool(initializer=_init_distance_worker, initargs=(scored_coords,)) as pool:
        results = pool.map(_calculate_distance_chunk, chunks)

    return np.concatenate(results)


def _init_weighted_worker(points2, value):
    """Initializer for weighted distance calculation pool workers."""
    global _points2_global, _value_global
    _points2_global = points2
    _value_global = value


def _calculate_weighted_distance_chunk(chunk):
    """Calculate weighted distance sums for a chunk of points."""
    weighted_distances = []
    for point1 in chunk:
        inv_distances = 1 / cdist([point1], _points2_global, metric='minkowski', p=4).flatten()
        weighted_distance = np.dot(inv_distances, _value_global)
        weighted_distances.append(weighted_distance)
    return weighted_distances


def calculate_weighted_distances(unscored_coords, scored_coords, scores, p=4, n_chunks=5):
    """Calculate parallel weighted distance sums (score * 1/distance).

    Parameters
    ----------
    unscored_coords : np.ndarray
        Coordinates of unscored words, shape (N, D).
    scored_coords : np.ndarray
        Coordinates of scored words, shape (M, D).
    scores : np.ndarray
        Sentiment scores for scored words, shape (M,).
    p : int
        Minkowski distance parameter.
    n_chunks : int
        Number of chunks for parallel processing.

    Returns
    -------
    np.ndarray : Weighted distance sums for each unscored word, shape (N,).
    """
    chunks = np.array_split(unscored_coords, n_chunks)

    with Pool(initializer=_init_weighted_worker, initargs=(scored_coords, scores)) as pool:
        results = pool.map(_calculate_weighted_distance_chunk, chunks)

    return np.concatenate(results)


def compute_final_sentiment(weighted_distances, distances):
    """Compute final sentiment scores by dividing weighted distances by distances.

    Parameters
    ----------
    weighted_distances : np.ndarray
        Weighted distance sums, shape (N,).
    distances : np.ndarray
        Distance sums, shape (N,).

    Returns
    -------
    np.ndarray : Final sentiment scores, shape (N,).
    """
    return np.divide(weighted_distances, distances)


# ---------------------------------------------------------------------------
# CKIPTagger-based sentiment (from PttStock_v2 cell 25)
# ---------------------------------------------------------------------------

def calculate_push_sentiment_ckip(pushes, ws, opinion_pos, opinion_neg):
    """Calculate positive and negative sentiment from pushes using CKIPTagger.

    Parameters
    ----------
    pushes : list[dict]
        Push dictionaries with 'Content' key.
    ws : ckiptagger.WS
        CKIPTagger word segmentation model.
    opinion_pos : pd.DataFrame
        Positive opinion dictionary with ['word', 'sentiment_score'].
    opinion_neg : pd.DataFrame
        Negative opinion dictionary with ['word', 'sentiment_score'].

    Returns
    -------
    tuple : (positive_score, negative_score, total_score)
    """
    contents_only = [d.get("Content", "N/A") for d in pushes]

    word_sentence_list = ws(
        contents_only,
        sentence_segmentation=True,
        segment_delimiter_set={",", "。", ":", "?", "!", ";"},
    )
    one_dimensional_wordlist = list(chain(*word_sentence_list))

    positive_grades = 0
    for _, row in opinion_pos.iterrows():
        if row['word'] in one_dimensional_wordlist:
            positive_grades += row['sentiment_score']

    negative_grades = 0
    for _, row in opinion_neg.iterrows():
        if row['word'] in one_dimensional_wordlist:
            negative_grades += row['sentiment_score']

    total = positive_grades + negative_grades
    return positive_grades, negative_grades, total

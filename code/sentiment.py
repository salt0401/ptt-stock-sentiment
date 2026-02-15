"""Sentiment analysis module.

Approach:
1. Skip-gram Word2Vec (sg=1) with vector_size=150, min_count=5
2. No PCA dimensionality reduction — uses raw W2V vectors
3. Cosine distance k-NN propagation (k=20) with confidence-weighted IDW
4. Confidence weighting: log(1+freq)/log(1+max_freq) per word
5. Batch processing for memory efficiency
6. np.argpartition for O(M) top-k selection
"""

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

from config import OPINION_DICT_PATH
from column_map import SENTIMENT_COLUMNS


# ---------------------------------------------------------------------------
# Opinion dictionary loading
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


# ---------------------------------------------------------------------------
# Word2Vec — skip-gram, higher dim, min_count filtering
# ---------------------------------------------------------------------------

def train_word2vec(sentences, vector_size=150, window=5, min_count=5,
                      sg=1, epochs=10, negative=10, workers=4):
    """Train an improved Word2Vec model.

    Changes from v1:
    - sg=1 (skip-gram) captures rare-word semantics better than CBOW
    - vector_size=150 retains more structure (no PCA needed)
    - min_count=5 filters noise tokens that appear < 5 times
    - epochs=10 for better convergence (gensim default is 5)
    - negative=10 for better negative sampling

    Parameters
    ----------
    sentences : list[list[str]]
        Segmented sentences (list of token lists).
    vector_size : int
        Dimensionality of word vectors.
    window : int
        Context window size.
    min_count : int
        Ignore words with total frequency lower than this.
    sg : int
        1 for skip-gram, 0 for CBOW.
    epochs : int
        Number of training epochs.
    negative : int
        Number of negative samples.
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
        sg=sg,
        epochs=epochs,
        negative=negative,
        workers=workers,
    )
    return model


# ---------------------------------------------------------------------------
# Sentiment score mapping v2 — with confidence column
# ---------------------------------------------------------------------------

def map_sentiment_scores(vocab, opinion_dict, word_freq=None):
    """Map sentiment scores to vocabulary with confidence weighting.

    Adds a 'confidence' column: log(1+freq) / log(1+max_freq).
    Words not in the opinion dictionary get sentiment_score=0.

    Parameters
    ----------
    vocab : list[str]
        Word2Vec vocabulary (ordered).
    opinion_dict : pd.DataFrame
        DataFrame with columns ['word', 'sentiment_score'].
    word_freq : dict, optional
        {word: count}. If provided, computes confidence column.

    Returns
    -------
    pd.DataFrame : Columns ['word', 'sentiment_score', 'confidence'].
    """
    emotion_map = opinion_dict.set_index('word')['sentiment_score'].to_dict()
    df = pd.DataFrame({'word': vocab})
    df['sentiment_score'] = df['word'].map(emotion_map).fillna(0)

    if word_freq is not None:
        max_freq = max(word_freq.values()) if word_freq else 1
        log_max = np.log(1 + max_freq)
        df['confidence'] = df['word'].map(
            lambda w: np.log(1 + word_freq.get(w, 0)) / log_max
        )
    else:
        df['confidence'] = 1.0

    return df


# ---------------------------------------------------------------------------
# Split scored / unscored v2 — raw vectors, no PCA
# ---------------------------------------------------------------------------

def split_scored_unscored(merged_df, vectors):
    """Split into scored and unscored, attaching raw W2V vectors.

    Unlike v1, does NOT apply PCA. Returns the full-dimensional vectors.

    Parameters
    ----------
    merged_df : pd.DataFrame
        DataFrame with ['word', 'sentiment_score', 'confidence'].
    vectors : np.ndarray
        W2V vectors, shape (len(merged_df), vector_size).

    Returns
    -------
    tuple : (df_unscored, df_scored, vecs_unscored, vecs_scored)
        df_unscored/df_scored: DataFrames with word, sentiment_score, confidence.
        vecs_unscored/vecs_scored: np.ndarray of shape (N, vector_size).
    """
    mask_scored = merged_df['sentiment_score'] != 0.0

    df_unscored = merged_df[~mask_scored].reset_index(drop=True)
    df_scored = merged_df[mask_scored].reset_index(drop=True)

    vecs_unscored = vectors[~mask_scored.values]
    vecs_scored = vectors[mask_scored.values]

    return df_unscored, df_scored, vecs_unscored, vecs_scored


# ---------------------------------------------------------------------------
# k-NN cosine sentiment propagation
# ---------------------------------------------------------------------------

def _cosine_similarity_batch(query_vecs, ref_vecs):
    """Compute cosine similarity matrix between query and reference vectors.

    Parameters
    ----------
    query_vecs : np.ndarray, shape (Q, D)
    ref_vecs : np.ndarray, shape (R, D)

    Returns
    -------
    np.ndarray : shape (Q, R), cosine similarities.
    """
    # Normalize to unit vectors
    query_norm = query_vecs / (np.linalg.norm(query_vecs, axis=1, keepdims=True) + 1e-10)
    ref_norm = ref_vecs / (np.linalg.norm(ref_vecs, axis=1, keepdims=True) + 1e-10)
    return query_norm @ ref_norm.T


def propagate_sentiment_knn(vecs_unscored, vecs_scored, scores_scored,
                            confidence_scored=None, k=20, batch_size=5000):
    """Propagate sentiment via k-nearest-neighbor cosine-weighted interpolation.

    For each unscored word u:
        score(u) = sum(sim_i * conf_i * score_i) / sum(sim_i * conf_i)
    where i ranges over the k nearest scored words by cosine similarity,
    and conf_i is the confidence weight of scored word i.

    Uses np.argpartition for efficient O(M) top-k selection per query.

    Parameters
    ----------
    vecs_unscored : np.ndarray, shape (N, D)
        Vectors of unscored words.
    vecs_scored : np.ndarray, shape (M, D)
        Vectors of scored words.
    scores_scored : np.ndarray, shape (M,)
        Sentiment scores of scored words.
    confidence_scored : np.ndarray, shape (M,), optional
        Confidence weights for scored words. Defaults to all 1.0.
    k : int
        Number of nearest neighbors.
    batch_size : int
        Process this many unscored words per batch to limit memory.

    Returns
    -------
    np.ndarray : shape (N,), propagated sentiment scores.
    """
    N = vecs_unscored.shape[0]
    M = vecs_scored.shape[0]
    k = min(k, M)  # Can't have more neighbors than scored words

    if confidence_scored is None:
        confidence_scored = np.ones(M, dtype=np.float64)

    # Pre-weight: score * confidence for each scored word
    weighted_scores = scores_scored * confidence_scored

    # Normalize reference vectors once
    ref_norms = np.linalg.norm(vecs_scored, axis=1, keepdims=True) + 1e-10
    ref_normalized = vecs_scored / ref_norms

    result = np.zeros(N, dtype=np.float64)

    n_batches = (N + batch_size - 1) // batch_size
    for b in range(n_batches):
        start = b * batch_size
        end = min(start + batch_size, N)
        batch_vecs = vecs_unscored[start:end]

        # Cosine similarity: (batch, M)
        query_norms = np.linalg.norm(batch_vecs, axis=1, keepdims=True) + 1e-10
        query_normalized = batch_vecs / query_norms
        sim_matrix = query_normalized @ ref_normalized.T  # (batch, M)

        # For each query, find top-k by argpartition (O(M) per query)
        # argpartition puts the k largest at the end (negated = smallest = largest sim)
        top_k_indices = np.argpartition(-sim_matrix, k, axis=1)[:, :k]  # (batch, k)

        # Gather similarities and compute weighted average
        batch_n = end - start
        for i in range(batch_n):
            idx = top_k_indices[i]
            sims = sim_matrix[i, idx]

            # Clamp negative similarities to 0 (dissimilar words shouldn't contribute)
            sims = np.maximum(sims, 0.0)

            weights = sims * confidence_scored[idx]
            weight_sum = weights.sum()

            if weight_sum > 0:
                result[start + i] = (sims * weighted_scores[idx]).sum() / weight_sum
            else:
                result[start + i] = 0.0

        if (b + 1) % 10 == 0 or (b + 1) == n_batches:
            print(f"    Batch {b+1}/{n_batches} done ({end}/{N} words)")

    return result


# ---------------------------------------------------------------------------
# Convenience: full propagation pipeline
# ---------------------------------------------------------------------------

def build_sentiment_dictionary(model, opinion_dict, word_freq=None,
                                  k=20, batch_size=5000):
    """Full pipeline: map scores, split, propagate, merge.

    Parameters
    ----------
    model : Word2Vec
        Trained gensim Word2Vec model.
    opinion_dict : pd.DataFrame
        Opinion dictionary with ['word', 'sentiment_score'].
    word_freq : dict, optional
        {word: count} for confidence weighting.
    k : int
        Number of nearest neighbors.
    batch_size : int
        Batch size for k-NN propagation.

    Returns
    -------
    pd.DataFrame : Complete dictionary with ['word', 'sentiment_score'].
    pd.DataFrame : Computed (previously unscored) subset.
    """
    vocab = model.wv.index_to_key
    vectors = model.wv[vocab]

    print("  Mapping sentiment scores...")
    mapped_df = map_sentiment_scores(vocab, opinion_dict, word_freq)

    print("  Splitting scored / unscored...")
    df_unscored, df_scored, vecs_unscored, vecs_scored = \
        split_scored_unscored(mapped_df, vectors)

    scored_values = df_scored['sentiment_score'].to_numpy()
    confidence_values = df_scored['confidence'].to_numpy()

    print(f"  Scored words: {len(df_scored)}, Unscored words: {len(df_unscored)}")
    print(f"  Propagating sentiment (k={k}, batches of {batch_size})...")

    propagated = propagate_sentiment_knn(
        vecs_unscored, vecs_scored, scored_values,
        confidence_scored=confidence_values,
        k=k, batch_size=batch_size,
    )
    df_unscored = df_unscored.copy()
    df_unscored['sentiment_score'] = propagated

    # Merge: computed + original opinion dict
    computed = df_unscored[['word', 'sentiment_score']]
    total = pd.concat([computed, opinion_dict[['word', 'sentiment_score']]],
                      ignore_index=True)

    return total, computed

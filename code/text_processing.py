"""Text processing utilities — extracted from multiple notebooks.

Provides functions for:
- Push/comment content extraction and URL filtering
- Jieba word segmentation
- Word frequency counting
- Post title filtering
- Author-level spam filtering
- Log-weighted word frequency
"""

import re
import math
import string

import jieba
import pandas as pd


# ---------------------------------------------------------------------------
# Push content extraction (from FinalSentimentScore cells 6, 15)
# ---------------------------------------------------------------------------

def extract_push_contents(pushes):
    """Extract Content field from push dicts, filtering out URLs.

    Parameters
    ----------
    pushes : list[dict]
        List of push dictionaries with 'Content' key.

    Returns
    -------
    list[str] : Cleaned push content strings.
    """
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    contents = [
        item['Content']
        for item in pushes
        if 'Content' in item and not re.search(url_pattern, item['Content'])
    ]
    return contents


def segment_with_jieba(sentences):
    """Segment a list of sentences using jieba.

    Parameters
    ----------
    sentences : list[str]
        Raw sentences to segment.

    Returns
    -------
    list[list[str]] : Each sentence as a list of segmented words.
    """
    segmented = []
    for sentence in sentences:
        words = list(jieba.cut(sentence))
        segmented.append(words)
    return segmented


DEFAULT_EXCLUDED_WORDS = [
    '我', '你', ',', '！', '，', '了', '又', '？',
    '的', '是', '就', '要', '在', '都', '有', '嗎', '也', '會',
]


def build_word_frequency(segmented_sentences, excluded_words=None):
    """Count word frequencies from segmented sentences, excluding stopwords.

    Parameters
    ----------
    segmented_sentences : list[list[str]]
        2D list of segmented words.
    excluded_words : list[str], optional
        Words to exclude. Defaults to DEFAULT_EXCLUDED_WORDS.

    Returns
    -------
    dict : {word: count}, sorted by count descending.
    """
    if excluded_words is None:
        excluded_words = DEFAULT_EXCLUDED_WORDS

    word_count_dict = {}
    for sublist in segmented_sentences:
        for word in sublist:
            word_cleaned = word.strip()
            if word_cleaned and word_cleaned not in string.punctuation and word_cleaned not in excluded_words:
                word_count_dict[word_cleaned] = word_count_dict.get(word_cleaned, 0) + 1

    # Sort by count descending
    sorted_word_count = dict(sorted(word_count_dict.items(), key=lambda x: x[1], reverse=True))
    return sorted_word_count


# ---------------------------------------------------------------------------
# Post title filtering (from PttStock_v2 cell 9)
# ---------------------------------------------------------------------------

def filter_posts_by_title(df, include_words=None, exclude_words=None):
    """Filter a DataFrame of posts by title keywords.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a 'Title' column.
    include_words : list[str], optional
        If provided, keep only rows whose Title contains ALL of these words.
    exclude_words : list[str], optional
        Remove rows whose Title contains ANY of these words.

    Returns
    -------
    pd.DataFrame : Filtered DataFrame with reset index.
    """
    result = df.copy()
    if exclude_words:
        result = result[~result['Title'].str.contains('|'.join(exclude_words))]
    if include_words:
        for word in include_words:
            result = result[result['Title'].str.contains(word)]
    return result.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Author-level spam filtering (from 留言處理(取log) cells 0, 2)
# ---------------------------------------------------------------------------

def filter_comments_by_author(comments, threshold_ratio=0.01, min_threshold=2):
    """Cap per-author comment count to prevent spam dominance.

    If total comments >= 1/threshold_ratio, the cap is
    total_comments * threshold_ratio. Otherwise uses min_threshold.

    Parameters
    ----------
    comments : list[dict]
        Each dict must have an 'author' (or 'Userid') key.
    threshold_ratio : float
        Maximum fraction of total comments per author.
    min_threshold : int
        Minimum threshold when total is small.

    Returns
    -------
    list[dict] : Filtered comments.
    """
    total_comments = len(comments)
    min_count_for_ratio = int(1 / threshold_ratio) if threshold_ratio > 0 else 100

    if total_comments >= min_count_for_ratio:
        threshold = int(total_comments * threshold_ratio)
    else:
        threshold = min_threshold

    author_comment_count = {}
    filtered_comments = []

    for comment in comments:
        author = comment.get("author") or comment.get("Userid", "unknown")
        author_comment_count[author] = author_comment_count.get(author, 0) + 1

        if author_comment_count[author] <= threshold:
            filtered_comments.append(comment)

    return filtered_comments


# ---------------------------------------------------------------------------
# Log-weighted word frequency (from 留言處理(取log) cell 4)
# ---------------------------------------------------------------------------

def log_weight(word_count):
    """Compute log-weighted value for a word count.

    Uses log(word_count + 1) to avoid log(0).

    Parameters
    ----------
    word_count : int
        Number of times a word appears.

    Returns
    -------
    float : Log-weighted value.
    """
    return math.log(word_count + 1)

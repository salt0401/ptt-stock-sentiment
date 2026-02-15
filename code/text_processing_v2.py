"""Improved text processing utilities (v2).

Improvements over text_processing.py:
- File-based stopword loading (~300 words vs 20 hardcoded)
- Custom jieba dictionary for PTT stock compound terms
- Push tag extraction (Tag + Userid + Ipdatetime alongside Content)
- Token filtering: removes single chars, pure numbers, len<2
- Frequency counting with expanded stopword set
"""

import re
import string
from pathlib import Path

import jieba

from config import DATA_DIR


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

STOPWORDS_DIR = DATA_DIR / "stopwords"
STOPWORDS_PATH = STOPWORDS_DIR / "ptt_stopwords.txt"
CUSTOM_DICT_PATH = STOPWORDS_DIR / "ptt_custom_dict.txt"


# ---------------------------------------------------------------------------
# Stopword loading
# ---------------------------------------------------------------------------

def load_stopwords(path=None):
    """Load stopwords from a text file (one word per line, # for comments).

    Parameters
    ----------
    path : str or Path, optional
        Path to stopword file. Defaults to STOPWORDS_PATH.

    Returns
    -------
    set[str] : Set of stopwords.
    """
    if path is None:
        path = STOPWORDS_PATH
    path = Path(path)
    if not path.exists():
        print(f"  [WARN] Stopword file not found: {path}, using empty set")
        return set()

    words = set()
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                words.add(line)
    return words


# ---------------------------------------------------------------------------
# Jieba custom dictionary
# ---------------------------------------------------------------------------

_jieba_custom_loaded = False


def init_jieba_custom_dict(path=None):
    """Load custom jieba dictionary for PTT stock terms.

    Safe to call multiple times; only loads once.

    Parameters
    ----------
    path : str or Path, optional
        Path to jieba user dictionary. Defaults to CUSTOM_DICT_PATH.
    """
    global _jieba_custom_loaded
    if _jieba_custom_loaded:
        return
    if path is None:
        path = CUSTOM_DICT_PATH
    path = Path(path)
    if path.exists():
        jieba.load_userdict(str(path))
        print(f"  Loaded custom jieba dictionary: {path}")
    else:
        print(f"  [WARN] Custom dict not found: {path}")
    _jieba_custom_loaded = True


# ---------------------------------------------------------------------------
# Push content extraction with tags
# ---------------------------------------------------------------------------

_URL_PATTERN = re.compile(
    r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
)


def extract_push_contents_with_tags(pushes):
    """Extract push Content along with Tag, Userid, and Ipdatetime.

    Filters out pushes whose content is a URL.

    Parameters
    ----------
    pushes : list[dict]
        Push dicts with keys: Tag, Userid, Content, Ipdatetime.

    Returns
    -------
    list[dict] : Each dict has keys: tag, userid, content, ipdatetime.
        tag values: '推' (push/positive), '噓' (boo/negative), '→' (neutral arrow).
    """
    results = []
    for item in pushes:
        content = item.get("Content", "")
        if not content or _URL_PATTERN.search(content):
            continue
        results.append({
            "tag": item.get("Tag", "→").strip(),
            "userid": item.get("Userid", ""),
            "content": content,
            "ipdatetime": item.get("Ipdatetime", ""),
        })
    return results


def extract_push_contents(pushes):
    """Extract only Content field from pushes, filtering URLs (v1 compatible).

    Parameters
    ----------
    pushes : list[dict]

    Returns
    -------
    list[str] : Cleaned content strings.
    """
    return [
        item["content"]
        for item in extract_push_contents_with_tags(pushes)
    ]


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------

def segment_with_jieba_v2(sentences):
    """Segment sentences using jieba with custom dictionary loaded.

    Parameters
    ----------
    sentences : list[str]
        Raw sentences.

    Returns
    -------
    list[list[str]] : Segmented sentences.
    """
    init_jieba_custom_dict()
    segmented = []
    for sentence in sentences:
        words = list(jieba.cut(sentence))
        segmented.append(words)
    return segmented


# ---------------------------------------------------------------------------
# Token filtering
# ---------------------------------------------------------------------------

_PURE_NUMBER_RE = re.compile(r'^[\d.,%+\-]+$')
_PURE_PUNCT_RE = re.compile(r'^[' + re.escape(string.punctuation) + r'。，！？、；：「」『』（）【】《》〈〉""''…─—～]+$')


def filter_tokens(tokens, stopwords=None, min_len=2):
    """Filter a list of tokens, removing noise.

    Removes:
    - Tokens in stopwords set
    - Pure punctuation
    - Pure numbers (including decimals, percentages)
    - Tokens shorter than min_len
    - Whitespace-only tokens

    Parameters
    ----------
    tokens : list[str]
        Word tokens from segmentation.
    stopwords : set[str], optional
        Stopwords to remove.
    min_len : int
        Minimum token length (in characters).

    Returns
    -------
    list[str] : Filtered tokens.
    """
    if stopwords is None:
        stopwords = set()

    filtered = []
    for tok in tokens:
        tok = tok.strip()
        if not tok:
            continue
        if len(tok) < min_len:
            continue
        if tok in stopwords:
            continue
        if _PURE_NUMBER_RE.match(tok):
            continue
        if _PURE_PUNCT_RE.match(tok):
            continue
        filtered.append(tok)
    return filtered


# ---------------------------------------------------------------------------
# Word frequency (v2)
# ---------------------------------------------------------------------------

def build_word_frequency_v2(segmented_sentences, stopwords=None, min_len=2):
    """Count word frequencies with expanded stopword filtering.

    Parameters
    ----------
    segmented_sentences : list[list[str]]
        2D list of segmented words.
    stopwords : set[str], optional
        Stopwords set. If None, loads from default file.
    min_len : int
        Minimum token length.

    Returns
    -------
    dict : {word: count}, sorted by count descending.
    """
    if stopwords is None:
        stopwords = load_stopwords()

    word_count = {}
    for sentence in segmented_sentences:
        for tok in filter_tokens(sentence, stopwords, min_len):
            word_count[tok] = word_count.get(tok, 0) + 1

    return dict(sorted(word_count.items(), key=lambda x: x[1], reverse=True))

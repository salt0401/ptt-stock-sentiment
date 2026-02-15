# PTT Stock Sentiment Analysis & Fama-French Five-Factor Model

Graduation thesis project that builds a **sentiment dictionary from PTT (Taiwan's largest forum) stock board comments** using Word2Vec embeddings and k-nearest-neighbor propagation, then tests whether the resulting sentiment signal has explanatory power in a **Fama-French 5-Factor regression** on Taiwanese stock returns.

## Motivation

Can the collective mood of anonymous forum commenters predict stock market movements? PTT's Stock board generates thousands of comments daily with built-in sentiment labels: users tag each comment as **push** (bullish), **boo** (bearish), or **arrow** (neutral). We exploit this structure to:

1. Build a domain-specific sentiment dictionary from Word2Vec embeddings
2. Validate the dictionary against push/boo/arrow ground-truth labels
3. Test whether aggregated daily sentiment adds explanatory power beyond the standard Fama-French 5-factor model

## Key Results

| Metric | Value |
|--------|-------|
| Dictionary size | ~78K words |
| Comment coverage | 94.1% |
| FF5+SENT p-value | **0.017** (significant at 5%) |
| SENT t-statistic | 2.39 |
| Delta R-squared | +0.000072 |
| Delta AIC | -6.77 |

The sentiment signal is **statistically significant** (p=0.017) in the FF5 regression with Newey-West standard errors.

## Pipeline Architecture

```
PTT Scraper --> Raw Comments (1,415 posts, 1.69M comments)
                    |
                    v
            Text Processing
            (jieba segmentation + stopwords + custom PTT dictionary)
                    |
                    v
            Word2Vec Training
            (Skip-gram, 150D, min_count=5)
                    |
                    v
      Seed Dictionary (opinion_dict.xlsx)
            + Word2Vec vectors
                    |
                    v
          k-NN Sentiment Propagation
          (k=20, cosine similarity, confidence-weighted IDW)
                    |
                    v
         Full Sentiment Dictionary
         (~78K words with sentiment scores)
                    |
              +-----+-----+
              v           v
      Push-Tag Eval    FF5 Regression
      (3-class F1,     (OLS with HAC/
       kappa, etc.)     Newey-West SE)
```

## Directory Structure

```
.
├── code/
│   ├── config.py                  # Centralized path configuration
│   ├── column_map.py              # Chinese -> English column name mappings
│   ├── scraper.py                 # PTT web scraper
│   ├── text_processing.py         # Jieba segmentation, stopwords, token filtering
│   ├── sentiment.py               # W2V training, k-NN cosine propagation
│   ├── five_factor.py             # Fama-French 5-factor computation
│   ├── evaluate_sentiment.py      # Push-tag classification evaluation
│   ├── evaluate_ff5_regression.py # FF5 + Sentiment OLS regression
│   ├── run_scraper.py             # Entry point: scraping pipeline
│   ├── run_sentiment.py           # Entry point: sentiment pipeline
│   └── run_five_factor.py         # Entry point: five-factor pipeline
├── data/
│   ├── comments/raw/              # Scraped PTT data (not in repo, ~300MB)
│   ├── financial/                 # Stock prices, FF factor data
│   │   └── aggregated_table/      # total_table.csv (FF5 + returns + sentiment)
│   ├── stopwords/                 # Stopword list + custom jieba dictionary
│   ├── nlp_models/                # CKIPTagger models (not in repo, ~2.5GB)
│   └── external/                  # External datasets (not in repo)
├── output/
│   ├── opinion_dict.xlsx          # Seed sentiment dictionary (~5,400 words)
│   ├── total_dictionary.csv       # Full propagated dictionary (~78K words)
│   ├── computed_sentiment_dict.csv# Propagated subset (words not in seed dict)
│   ├── daily_sentiment.csv        # Aggregated daily sentiment scores
│   ├── vocab.json                 # Word2Vec vocabulary
│   └── word_frequency.json        # Word frequency counts
└── README.md
```

## Quick Start

### Prerequisites

```bash
pip install pandas numpy scipy scikit-learn gensim jieba statsmodels openpyxl matplotlib
```

### 1. Scrape PTT stock board posts

```bash
cd code
python run_scraper.py --days 350 --max-posts 1500
```

### 2. Run sentiment pipeline

```bash
python run_sentiment.py
```

This trains Word2Vec, propagates sentiment scores via k-NN, and saves results to `output/`.

### 3. Evaluate sentiment dictionary

```bash
# Push-tag evaluation with threshold grid search
python evaluate_sentiment.py --dict-csv ../output/total_dictionary.csv --grid-search

# FF5 regression
python evaluate_ff5_regression.py --sentiment-csv ../output/daily_sentiment.csv
```

### 4. Calculate Fama-French 5 factors (standalone)

```bash
python run_five_factor.py --year-start 2018 --year-end 2023
```

## Data Requirements

Files not included in the repository (too large):

| File | Size | Description |
|------|------|-------------|
| `data/comments/raw/*.pkl` | ~300MB | Scraped PTT posts (run `run_scraper.py`) |
| `data/nlp_models/` | ~2.5GB | CKIPTagger models (optional) |
| `data/external/` | ~40MB | CSentiPackage external dataset |
| `data/financial/excel_raw/` | ~27MB | Raw monthly stock price Excel files |
| `data/financial/fama-french/` | ~67MB | Fama-French factor computation data |

## Technical Details

### Sentiment Propagation

Given a seed dictionary of ~5,400 labeled words and Word2Vec embeddings for ~78K words:

1. **Embed**: Train Skip-gram Word2Vec (150D, min_count=5, 10 epochs)
2. **Score seeds**: Map seed words to their known sentiment scores, compute confidence = log(1+freq)/log(1+max_freq)
3. **Propagate**: For each unscored word, find k=20 nearest neighbors (cosine similarity) among scored words, compute weighted average: score = sum(sim_i * conf_i * score_i) / sum(sim_i * conf_i)
4. **Batch processing**: Process in batches of 5,000 words with np.argpartition for O(M) top-k selection

### FF5 Regression

```
excess_return_t = a + b1*SMB_t + b2*HML_t + b3*RMW_t + b4*CMA_t + b5*SENT_t + e_t
```

- Dependent variable: `spread` = daily return - risk-free rate
- Standard errors: HAC (Newey-West) with lag = int(4*(T/100)^(2/9))
- N = 1,326 trading days (2018-2023)

## License

This project is part of a graduation thesis. Feel free to use for academic and research purposes.

# PTT Stock Sentiment Analysis & Fama-French Five-Factor Model

Graduation thesis project that builds a **sentiment dictionary from PTT (Taiwan's largest forum) stock board comments**, using Word2Vec embeddings and k-nearest-neighbor propagation, then tests whether the resulting sentiment signal has explanatory power in a **Fama-French 5-Factor regression** on Taiwanese stock returns.

## Motivation

Can the collective mood of anonymous forum commenters predict stock market movements? PTT's Stock board (`/r/Stock`) generates thousands of comments daily with built-in sentiment labels: users tag each comment as **push** (bullish), **boo** (bearish), or **arrow** (neutral). We exploit this structure to:

1. Build a domain-specific sentiment dictionary from Word2Vec embeddings
2. Validate the dictionary against push/boo/arrow ground-truth labels
3. Test whether aggregated daily sentiment adds explanatory power beyond the standard Fama-French 5-factor model

## Key Results

| Metric | v1 (Baseline) | v2 (Improved) |
|--------|---------------|---------------|
| Dictionary coverage | 54.0% | **94.1%** |
| Cohen's kappa | 0.0028 | **0.0069** |
| FF5+SENT p-value | 0.097 | **0.017** |
| Delta R-squared | +0.000022 | **+0.000072** |
| Delta AIC | -0.63 | **-6.77** |

The improved pipeline achieves **statistically significant** sentiment signal (p=0.017) in the FF5 regression, while the baseline pipeline does not cross the 5% threshold.

## Pipeline Architecture

```
PTT Scraper ──► Raw Comments (1,415 posts, 1.69M comments)
                    │
                    ▼
            Text Processing
            (jieba segmentation + stopwords + custom PTT dictionary)
                    │
                    ▼
            Word2Vec Training
            (Skip-gram, 150D, min_count=5)
                    │
                    ▼
      Seed Dictionary (opinion_dict.xlsx)
            + Word2Vec vectors
                    │
                    ▼
          k-NN Sentiment Propagation
          (k=20, cosine similarity, confidence-weighted IDW)
                    │
                    ▼
         Full Sentiment Dictionary
         (~78K words with sentiment scores)
                    │
              ┌─────┴─────┐
              ▼           ▼
      Push-Tag Eval    FF5 Regression
      (3-class F1,     (OLS with HAC/
       kappa, etc.)     Newey-West SE)
```

## v1 vs v2: What Changed

| Component | v1 (Baseline) | v2 (Improved) |
|-----------|---------------|---------------|
| Word2Vec | CBOW, 50D, min_count=1 | **Skip-gram, 150D, min_count=5** |
| Dimensionality | PCA 50D → 4D | **Full 150D (no PCA)** |
| Distance metric | Minkowski p=4 (all words) | **Cosine k-NN (k=20)** |
| Propagation | IDW over all scored words | **Confidence-weighted IDW over k neighbors** |
| Stopwords | 20 hardcoded words | **384 words (file-based)** |
| Custom dict | None | **75 PTT stock compound terms** |
| Token filter | Minimal | **Remove single chars, pure numbers, len<2** |
| Coverage | 54% of comments | **94% of comments** |

## Directory Structure

```
.
├── code/
│   ├── config.py                 # Centralized path configuration
│   ├── column_map.py             # Chinese → English column name mappings
│   ├── scraper.py                # PTT web scraper
│   ├── text_processing.py        # v1: jieba segmentation, word frequency
│   ├── text_processing_v2.py     # v2: improved segmentation + stopwords + filtering
│   ├── sentiment.py              # v1: W2V + PCA + Minkowski IDW
│   ├── sentiment_v2.py           # v2: W2V skip-gram + cosine k-NN propagation
│   ├── five_factor.py            # Fama-French 5-factor computation
│   ├── evaluate_sentiment.py     # Push-tag classification evaluation
│   ├── evaluate_ff5_regression.py# FF5 + Sentiment OLS regression
│   ├── run_scraper.py            # Entry point: scraping pipeline
│   ├── run_sentiment.py          # Entry point: v1 sentiment pipeline
│   ├── run_sentiment_v2.py       # Entry point: v2 sentiment pipeline
│   ├── run_five_factor.py        # Entry point: five-factor pipeline
│   ├── run_comparison.py         # Master: v1 vs v2 side-by-side comparison
│   └── plot_comparison.py        # Generate visualization plots
├── data/
│   ├── comments/raw/             # Scraped PTT data (not in repo, ~300MB)
│   ├── financial/                # Stock prices, FF factor data
│   │   └── aggregated_table/     # total_table.csv (FF5 + returns + sentiment)
│   ├── stopwords/                # Stopword list + custom jieba dictionary
│   ├── nlp_models/               # CKIPTagger models (not in repo, ~2.5GB)
│   └── external/                 # External datasets (not in repo)
├── output/
│   ├── opinion_dict.xlsx         # Seed sentiment dictionary
│   ├── computed_sentiment_dict.csv # v1 propagated dictionary
│   ├── v2/                       # v2 pipeline outputs
│   │   ├── total_dictionary_v2.csv
│   │   ├── daily_sentiment_v2.csv
│   │   └── ...
│   └── comparison/               # v1 vs v2 comparison results
│       ├── figures/              # 7 visualization plots
│       ├── push_tag_comparison.json
│       └── ff5_regression_comparison.json
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

### 2. Run v2 sentiment pipeline (recommended)

```bash
python run_sentiment_v2.py
```

This trains Word2Vec, propagates sentiment scores via k-NN, and saves results to `output/v2/`.

### 3. Run v1 vs v2 comparison

```bash
# Full comparison (push-tag evaluation + FF5 regression)
python run_comparison.py --skip-training

# Skip push-tag eval (faster, regression only)
python run_comparison.py --skip-training --no-push-eval
```

### 4. Generate visualization plots

```bash
python plot_comparison.py
```

Produces 7 figures in `output/comparison/figures/`:
1. Dictionary score distributions
2. Push-tag mean scores by class
3. Confusion matrices (row-normalized)
4. FF5 regression comparison (Delta R², t-stat, Delta AIC)
5. Daily sentiment time series vs market return
6. Summary metrics dashboard
7. Propagated score distributions (v1 collapse vs v2 spread)

### 5. Calculate Fama-French 5 factors (standalone)

```bash
python run_five_factor.py --year-start 2018 --year-end 2023
```

## Data Requirements

Files not included in the repository (too large):

| File | Size | Description |
|------|------|-------------|
| `data/comments/raw/*.pkl` | ~300MB | Scraped PTT posts (run `run_scraper.py`) |
| `data/nlp_models/` | ~2.5GB | CKIPTagger models (optional, v1 only) |
| `data/external/` | ~40MB | CSentiPackage external dataset |
| `data/financial/excel_raw/` | ~27MB | Raw monthly stock price Excel files |
| `data/financial/fama-french/` | ~67MB | Fama-French factor computation data |

## Technical Details

### Sentiment Propagation (v2)

Given a seed dictionary of ~5,400 labeled words and Word2Vec embeddings for ~78K words:

1. **Embed**: Train Skip-gram Word2Vec (150D, min_count=5, 10 epochs)
2. **Score seeds**: Map seed words to their known sentiment scores, compute confidence = log(1+freq)/log(1+max_freq)
3. **Propagate**: For each unscored word, find k=20 nearest neighbors (cosine similarity) among scored words, compute weighted average: score = sum(sim_i * conf_i * score_i) / sum(sim_i * conf_i)
4. **Batch processing**: Process in batches of 5,000 words with np.argpartition for O(M) top-k selection

### FF5 Regression

```
excess_return_t = α + β₁SMB_t + β₂HML_t + β₃RMW_t + β₄CMA_t + β₅SENT_t + ε_t
```

- Dependent variable: `spread` = daily return - risk-free rate
- Standard errors: HAC (Newey-West) with lag = int(4*(T/100)^(2/9))
- N = 1,326 trading days (2018-2023)

## License

This project is part of a graduation thesis. Feel free to use for academic and research purposes.

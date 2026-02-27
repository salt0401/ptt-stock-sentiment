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
| Dictionary size | ~131K words |
| Comment coverage | >90% |
| FF5+SENT_lag1 p-value | **0.270** (loss of significance when predicting next day) |
| SENT_lag1 t-statistic | 1.10 |
| Delta R-squared | +0.000011 |
| SENT VIF | 1.0016 (No multicollinearity) |
| **XGBoost Downstream Net Return** | **+50.0%** (Out-of-sample, Sharpe 1.08) |

*Note: After correcting for endogeneity (look-ahead bias) by lagging sentiment ($SENT_{t-1}$), the predictive power in the linear regression model is no longer statistically significant. However, augmenting the linear pipeline with Contextual LLM features (FinBERT), Author/Engagement Weighting, and applying a non-linear XGBoost model captures a highly profitable sentiment signal.*

## Pipeline Architecture

```
PTT Scraper --> Raw Comments (1,415 posts, 1.69M comments)
                    |
                    v
            Text Processing
            (jieba segmentation + stopwords + custom PTT dictionary)
                    |
                    v
            Word2Vec / FastText / FinBERT
            (Embeddings & Contextual Evaluation)
                    |
                    v
           Sentiment Calculation & Weighting
           (k-NN propagation + Author Credibility + Engagement)
                    |
                    v
              Daily Sentiment Features
              (Base, Weighted, Volatility)
                    |
              +-----+-----+
              v           v
        FF5 Regression  XGBoost Trading
        (OLS with HAC)  (Non-linear model)
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
│   ├── evaluate_xgboost_trading.py# ML quantitative trading strategy (Walk-Forward)
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
2. **Score seeds**: Map seed words to their known sentiment scores, compute robust confidence = `1 + log1p(freq)` to prevent high-frequency stopwords from squashing weights.
3. **Propagate**: For each unscored word, find k=20 nearest neighbors (cosine similarity) among scored words, compute weighted average: `score = sum(sim_i * conf_i * score_i) / sum(sim_i * conf_i)`
4. **Batch processing**: Process in batches of 5,000 words with np.argpartition for O(M) top-k selection

### FF5 Regression

```
excess_return_t = a + b1*SMB_t + b2*HML_t + b3*RMW_t + b4*CMA_t + b5*SENT_{t-1} + e_t
```

- Dependent variable: `spread` = daily return - risk-free rate
- **Endogeneity Fix**: Sentiment is lagged by 1 day (`SENT_{t-1}`) to test true predictive power rather than day-of correlation.
- **Multicollinearity Checks**: Variance Inflation Factor (VIF) is calculated to ensure the sentiment signal is independent of Fama-French factors.
- Standard errors: HAC (Newey-West) with lag = `int(4*(T/100)^(2/9))`
- N = 1,325 trading days (2018-2023)

### Advanced Machine Learning Trading (XGBoost)

Building upon the linear regression, the `evaluate_xgboost_trading.py` script replaces simple linear models with a robust quantitative strategy utilizing rich feature engineering:
1. **Multi-Feature Sentiment:** Incorporates baseline Word2Vec dictionary scores alongside Contextual classification scores (FinBERT).
2. **Metadata Weighting:** Applies an expanding-window frequency count to weight Author Credibility (prioritizing experienced users without look-ahead bias) and scales by post Engagement (push/reply counts).
3. **Rolling Window Validation:** Uses a strictly out-of-sample Walk-Forward architecture (1-Year Train -> 1-Quarter Validation -> 1-Quarter Test).
4. **State-Machine Execution:** Adopts an asymmetric Hysteresis filter with a calibrated probability boundary algorithm (e.g., [0.56 / 0.44]). Factoring in 0.2% one-way friction costs, the model secured a **50.0% net out-of-sample return (Sharpe 1.08)**. Feature importance analysis confirms that the Author/Engagement weighted sentiment metrics and FinBERT contextual metrics are primary drivers of the model's predictive splits.

## License

This project is part of a graduation thesis. Feel free to use for academic and research purposes.

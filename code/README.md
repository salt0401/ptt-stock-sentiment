# Code Modules

## Entry Points

| Script | Description | Usage |
|---|---|---|
| `run_scraper.py` | Scrape PTT Stock board posts | `python run_scraper.py --days 350` |
| `run_sentiment.py` | Full sentiment analysis pipeline | `python run_sentiment.py` |
| `run_five_factor.py` | Fama-French 5-factor computation | `python run_five_factor.py` |

All entry points support `--help` for full argument details.

## Library Modules

| Module | Description |
|---|---|
| `config.py` | Centralized path configuration (all file paths defined here) |
| `column_map.py` | Chinese-to-English column name mappings for DataFrames |
| `scraper.py` | `PTTScraper` class for scraping PTT bulletin board |
| `text_processing.py` | Jieba segmentation, URL filtering, word frequency, spam filtering |
| `sentiment.py` | Word2Vec training, PCA reduction, distance-weighted sentiment propagation |
| `five_factor.py` | Stock data loading, grouping (size/PB/ROE/asset growth), factor calculation |

## Module Dependency Diagram

```
run_scraper.py
  ├── config.py
  ├── scraper.py
  └── text_processing.py

run_sentiment.py
  ├── config.py
  ├── text_processing.py
  └── sentiment.py
       └── column_map.py

run_five_factor.py
  ├── config.py
  └── five_factor.py
       └── column_map.py
```

## Archive

The `archive/` subdirectory contains old notebook versions that were used during development:
- `PttStock_v0.ipynb`, `PttStock_v1.ipynb` -- early scraper prototypes
- `W2VTest.ipynb` -- Word2Vec experiments
- `five_factor_old.ipynb` -- original five-factor notebook

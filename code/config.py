"""Centralized path configuration for the NLP project."""

from pathlib import Path

# Project root: one level up from code/
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Data directories ---
DATA_DIR = PROJECT_ROOT / "data"
NLP_MODELS_DIR = DATA_DIR / "nlp_models"
FINANCIAL_DIR = DATA_DIR / "financial"
COMMENTS_DIR = DATA_DIR / "comments"
COMMENTS_RAW_DIR = COMMENTS_DIR / "raw"

# --- Output directory ---
OUTPUT_DIR = PROJECT_ROOT / "output"

# --- External data ---
EXTERNAL_DIR = DATA_DIR / "external"
CSENTI_DIR = EXTERNAL_DIR / "csenti_package"

# --- Specific files ---
OPINION_DICT_PATH = OUTPUT_DIR / "opinion_dict.xlsx"

# Financial data
STOCK_PRICE_DIR = FINANCIAL_DIR  # monthly CSV files like 2018-1.csv
ROE_DATA_PATH = FINANCIAL_DIR / "market_daily_price.csv"
ASSET_DATA_PATH = FINANCIAL_DIR  # total_assets.csv

# Pickle files (scraped data)
PICKLE_AFTER_PATH = COMMENTS_RAW_DIR / "after_market.pkl"
PICKLE_DURING_PATH = COMMENTS_RAW_DIR / "during_market.pkl"

# Fama-French output
FAMA_FRENCH_OUTPUT_DIR = FINANCIAL_DIR / "fama-french"

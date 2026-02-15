"""Centralized Chinese → English column name mappings.

All modules import from here to ensure consistency when renaming
DataFrame columns after loading Chinese-header CSV/Excel files.
"""

# Stock price columns (from monthly CSVs like 2018-1.csv)
STOCK_COLUMNS = {
    '證券代碼': 'stock_id',
    '年月日': 'date',
    '收盤價(元)': 'close_price',
    '流通在外股數(千股)': 'shares_outstanding_k',
    '股價淨值比-TEJ': 'pb_ratio',
    '日報酬率 %': 'daily_return_pct',
}

# ROE data columns
ROE_COLUMNS = {
    '公司': 'stock_id',
    '年月': 'date_str',
    '月份': 'month',
    '營業費用': 'operating_expenses',
    '營業毛利': 'gross_profit',
    '營業成本': 'cost_of_revenue',
    '營業收入淨額': 'net_revenue',
}

# Asset data columns
ASSET_COLUMNS = {
    '公司': 'stock_id',
    '年月': 'date_str',
    '月份': 'month',
    '資產總額': 'total_assets',
}

# Sentiment dictionary columns
SENTIMENT_COLUMNS = {
    '情緒字詞': 'word',
    '情緒分數': 'sentiment_score',
}

# Derived/computed column names (used in five_factor.py)
DERIVED_COLUMNS = {
    '市值': 'market_cap',
    '市值分組': 'size_group',
    '股價淨值比分組': 'pb_group',
    '組別': 'group',
    '年份': 'year',
    '月份': 'month',
    '月份分組': 'quarter_group',
    'ROE分組': 'roe_group',
    '市值xROE組別': 'size_roe_group',
    '資產成長率': 'asset_growth_rate',
    '資產成長率分組': 'asset_growth_group',
    '市值x資產成長率組別': 'size_asset_growth_group',
    '資產總額': 'total_assets',
    '資產總額_上個月': 'total_assets_prev',
}

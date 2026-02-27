# PTT 股版情緒分析與 Fama-French 五因子模型 (PTT Stock Sentiment Analysis & Fama-French 五因子模型)

這是一個專案，探討能否利用 Word2Vec 詞向量與 k-近鄰 (k-NN) 演算法，從 **PTT（台灣最大匿名論壇）股版留言中建立專屬的情緒字典**，並進一步測試這個情緒訊號加入**Fama-French 五因子迴歸模型**後，是否對台灣股市報酬率具備解釋能力。

## 研究動機 (Motivation)

匿名論壇鄉民的集體情緒能夠預測股市走向嗎？PTT 股版每天會產生上萬則留言，並內建了天然的情緒標籤：使用者會將每一則留言標記為 **推 (偏多)**、**噓 (偏空)**，或是 **→ (中立)**。我們利用這樣的推文結構來達成以下目標：

1. 透過 Word2Vec 詞向量技術建立專屬的台股情緒字典。
2. 以推/噓/→的真實標籤來驗證字典的準確度。
3. 測試將每日情緒聚合後，能否在標準的 Fama-French 五因子模型之外提供額外的解釋力。

## 關鍵成果 (Key Results)

| 指標 (Metric) | 數值 (Value) |
|--------|-------|
| 字典規模 (Dictionary size) | 約 13.1 萬詞 |
| 留言覆蓋率 (Comment coverage) | >90% |
| FF5+SENT_lag1 p-value | **0.270** (在預測隔天報酬時失去統計顯著性) |
| SENT_lag1 t-statistic | 1.10 |
| Delta R-squared | +0.000011 |
| SENT VIF | 1.0016 (無共線性問題) |
| **XGBoost 量化交易淨報酬** | **+50.0%** (純樣本外預測, Sharpe 1.08) |

*註：在將情緒指標向後平移一天（$SENT_{t-1}$）以修正內生性（前視偏差）問題後，線性迴歸模型中的預測力便不再顯著。然而，若在線性架構外擴充結合上下文 LLM 特徵（FinBERT）、作者公信力與熱度加權，並應用非線性的 XGBoost 機器學習模型，則能捕捉到具備高度獲利能力的情緒訊號。*

## 系統架構 (Pipeline Architecture)

```
PTT 爬蟲 --> 原始留言 (1,415 篇文章, 169 萬則留言)
                    |
                    v
                 文本處理
         (結巴斷詞 + 停用詞 + PTT專屬常用詞典)
                    |
                    v
          Word2Vec / FastText / FinBERT
            (詞向量嵌入與上下文評估)
                    |
                    v
              情緒計算與權重調整
      (k-NN 情緒傳遞 + 作者公信力 + 文章互動量)
                    |
                    v
               每日情緒特徵
           (基礎、加權、波動度等)
                    |
              +-----+-----+
              v           v
       FF5 五因子迴歸    XGBoost 機器學習量化交易
      (OLS with HAC)    (非線性模型)
```

## 目錄結構 (Directory Structure)

```
.
├── code/
│   ├── config.py                  # 統一的路徑與參數設定
│   ├── column_map.py              # 中英欄位名稱對照表
│   ├── scraper.py                 # PTT 網頁爬蟲
│   ├── text_processing.py         # 結巴斷詞、停用詞與過濾處理
│   ├── sentiment.py               # W2V 訓練與 k-NN 餘弦相似度情緒傳遞
│   ├── five_factor.py             # Fama-French 五因子與報酬運算
│   ├── evaluate_sentiment.py      # 模型推噓文分類準確率評估
│   ├── evaluate_ff5_regression.py # FF5 + 情緒因子 OLS 迴歸檢定
│   ├── evaluate_xgboost_trading.py# 機器學習量化交易策略 (滾動視窗驗證)
│   ├── run_scraper.py             # 啟動腳本：執行爬蟲
│   ├── run_sentiment.py           # 啟動腳本：執行情緒字典管道
│   └── run_five_factor.py         # 啟動腳本：執行五因子資料處理
├── data/
│   ├── comments/raw/              # 爬取的 PTT 原始資料 (未放上 repo, 約300MB)
│   ├── financial/                 # 股價與五因子相關財務數據
│   │   └── aggregated_table/      # total_table.csv (整合 FF5, 報酬率, 每日情緒)
│   ├── stopwords/                 # 停用詞表與結巴特製詞典
│   ├── nlp_models/                # CKIPTagger 模型檔 (未放上 repo, 約2.5GB)
│   └── external/                  # 外部參考資料集 (未放上 repo)
├── output/
│   ├── opinion_dict.xlsx          # 初始情緒種子字典 (約 5,400 詞)
│   ├── total_dictionary.csv       # 情緒傳遞後完整字典 (約 7.8 萬詞)
│   ├── computed_sentiment_dict.csv# 傳遞計算後產生的詞彙 (不含種子詞)
│   ├── daily_sentiment.csv        # 聚合計算後的每日情緒分數
│   ├── vocab.json                 # Word2Vec 訓練完的詞彙表
│   └── word_frequency.json        # 詞頻統計紀錄
└── README.md (以及 README_zh-TW.md)
```

## 快速啟動 (Quick Start)

### 環境需求 (Prerequisites)

```bash
pip install pandas numpy scipy scikit-learn gensim jieba statsmodels openpyxl matplotlib xgboost
```

### 1. 爬取 PTT 股版文章

```bash
cd code
python run_scraper.py --days 350 --max-posts 1500
```

### 2. 執行情緒分析管道

```bash
python run_sentiment.py
```

這將訓練 Word2Vec，透過 k-NN 傳遞情緒分數，並將結果儲存於 `output/` 目錄。

### 3. 評估情緒字典

```bash
# 針對不同的閾值進行網格搜尋，評估推噓文預測準確度
python evaluate_sentiment.py --dict-csv ../output/total_dictionary.csv --grid-search

# 執行 FF5 迴歸分析
python evaluate_ff5_regression.py --sentiment-csv ../output/daily_sentiment.csv
```

### 4. 計算 Fama-French 五因子 (獨立運作)

```bash
python run_five_factor.py --year-start 2018 --year-end 2023
```

## 資料集需求 (Data Requirements)

因容量過大未包含在資源庫內的檔案清單：

| 檔案路徑 | 大小 | 說明 |
|------|------|-------------|
| `data/comments/raw/*.pkl` | ~300MB | 爬蟲抓回的 PTT 貼文與推文 (由 `run_scraper.py` 產生) |
| `data/nlp_models/` | ~2.5GB | CKIPTagger 模型檔案 (可選) |
| `data/external/` | ~40MB | CSentiPackage 外部中文情緒詞典數據 |
| `data/financial/excel_raw/` | ~27MB | TEJ 原始月次股價 Excel 素材 |
| `data/financial/fama-french/` | ~67MB | 用於計算 FF5 指標的財務報表資料 |

## 技術細節 (Technical Details)

### 詞彙情緒傳遞 (Sentiment Propagation)

在使用約 5,400 個預先標記情緒的種子詞彙與 7.8 萬個經編碼的 Word2Vec 詞向量模型下，我們採取以下步驟：

1. **詞向量嵌入 (Embed)**：訓練 Skip-gram Word2Vec 模型 (維度 150D, 最低詞頻 min_count=5, 10 個 epochs)。
2. **種子詞給分 (Score seeds)**：將現有的種子詞對應其情緒極性，並計算其信賴強度 (`confidence = 1 + log1p(freq)`)，防止高頻停用詞過度主導權重。
3. **擴大傳遞 (Propagate)**：對於未經給分的單詞，在已知分數詞群內找出前 k=20 個餘弦相似度 (cosine similarity) 最高的鄰居，並使用加權平均計算情緒：`score = sum(sim_i * conf_i * score_i) / sum(sim_i * conf_i)`。
4. **批次處理 (Batch processing)**：以 5,000 個單字為一個批次進行處理，並利用 `np.argpartition` 加速實作 O(M) 次的 Top-K 篩選。

### FF5 線性迴歸 (FF5 Regression)

```
excess_return_t = a + b1*SMB_t + b2*HML_t + b3*RMW_t + b4*CMA_t + b5*SENT_{t-1} + e_t
```

- **依變數 (Dependent variable)**: 每日大盤超額報酬率 (`spread` = daily return - risk-free rate)
- **內生性修正 (Endogeneity Fix)**：情緒因子向後平移一天 (`SENT_{t-1}`)，純粹測試其對於「未來的」市場走向之預測力，而非當日市場行情的落後反映。
- **共線性檢驗 (Multicollinearity Checks)**：計算變異數膨脹因子 (VIF)，確保情緒訊號獨立於其他五個 Fama-French 市場與基本面因子。
- **標準誤 (Standard errors)**: 使用 HAC (Newey-West) 異質變異系數與自我相關穩健標準誤，設 lag 數值為 `int(4*(T/100)^(2/9))`。
- **樣本數 (N)**：總計涵蓋 1,325 個交易日 (2018-2023期間)。

### 進階機器學習量化交易策略 (Advanced Machine Learning Trading, XGBoost)

在線性迴歸的基礎上，`evaluate_xgboost_trading.py` 腳本將單純的線性檢驗升級為使用豐富特徵工程的強健量化交易模型：
1. **多元情緒特徵 (Multi-Feature Sentiment):** 結合基礎 Word2Vec 字典分數與具備上下文判斷能力的 FinBERT 情感分類分數。
2. **Metadata 動態加權 (Metadata Weighting):** 應用滾動擴張窗口 (expanding-window) 紀錄並計算作者發言公信力 (對歷史常客賦予更高權重，同時嚴防前視偏差 Look-ahead bias)，並依據文章的熱度/互動量 (推/噓/回文數) 作為訊號乘數。
3. **嚴格滾動視窗驗證 (Rolling Window Validation):** 採用完全樣本外預測 (Out-of-sample) 的前向滾動架構 (Walk-Forward architecture：歷史1年訓練 -> 歷史1季驗證 -> 未來1季測試)。
4. **狀態機濾網執行 (State-Machine Execution):** 為減少這類高頻波動訊號造成的摩擦成本，模型導入了非對稱的遲滯狀態機 (Hysteresis) 邊界對位演算法 (如：設定為 `[0.56 / 0.44]`)。即使扣除嚴格的 0.2% 單邊摩擦手續與滑價成本，本策略依然能在純樣本外期間獲得高達 **50.0% 的累積淨報酬 (Sharpe 1.08)**。模型樹狀決策特徵重要性 (Feature Importance) 驗證了作者與互動熱度加權後的情緒指標、以及 FinBERT 指標是主導此預測切割的最主要驅動力。

# 方向A：另类数据（新闻情绪因子）人工介入执行指南

> **文档版本**: 1.0  
> **创建时间**: 2026-04-29  
> **优先级**: ⭐⭐⭐⭐⭐（最高）  
> **目标**: 突破 V229 的 IC ≈ 0.05 瓶颈，解决 2024 年收益 -55.37% 的系统性失效问题

---

## 一、环境准备

### 1.1 Python 依赖安装

在项目根目录下执行以下命令安装所需的 Python 库：

```bash
# 核心依赖（量化项目已有）
pip install pandas numpy loguru sqlalchemy pymysql pyarrow

# 深度学习 & NLP 依赖（新增）
pip install torch>=2.0.0,<2.5.0
pip install transformers>=4.40.0,<4.47.0
pip install sentencepiece>=0.2.0

# 网络请求 & 爬虫依赖（新增）
pip install requests>=2.31.0
pip install aiohttp>=3.9.0  # 异步 HTTP（可选，用于批量请求）
pip install beautifulsoup4>=4.12.0  # HTML 解析（可选）

# 进度条 & 缓存（可选，提升体验）
pip install tqdm>=4.66.0
pip install joblib>=1.3.0
```

### 1.2 验证安装

创建验证脚本 `scripts/verify_env.py`：

```python
"""验证环境是否配置正确"""
import sys

def verify():
    errors = []
    
    # 1. 验证 PyTorch
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
    except ImportError as e:
        errors.append(f"❌ PyTorch 安装失败: {e}")
    
    # 2. 验证 Transformers
    try:
        import transformers
        print(f"✅ Transformers {transformers.__version__}")
    except ImportError as e:
        errors.append(f"❌ Transformers 安装失败: {e}")
    
    # 3. 验证 Requests
    try:
        import requests
        print(f"✅ Requests {requests.__version__}")
    except ImportError as e:
        errors.append(f"❌ Requests 安装失败: {e}")
    
    # 4. 验证 GPU 可用性（可选）
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ CUDA 可用: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️ CUDA 不可用，将使用 CPU 推理（速度较慢但可行）")
    except:
        pass
    
    # 5. 验证项目依赖
    try:
        import polars
        print(f"✅ Polars {polars.__version__}")
    except ImportError:
        print("⚠️ Polars 未安装（如项目使用 pandas 可忽略）")
    
    try:
        import loguru
        print(f"✅ Loguru {loguru.__version__}")
    except ImportError:
        errors.append("❌ Loguru 安装失败")
    
    if errors:
        print("\n" + "=" * 50)
        for err in errors:
            print(err)
        sys.exit(1)
    else:
        print("\n✅ 所有依赖验证通过！")

if __name__ == "__main__":
    verify()
```

运行验证：
```bash
python scripts/verify_env.py
```

### 1.3 模型缓存目录

FinBERT 模型首次下载约 **400MB**，建议提前下载到本地缓存：

```bash
# 设置 HuggingFace 缓存目录（Windows）
set HF_HOME=D:\PythonProject\Quantitative-Trading\models\hf_cache

# Linux/Mac 用户可设置
export HF_HOME=/path/to/your/models
```

---

## 二、数据获取

### 2.1 数据源选择

| 数据源 | 数据类型 | 获取方式 | 成本 | 推荐度 |
|--------|---------|---------|------|--------|
| **新浪财经** | 新闻标题/摘要 | REST API（公开接口） | 免费 | ⭐⭐⭐⭐⭐ |
| **东方财富** | 分析师评级 | REST API（公开接口） | 免费 | ⭐⭐⭐⭐ |
| **雪球** | 社交媒体情绪 | 爬虫/第三方服务 | 免费/付费 | ⭐⭐⭐ |
| **股吧** | 散户讨论情绪 | 爬虫 | 免费 | ⭐⭐ |

### 2.2 新浪财经新闻数据获取

#### 2.2.1 API 接口说明

新浪财经新闻 API（公开接口）：
```
https://feed.mix.sina.com.cn/api/roll/get?catid={category}&pageid={pageid}&num={num}
```

**参数说明**：
| 参数 | 值 | 说明 |
|------|-----|------|
| catid | 73（股票新闻）| 新闻分类 ID |
| pageid | 1, 2, 3... | 页码 |
| num | 50 | 每页数量（最大 50） |

#### 2.2.2 新闻数据获取代码

创建 `src/data_sources/sina_news_fetcher.py`：

```python
"""
新浪财经新闻数据获取器
====================
获取每日股票相关新闻，用于后续情绪分析
"""

import requests
import time
import json
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
import pandas as pd


class SinaNewsFetcher:
    """新浪财经新闻数据获取器"""
    
    BASE_URL = "https://feed.mix.sina.com.cn/api/roll/get"
    
    # 请求间隔（秒），避免触发反爬
    REQUEST_INTERVAL = 0.5
    MAX_RETRIES = 3
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Accept": "application/json",
            "Referer": "https://finance.sina.com.cn",
        })
        logger.info("[SinaNewsFetcher] Initialized")
    
    def fetch_news_by_date(
        self,
        date: str,
        max_pages: int = 10,
        num_per_page: int = 50
    ) -> pd.DataFrame:
        """
        获取指定日期的新闻列表
        
        Args:
            date: 日期字符串，格式 'YYYY-MM-DD'
            max_pages: 最大页数
            num_per_page: 每页数量
            
        Returns:
            DataFrame with columns:
            - news_id: 新闻 ID
            - title: 新闻标题
            - summary: 新闻摘要
            - content: 新闻正文（部分接口提供）
            - ctime: 发布时间
            - url: 新闻链接
            - tags: 标签（可能包含股票代码）
        """
        all_news = []
        
        for page in range(1, max_pages + 1):
            params = {
                "catid": "73",  # 股票新闻
                "pageid": page,
                "num": num_per_page,
            }
            
            success = False
            for retry in range(self.MAX_RETRIES):
                try:
                    resp = self.session.get(
                        self.BASE_URL,
                        params=params,
                        timeout=10
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    
                    if "result" not in data or "data" not in data["result"]:
                        logger.warning(f"[SinaNewsFetcher] No data at page {page}")
                        break
                    
                    news_list = data["result"]["data"]
                    if not news_list:
                        break
                    
                    for item in news_list:
                        # 提取股票代码（从标题或标签中）
                        stock_codes = self._extract_stock_codes(item.get("title", ""))
                        
                        all_news.append({
                            "news_id": item.get("docid", ""),
                            "title": item.get("title", ""),
                            "summary": item.get("summary", ""),
                            "content": item.get("content", ""),
                            "ctime": item.get("ctime", ""),
                            "url": item.get("url", ""),
                            "tags": item.get("keywords", ""),
                            "stock_codes": stock_codes,
                            "fetch_date": date,
                        })
                    
                    success = True
                    logger.info(f"[SinaNewsFetcher] Page {page}: {len(news_list)} news fetched")
                    break
                    
                except requests.RequestException as e:
                    logger.warning(f"[SinaNewsFetcher] Request failed (retry {retry+1}): {e}")
                    time.sleep(2 ** retry)
            
            if not success:
                logger.error(f"[SinaNewsFetcher] Failed to fetch page {page} after {self.MAX_RETRIES} retries")
                break
            
            if len(all_news) > 0 and len(all_news) % 100 == 0:
                logger.info(f"[SinaNewsFetcher] Total news: {len(all_news)}")
            
            time.sleep(self.REQUEST_INTERVAL)
        
        df = pd.DataFrame(all_news)
        logger.info(f"[SinaNewsFetcher] Date {date}: {len(df)} news total")
        return df
    
    def _extract_stock_codes(self, text: str) -> List[str]:
        """
        从文本中提取股票代码（简单正则匹配）
        
        A 股代码格式：
        - 沪市: 60XXXX, 68XXXX（科创板）
        - 深市: 00XXXX, 30XXXX（创业板）
        """
        import re
        pattern = r'(?:[6][0356789]\d{4}|[03]\d{5})'
        matches = re.findall(pattern, text)
        return list(set(matches))
    
    def fetch_news_batch(
        self,
        start_date: str,
        end_date: str,
        max_pages_per_day: int = 5
    ) -> pd.DataFrame:
        """
        批量获取日期范围内的新闻
        
        Args:
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            max_pages_per_day: 每天最大页数
            
        Returns:
            合并后的 DataFrame
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        all_dfs = []
        current = start
        
        while current <= end:
            date_str = current.strftime("%Y-%m-%d")
            logger.info(f"[SinaNewsFetcher] Fetching news for {date_str}...")
            
            df = self.fetch_news_by_date(date_str, max_pages=max_pages_per_day)
            if not df.empty:
                all_dfs.append(df)
            
            current += timedelta(days=1)
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()


def get_news_fetcher() -> SinaNewsFetcher:
    """获取新闻获取器实例"""
    return SinaNewsFetcher()


if __name__ == "__main__":
    # 测试：获取最近一天的新闻
    fetcher = get_news_fetcher()
    df = fetcher.fetch_news_by_date("2024-01-15", max_pages=3)
    print(f"Fetched {len(df)} news articles")
    if not df.empty:
        print(df.head())
        df.to_csv("test_news.csv", index=False, encoding="utf-8-sig")
        print("Saved to test_news.csv")
```

### 2.3 东方财富分析师评级数据获取

创建 `src/data_sources/eastmoney_rating_fetcher.py`：

```python
"""
东方财富分析师评级数据获取器
===========================
获取分析师对股票的评级变化，用于构建分析师预期因子
"""

import requests
import time
from typing import Optional
from loguru import logger
import pandas as pd
from datetime import datetime


class EastMoneyRatingFetcher:
    """东方财富分析师评级获取器"""
    
    # 东方财富个股研报 API
    BASE_URL = "https://datacenter.eastmoney.com/securities/api/data/get"
    
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Referer": "https://data.eastmoney.com",
        })
        logger.info("[EastMoneyRatingFetcher] Initialized")
    
    def fetch_rating_changes(
        self,
        symbol: str,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        获取指定股票在日期范围内的评级变化
        
        Args:
            symbol: 股票代码（如 '600519'）
            start_date: 开始日期 'YYYY-MM-DD'
            end_date: 结束日期 'YYYY-MM-DD'
            
        Returns:
            DataFrame with columns:
            - report_date: 研报日期
            - symbol: 股票代码
            - rating: 评级（买入/增持/中性/减持/卖出）
            - target_price: 目标价
            - eps_forecast: EPS 预测
            - institution: 机构名称
        """
        # 评级映射
        rating_map = {
            "买入": 5,
            "增持": 4,
            "中性": 3,
            "减持": 2,
            "卖出": 1,
        }
        
        params = {
            "type": "RPT_CUSTOM_STOCK_RESEARCH_NEW",
            "sty": "ORG_CODE,ORG_NAME,STOCK_CODE,STOCK_NAME,REPORT_DATE,RATING,TARGET_PRICE,TARGET_PRICE_MAX,TARGET_PRICE_MIN",
            "filter": f"""(STOCK_CODE="{symbol}")(REPORT_DATE>='{start_date}')(REPORT_DATE<='{end_date}')""",
            "sr": "-1",
            "st": "REPORT_DATE",
            "ps": "100",
            "p": "1",
        }
        
        all_data = []
        
        try:
            resp = self.session.get(self.BASE_URL, params=params, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            
            if "result" not in data or "data" not in data["result"]:
                logger.info(f"[EastMoneyRating] No rating data for {symbol}")
                return pd.DataFrame()
            
            for item in data["result"]["data"]:
                all_data.append({
                    "report_date": item.get("REPORT_DATE", ""),
                    "symbol": item.get("STOCK_CODE", ""),
                    "stock_name": item.get("STOCK_NAME", ""),
                    "rating": item.get("RATING", ""),
                    "rating_score": rating_map.get(item.get("RATING", ""), 3),
                    "target_price": item.get("TARGET_PRICE", ""),
                    "institution": item.get("ORG_NAME", ""),
                })
            
            logger.info(f"[EastMoneyRating] {symbol}: {len(all_data)} ratings fetched")
            
        except Exception as e:
            logger.error(f"[EastMoneyRating] Error fetching {symbol}: {e}")
        
        return pd.DataFrame(all_data)
    
    def fetch_bulk_ratings(
        self,
        symbols: list,
        start_date: str,
        end_date: str
    ) -> pd.DataFrame:
        """
        批量获取多只股票的评级变化
        
        Args:
            symbols: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
        """
        all_dfs = []
        
        for i, symbol in enumerate(symbols):
            logger.info(f"[EastMoneyRating] Fetching {i+1}/{len(symbols)}: {symbol}")
            df = self.fetch_rating_changes(symbol, start_date, end_date)
            if not df.empty:
                all_dfs.append(df)
            time.sleep(0.3)  # 请求间隔
        
        if all_dfs:
            return pd.concat(all_dfs, ignore_index=True)
        return pd.DataFrame()


def get_rating_fetcher() -> EastMoneyRatingFetcher:
    """获取评级获取器实例"""
    return EastMoneyRatingFetcher()


if __name__ == "__main__":
    fetcher = get_rating_fetcher()
    df = fetcher.fetch_rating_changes("600519", "2024-01-01", "2024-03-31")
    print(f"Fetched {len(df)} rating changes")
    if not df.empty:
        print(df.head())
```

### 2.4 数据存储方案

获取到的新闻和评级数据需要存储到数据库，以便后续处理：

```sql
-- 新闻情绪数据表
CREATE TABLE IF NOT EXISTS news_sentiment (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    trade_date DATE NOT NULL,
    symbol VARCHAR(20) DEFAULT NULL,
    news_id VARCHAR(50) NOT NULL,
    title VARCHAR(500),
    summary TEXT,
    sentiment_score FLOAT DEFAULT NULL,
    sentiment_label VARCHAR(20) DEFAULT NULL COMMENT 'positive/neutral/negative',
    confidence FLOAT DEFAULT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_trade_date (trade_date),
    INDEX idx_symbol (symbol),
    UNIQUE KEY uk_news_id (news_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- 分析师评级数据表
CREATE TABLE IF NOT EXISTS analyst_rating (
    id BIGINT AUTO_INCREMENT PRIMARY KEY,
    report_date DATE NOT NULL,
    symbol VARCHAR(20) NOT NULL,
    rating VARCHAR(20),
    rating_score INT DEFAULT 3,
    target_price DECIMAL(10, 2) DEFAULT NULL,
    institution VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    INDEX idx_report_date (report_date),
    INDEX idx_symbol (symbol)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;
```

---

## 三、情绪模型加载

### 3.1 模型选择

| 模型 | 语言 | 来源 | 大小 | 推荐度 |
|------|------|------|------|--------|
| **FinBERT** (warrenxin) | 中文金融 | HuggingFace | ~400MB | ⭐⭐⭐⭐⭐ |
| **Chinese-FinBERT** | 中文金融 | HuggingFace | ~400MB | ⭐⭐⭐⭐ |
| **BERT-wwm** | 中文通用 | HuggingFace | ~400MB | ⭐⭐⭐ |

**推荐使用**：`warrenxin/FinBERT`，专门针对中文金融文本微调，情绪分类效果更好。

### 3.2 完整代码示例

创建 `src/sentiment_analyzer.py`：

```python
"""
金融新闻情绪分析器
==================
使用 FinBERT 模型对中文金融新闻进行情绪分类

【输入】
- 新闻标题 + 摘要文本

【输出】
- sentiment_score: 情绪得分 [-1, 1]（-1 极度消极，1 极度积极）
- sentiment_label: 情绪标签 (positive/neutral/negative)
- confidence: 置信度 [0, 1]
"""

from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
import numpy as np
from loguru import logger
from tqdm import tqdm


class FinSentimentAnalyzer:
    """
    金融情绪分析器
    
    【模型选择】
    - warrenxin/FinBERT: 中文金融领域微调
    - 支持三个类别: positive(积极), neutral(中性), negative(消极)
    """
    
    # 默认模型（中文金融）
    DEFAULT_MODEL = "warrenxin/FinBERT"
    
    # 备选模型
    BACKUP_MODELS = [
        "hfl/chinese-roberta-wwm-ext",
        "bert-base-chinese",
    ]
    
    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        batch_size: int = 32,
        max_length: int = 512,
    ):
        """
        初始化情绪分析器
        
        Args:
            model_name: 模型名称（默认使用 FinBERT）
            device: 计算设备 ('cuda', 'cpu', 或 None 自动选择)
            batch_size: 批量推理大小
            max_length: 最大文本长度
        """
        self.batch_size = batch_size
        self.max_length = max_length
        
        # 自动选择设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        logger.info(f"[FinSentimentAnalyzer] Using device: {self.device}")
        
        # 加载模型
        model_name = model_name or self.DEFAULT_MODEL
        logger.info(f"[FinSentimentAnalyzer] Loading model: {model_name}")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            
            # 获取模型配置
            self.config = self.model.config
            self.id2label = self.config.id2label
            self.label2id = self.config.label2id
            
            logger.info(f"[FinSentimentAnalyzer] Model labels: {self.id2label}")
            
        except Exception as e:
            logger.error(f"[FinSentimentAnalyzer] Failed to load {model_name}: {e}")
            raise
        
        # 移动到指定设备
        self.model.to(self.device)
        self.model.eval()  # 设置为评估模式
        
        logger.info("[FinSentimentAnalyzer] Model loaded successfully")
    
    def predict_single(self, text: str) -> Dict[str, float]:
        """
        对单条文本进行情绪预测
        
        Args:
            text: 输入文本
            
        Returns:
            包含以下键的字典:
            - positive: 积极概率
            - neutral: 中性概率
            - negative: 消极概率
            - score: 综合得分 [-1, 1]
            - label: 情绪标签
            - confidence: 置信度
        """
        if not text or not text.strip():
            return {
                "positive": 0.0,
                "neutral": 1.0,
                "negative": 0.0,
                "score": 0.0,
                "label": "neutral",
                "confidence": 0.0,
            }
        
        # Tokenize
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 推理
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=-1)
        
        # 提取概率
        prob_dict = {}
        for label, idx in self.label2id.items():
            prob_dict[label.lower()] = probs[0, idx].item()
        
        # 计算综合得分 [-1, 1]
        positive_prob = prob_dict.get("positive", 0.0)
        negative_prob = prob_dict.get("negative", 0.0)
        neutral_prob = prob_dict.get("neutral", 0.0)
        
        # 得分计算：(positive - negative) / (positive + negative + neutral)
        total = positive_prob + negative_prob + neutral_prob
        if total > 0:
            score = (positive_prob - negative_prob) / total
        else:
            score = 0.0
        
        # 确定标签
        if positive_prob >= negative_prob and positive_prob >= neutral_prob:
            label = "positive"
            confidence = positive_prob
        elif negative_prob >= positive_prob and negative_prob >= neutral_prob:
            label = "negative"
            confidence = negative_prob
        else:
            label = "neutral"
            confidence = neutral_prob
        
        return {
            "positive": positive_prob,
            "neutral": neutral_prob,
            "negative": negative_prob,
            "score": round(score, 4),
            "label": label,
            "confidence": round(confidence, 4),
        }
    
    def predict_batch(
        self,
        texts: List[str],
        show_progress: bool = True
    ) -> List[Dict[str, float]]:
        """
        批量情绪预测
        
        Args:
            texts: 文本列表
            show_progress: 是否显示进度条
            
        Returns:
            预测结果列表
        """
        results = []
        
        # 分批处理
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Processing batches",
            disable=not show_progress,
        ):
            batch_texts = texts[i:i + self.batch_size]
            
            # Tokenize 批次
            inputs = self.tokenizer(
                batch_texts,
                return_tensors="pt",
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # 推理
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
            
            # 处理每条结果
            for j in range(len(batch_texts)):
                prob_dict = {}
                for label, idx in self.label2id.items():
                    prob_dict[label.lower()] = probs[j, idx].item()
                
                positive_prob = prob_dict.get("positive", 0.0)
                negative_prob = prob_dict.get("negative", 0.0)
                neutral_prob = prob_dict.get("neutral", 0.0)
                
                total = positive_prob + negative_prob + neutral_prob
                score = (positive_prob - negative_prob) / total if total > 0 else 0.0
                
                if positive_prob >= negative_prob and positive_prob >= neutral_prob:
                    label = "positive"
                    confidence = positive_prob
                elif negative_prob >= positive_prob and negative_prob >= neutral_prob:
                    label = "negative"
                    confidence = negative_prob
                else:
                    label = "neutral"
                    confidence = neutral_prob
                
                results.append({
                    "positive": round(positive_prob, 4),
                    "neutral": round(neutral_prob, 4),
                    "negative": round(negative_prob, 4),
                    "score": round(score, 4),
                    "label": label,
                    "confidence": round(confidence, 4),
                })
        
        return results
    
    def analyze_news_dataframe(
        self,
        df: pd.DataFrame,
        text_cols: List[str] = ["title", "summary"],
        output_col: str = "sentiment"
    ) -> pd.DataFrame:
        """
        对 DataFrame 中的新闻进行情绪分析
        
        Args:
            df: 包含新闻的 DataFrame
            text_cols: 用于分析的文本列名列表
            output_col: 输出列名前缀
            
        Returns:
            添加了情绪列的 DataFrame
        """
        logger.info(f"[FinSentimentAnalyzer] Analyzing {len(df)} news articles...")
        
        # 合并文本列
        texts = []
        for _, row in df.iterrows():
            text_parts = []
            for col in text_cols:
                if col in row and pd.notna(row[col]) and str(row[col]).strip():
                    text_parts.append(str(row[col]))
            texts.append(" ".join(text_parts))
        
        # 批量预测
        results = self.predict_batch(texts)
        
        # 添加结果到 DataFrame
        df[f"{output_col}_score"] = [r["score"] for r in results]
        df[f"{output_col}_label"] = [r["label"] for r in results]
        df[f"{output_col}_confidence"] = [r["confidence"] for r in results]
        df[f"{output_col}_positive"] = [r["positive"] for r in results]
        df[f"{output_col}_neutral"] = [r["neutral"] for r in results]
        df[f"{output_col}_negative"] = [r["negative"] for r in results]
        
        logger.info(f"[FinSentimentAnalyzer] Analysis complete")
        logger.info(f"  Score distribution: mean={df[f'{output_col}_score'].mean():.4f}, "
                    f"std={df[f'{output_col}_score'].std():.4f}")
        
        return df
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        return {
            "model_name": self.config._name_or_path,
            "num_labels": self.config.num_labels,
            "id2label": self.id2label,
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
        }


def get_sentiment_analyzer(
    model_name: Optional[str] = None,
    device: Optional[str] = None
) -> FinSentimentAnalyzer:
    """获取情绪分析器实例"""
    return FinSentimentAnalyzer(model_name=model_name, device=device)


if __name__ == "__main__":
    # 测试情绪分析器
    analyzer = get_sentiment_analyzer()
    
    # 测试样本
    test_texts = [
        "贵州茅台发布年报，净利润增长20%，股价创新高",  # 积极
        "某公司财务造假被证监会立案调查",  # 消极
        "今日市场成交量维持在合理区间",  # 中性
    ]
    
    print("\n" + "=" * 60)
    print("情绪分析测试结果")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.predict_single(text)
        print(f"\n文本: {text}")
        print(f"得分: {result['score']:.4f}")
        print(f"标签: {result['label']} (置信度: {result['confidence']:.4f})")
        print(f"  积极: {result['positive']:.4f}, 中性: {result['neutral']:.4f}, 消极: {result['negative']:.4f}")
```

### 3.3 性能优化建议

```python
"""
性能优化配置
============
针对不同硬件环境的优化策略
"""

# CPU 环境优化
CPU_CONFIG = {
    "batch_size": 16,        # 较小批次
    "max_length": 256,       # 截断长度
    "num_threads": 4,        # 线程数
}

# GPU 环境优化（推荐）
GPU_CONFIG = {
    "batch_size": 64,        # 较大批次
    "max_length": 512,       # 完整长度
    "use_amp": True,         # 混合精度推理
}

# 推理加速设置
def setup_inference_optimization(use_gpu: bool = True):
    """设置推理优化"""
    import torch
    
    if use_gpu and torch.cuda.is_available():
        # 启用混合精度
        torch.backends.cudnn.benchmark = True
        
        # 设置显存管理
        torch.cuda.empty_cache()
        
        print("✅ GPU 推理已启用")
    else:
        # CPU 优化：设置线程数
        torch.set_num_threads(4)
        print("⚠️ 使用 CPU 推理")
```

---

## 四、因子整合

### 4.1 情绪因子计算逻辑

创建 `src/sentiment_factor.py`：

```python
"""
情绪因子计算模块
================
将每日新闻情绪得分整合为量化因子，用于 Alpha 模型

【因子定义】
1. daily_sentiment: 个股当日平均情绪得分
2. sentiment_ma_5: 5日情绪滚动均值
3. sentiment_ma_20: 20日情绪滚动均值
4. sentiment_change: 情绪变化率 (5日均值 / 20日均值 - 1)
5. sentiment_volatility: 情绪波动率（20日标准差）
"""

from typing import Optional
import pandas as pd
import numpy as np
from loguru import logger

# 内存优化
pd.options.mode.chained_assignment = None


class SentimentFactorCalculator:
    """
    情绪因子计算器
    
    【输入数据格式】
    - trade_date: 交易日期 (int, YYYYMMDD)
    - symbol: 股票代码 (str)
    - sentiment_score: 情绪得分 (float, [-1, 1])
    - sentiment_confidence: 置信度 (float, [0, 1])
    
    【输出因子】
    - sentiment_daily: 当日情绪得分
    - sentiment_ma5: 5日移动平均
    - sentiment_ma20: 20日移动平均
    - sentiment_change: 变化率 (ma5/ma20 - 1)
    - sentiment_vol: 情绪波动率
    """
    
    def __init__(
        self,
        short_window: int = 5,
        long_window: int = 20,
        min_confidence: float = 0.3,
    ):
        """
        初始化计算器
        
        Args:
            short_window: 短期窗口（情绪 MA）
            long_window: 长期窗口（情绪 MA）
            min_confidence: 最低置信度阈值
        """
        self.short_window = short_window
        self.long_window = long_window
        self.min_confidence = min_confidence
        
        logger.info(f"[SentimentFactor] Initialized with "
                    f"short={short_window}, long={long_window}, "
                    f"min_confidence={min_confidence}")
    
    def compute_factors(
        self,
        sentiment_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """
        计算情绪因子
        
        Args:
            sentiment_df: 情绪数据 DataFrame
                Required columns: trade_date, symbol, sentiment_score
            price_df: 价格数据 DataFrame（用于对齐日期和股票）
                Required columns: trade_date, symbol, close
            
        Returns:
            DataFrame with columns:
            - trade_date, symbol
            - sentiment_daily: 日度情绪得分
            - sentiment_ma{short}: 短期情绪均值
            - sentiment_ma{long}: 长期情绪均值
            - sentiment_change: 情绪变化率
            - sentiment_vol: 情绪波动率
        """
        logger.info(f"[SentimentFactor] Computing factors from {len(sentiment_df)} rows...")
        
        # 确保日期格式一致
        df = sentiment_df.copy()
        df['trade_date'] = pd.to_datetime(df['trade_date']).astype(str).str.replace('-', '').astype(int)
        
        # 1. 按日期+股票聚合情绪得分（加权平均，使用置信度作为权重）
        if 'sentiment_confidence' in df.columns:
            # 使用置信度加权
            df['weighted_score'] = df['sentiment_score'] * df['sentiment_confidence']
            df['weight_sum'] = df.groupby(['trade_date', 'symbol'])['sentiment_confidence'].transform('sum')
            df['sentiment_daily'] = df.groupby(['trade_date', 'symbol'])['weighted_score'].transform('sum') / df['weight_sum'].replace(0, np.nan)
        else:
            # 简单平均
            df['sentiment_daily'] = df.groupby(['trade_date', 'symbol'])['sentiment_score'].transform('mean')
        
        # 去重：每个日期+股票只保留一行
        df = df.groupby(['trade_date', 'symbol']).agg({
            'sentiment_daily': 'first',
        }).reset_index()
        
        # 2. 计算滚动情绪均值
        df = df.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 短期 MA
        df[f'sentiment_ma{self.short_window}'] = df.groupby('symbol')['sentiment_daily'].transform(
            lambda x: x.rolling(self.short_window, min_periods=1).mean()
        )
        
        # 长期 MA
        df[f'sentiment_ma{self.long_window}'] = df.groupby('symbol')['sentiment_daily'].transform(
            lambda x: x.rolling(self.long_window, min_periods=5).mean()
        )
        
        # 3. 情绪变化率
        long_col = f'sentiment_ma{self.long_window}'
        short_col = f'sentiment_ma{self.short_window}'
        df['sentiment_change'] = (
            df[short_col] / df[long_col].replace(0, np.nan) - 1
        )
        
        # 4. 情绪波动率
        df['sentiment_vol'] = df.groupby('symbol')['sentiment_daily'].transform(
            lambda x: x.rolling(self.long_window, min_periods=5).std()
        )
        
        # 5. 处理缺失值
        from src.data_healer import heal
        numeric_cols = [
            'sentiment_daily', f'sentiment_ma{self.short_window}',
            f'sentiment_ma{self.long_window}', 'sentiment_change', 'sentiment_vol'
        ]
        df = heal(df, numeric_cols=numeric_cols)
        
        # 6. 选择输出列
        output_cols = ['trade_date', 'symbol', 'sentiment_daily',
                       f'sentiment_ma{self.short_window}', f'sentiment_ma{self.long_window}',
                       'sentiment_change', 'sentiment_vol']
        output_cols = [c for c in output_cols if c in df.columns]
        
        result = df[output_cols].copy()
        logger.info(f"[SentimentFactor] Factors computed: {len(result)} rows")
        
        return result
    
    def merge_with_price_data(
        self,
        sentiment_factors: pd.DataFrame,
        price_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        将情绪因子与价格数据对齐合并
        
        Args:
            sentiment_factors: 情绪因子 DataFrame
            price_df: 价格数据 DataFrame（来自 backtest_engine.load_data）
            
        Returns:
            合并后的 DataFrame
        """
        logger.info("[SentimentFactor] Merging sentiment factors with price data...")
        
        # 确保日期格式一致
        price_df = price_df.copy()
        if price_df['trade_date'].dtype == object:
            price_df['trade_date'] = price_df['trade_date'].astype(int)
        
        # 左连接
        merged = price_df.merge(
            sentiment_factors,
            on=['trade_date', 'symbol'],
            how='left'
        )
        
        # 统计覆盖率
        total = len(merged)
        has_sentiment = merged['sentiment_daily'].notna().sum()
        coverage = has_sentiment / total if total > 0 else 0
        
        logger.info(f"[SentimentFactor] Merge complete: {total} rows, "
                    f"{has_sentiment} with sentiment ({coverage:.1%})")
        
        return merged


def get_sentiment_calculator(
    short_window: int = 5,
    long_window: int = 20,
) -> SentimentFactorCalculator:
    """获取情绪因子计算器实例"""
    return SentimentFactorCalculator(short_window=short_window, long_window=long_window)


if __name__ == "__main__":
    # 测试
    logger.info("SentimentFactor module loaded successfully")
    
    # 创建测试数据
    test_data = pd.DataFrame({
        'trade_date': [20240101, 20240102, 20240103, 20240101, 20240102],
        'symbol': ['600519', '600519', '600519', '000858', '000858'],
        'sentiment_score': [0.5, 0.3, -0.2, 0.1, -0.4],
        'sentiment_confidence': [0.8, 0.6, 0.7, 0.5, 0.9],
    })
    
    calc = get_sentiment_calculator()
    factors = calc.compute_factors(test_data)
    print(factors)
```

### 4.2 整合到 V229 Alpha 模型

创建 `src/alpha_model_v232.py`（在 V229 基础上添加情绪因子）：

```python
"""
Alpha Model V232 - V229 + 新闻情绪因子
======================================

【核心改进】
在 V229 三因子（极端超卖 35% + 行业相对弱势 35% + 低波动 30%）基础上，
新增新闻情绪因子，尝试突破 IC ≈ 0.05 的瓶颈。

【因子体系】
1. 极端超卖 (30%) - 5日累计跌幅的截面排名
2. 行业相对弱势 (25%) - 个股相对于行业指数的超额收益
3. 低波动率 (20%) - 20日波动率的截面排名
4. 新闻情绪 (25%) - 情绪变化率（5日均值/20日均值-1）

【假设】
- 新闻情绪在 2024 年牛市中可能更有效，因为正面新闻推动市场情绪
- 情绪变化率捕捉的是"情绪改善"而非"绝对情绪"
"""

from typing import Dict, Optional
import pandas as pd
import numpy as np
from loguru import logger

# 内存优化
pd.options.mode.chained_assignment = None

# 版本号
VERSION = "V232"


class AlphaModelV232:
    """
    V232 Alpha Model - V229 + 新闻情绪因子
    
    【核心职责】
    1. 极端超卖: 5日累计跌幅的截面排名
    2. 行业相对弱势: 个股相对于行业指数的超额收益
    3. 低波动率: 20日波动率的截面排名
    4. 新闻情绪: 情绪变化率（新增）
    
    【唯一接口】
    - compute_score(df) -> DataFrame[trade_date, symbol, score]
    """
    
    # ==================== 配置参数 ====================
    # 因子权重（V229 基础上调整）
    W_EXTREME_OS = 0.30         # 极端超卖（从 0.35 降低）
    W_INDUSTRY_REL = 0.25       # 行业相对弱势（从 0.35 降低）
    W_LOW_VOL = 0.20            # 低波动率（从 0.30 降低）
    W_SENTIMENT = 0.25          # 新闻情绪（新增）
    
    # 窗口参数
    OVERSOLD_WINDOW = 5
    VOL_WINDOW = 5
    VOL_WINDOW_LOW = 20
    INDUSTRY_WINDOW = 20
    
    def __init__(self):
        """初始化 Alpha Model"""
        logger.info("=" * 70)
        logger.info("V232 Alpha Model Initialized (V229 + News Sentiment)")
        logger.info("=" * 70)
        logger.info(f"  Extreme Oversold Weight: {self.W_EXTREME_OS:.2f}")
        logger.info(f"  Industry Relative Weight: {self.W_INDUSTRY_REL:.2f}")
        logger.info(f"  Low Volatility Weight: {self.W_LOW_VOL:.2f}")
        logger.info(f"  News Sentiment Weight: {self.W_SENTIMENT:.2f}")
        logger.info("=" * 70)
    
    # ==================== 唯一公开接口 ====================
    
    def compute_score(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算 Alpha 评分 (唯一公开接口)
        
        【输入要求】
        - 必须包含: trade_date, symbol, open, high, low, close, volume
        - 必须包含: industry_code
        - 可选包含: sentiment_daily, sentiment_ma5, sentiment_ma20, 
                    sentiment_change, sentiment_vol
        """
        logger.info(f"[V232] Computing alpha scores for {len(df)} rows...")
        
        # 1. 数据预处理
        df = self._preprocess_data(df)
        
        # 2. 计算各因子
        logger.info("[V232] Computing extreme oversold factor...")
        df['f_extreme_os'] = self._compute_extreme_oversold(df)
        
        logger.info("[V232] Computing industry relative weakness factor...")
        df['f_industry_rel'] = self._compute_industry_relative(df)
        
        logger.info("[V232] Computing low volatility factor...")
        df['f_low_vol'] = self._compute_low_volatility(df)
        
        # 3. 情绪因子（如果存在）
        if 'sentiment_change' in df.columns:
            logger.info("[V232] Using pre-computed sentiment factor...")
            df['f_sentiment'] = df['sentiment_change']
        else:
            logger.warning("[V232] No sentiment factor found, using default 0.5")
            df['f_sentiment'] = 0.5
        
        # 4. 截面百分位排名标准化
        logger.info("[V232] Normalizing factors with cross-sectional percentile rank...")
        factor_cols = ['f_extreme_os', 'f_industry_rel', 'f_low_vol', 'f_sentiment']
        for col in factor_cols:
            df[col + '_rank'] = self._cross_sectional_rank(df, col)
        
        # 5. 加权合成
        logger.info("[V232] Combining factors with fixed weights...")
        df['score_raw'] = (
            self.W_EXTREME_OS * df['f_extreme_os_rank'] +
            self.W_INDUSTRY_REL * df['f_industry_rel_rank'] +
            self.W_LOW_VOL * df['f_low_vol_rank'] +
            self.W_SENTIMENT * df['f_sentiment_rank']
        )
        
        # 6. 最终截面 Rank
        df['score'] = df.groupby('trade_date')['score_raw'].rank(pct=True, na_option='keep')
        
        # 处理 NaN
        from src.data_healer import heal
        df = heal(df, numeric_cols=['score'])
        
        # 7. 输出
        result_cols = ['trade_date', 'symbol', 'score']
        if 'close' in df.columns:
            result_cols.append('close')
        result = df[result_cols].copy()
        
        logger.info(f"[V232] Score computed: mean={result['score'].mean():.4f}, "
                    f"std={result['score'].std():.4f}")
        
        return result
    
    # ==================== 数据预处理 ====================
    
    def _preprocess_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """数据预处理"""
        from src.data_healer import heal
        
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        # 计算日收益率
        result['daily_ret'] = result.groupby('symbol')['close'].pct_change()
        
        # 计算行业指数
        result = self._compute_industry_index(result)
        
        # N 日累计收益率
        result[f'ret_{self.OVERSOLD_WINDOW}d'] = result.groupby('symbol')['close'].transform(
            lambda x, d=self.OVERSOLD_WINDOW: (x / x.shift(d) - 1)
        )
        
        # 成交量移动平均
        result[f'vol_ma_{self.VOL_WINDOW}d'] = result.groupby('symbol')['volume'].transform(
            lambda x, d=self.VOL_WINDOW: x.rolling(d, min_periods=3).mean()
        )
        
        # 成交量比率
        result['vol_ratio'] = result['volume'] / result[f'vol_ma_{self.VOL_WINDOW}d'].replace(0, np.nan)
        
        # 20 日波动率
        result[f'vol_{self.VOL_WINDOW_LOW}d'] = result.groupby('symbol')['daily_ret'].transform(
            lambda x, d=self.VOL_WINDOW_LOW: x.rolling(d, min_periods=10).std()
        )
        
        result['is_down_day'] = (result['close'] < result['open']).astype(float)
        
        # 替换 inf
        result = result.replace([np.inf, -np.inf], np.nan)
        
        # 基础列 healing
        heal_cols = ['daily_ret', f'ret_{self.OVERSOLD_WINDOW}d', 'vol_ratio',
                     f'vol_{self.VOL_WINDOW_LOW}d', 'is_down_day',
                     'industry_idx_ret', 'industry_rel_ret']
        available_cols = [c for c in heal_cols if c in result.columns]
        if available_cols:
            result = heal(result, numeric_cols=available_cols)
        
        # 行业相对累计收益
        result[f'industry_rel_{self.INDUSTRY_WINDOW}d'] = result.groupby('symbol')['industry_rel_ret'].transform(
            lambda x, d=self.INDUSTRY_WINDOW: x.rolling(d, min_periods=10).sum()
        )
        
        result = heal(result, numeric_cols=[f'industry_rel_{self.INDUSTRY_WINDOW}d'])
        
        return result
    
    def _compute_industry_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算行业指数和个股相对行业的超额收益"""
        industry_daily = df.groupby(['trade_date', 'industry_code'])['daily_ret'].median().reset_index()
        industry_daily.columns = ['trade_date', 'industry_code', 'industry_idx_ret']
        
        df = df.merge(industry_daily, on=['trade_date', 'industry_code'], how='left')
        df['industry_rel_ret'] = df['daily_ret'] - df['industry_idx_ret']
        
        return df
    
    def _cross_sectional_rank(self, df: pd.DataFrame, col: str) -> pd.Series:
        """横截面百分位排名标准化"""
        rank = df.groupby('trade_date')[col].rank(pct=True, na_option='keep')
        return rank.fillna(0.5)
    
    def _compute_extreme_oversold(self, df: pd.DataFrame) -> pd.Series:
        """极端超卖因子"""
        ret_nd = df.get(f'ret_{self.OVERSOLD_WINDOW}d', pd.Series(0, index=df.index))
        return -ret_nd
    
    def _compute_industry_relative(self, df: pd.DataFrame) -> pd.Series:
        """行业相对弱势因子"""
        industry_rel = df.get(f'industry_rel_{self.INDUSTRY_WINDOW}d', pd.Series(0, index=df.index))
        return -industry_rel
    
    def _compute_low_volatility(self, df: pd.DataFrame) -> pd.Series:
        """低波动率因子"""
        vol_20d = df.get(f'vol_{self.VOL_WINDOW_LOW}d', pd.Series(0, index=df.index))
        return -vol_20d
    
    def get_factor_ics(self, df: pd.DataFrame) -> Dict[str, float]:
        """获取各因子的 IC"""
        ics = {}
        if 't1_return' not in df.columns:
            return ics
        
        for col in ['f_extreme_os', 'f_industry_rel', 'f_low_vol', 'f_sentiment', 'score']:
            if col in df.columns:
                ic = self._calculate_daily_ic(df, col, 't1_return')
                ics[col] = ic
        return ics
    
    def _calculate_daily_ic(self, df: pd.DataFrame, factor_col: str, return_col: str) -> float:
        """计算日度平均 IC"""
        ic_values = []
        
        for date in sorted(df['trade_date'].unique()):
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 10:
                continue
            
            factor_vals = day_data[factor_col]
            return_vals = day_data[return_col]
            
            mask = factor_vals.notna() & return_vals.notna()
            if mask.sum() < 10:
                continue
            
            corr = factor_vals[mask].corr(return_vals[mask], method='spearman')
            if not np.isnan(corr):
                ic_values.append(corr)
        
        return float(np.mean(ic_values)) if ic_values else 0.0


def get_alpha_model() -> AlphaModelV232:
    """获取 Alpha Model 实例"""
    return AlphaModelV232()


if __name__ == "__main__":
    logger.info("V232 Alpha Model loaded successfully")
    model = get_alpha_model()
    logger.info(f"Version: {VERSION}")
```

---

## 五、回测验证

### 5.1 完整回测流程

创建 `run_v232.py`：

```python
"""
V232 回测脚本 - V229 + 新闻情绪因子
====================================

【运行方式】
python run_v232.py

【流程】
1. 获取新闻数据
2. 情绪分析
3. 计算情绪因子
4. 合并到价格数据
5. 运行回测
6. 对比 IC 变化
"""

import sys
import os
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np
from loguru import logger
from dotenv import load_dotenv

# 配置日志
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add("reports/v232_backtest.log", level="DEBUG", rotation="10 MB")

# 加载环境变量
load_dotenv()


def main():
    """主回测流程"""
    logger.info("=" * 70)
    logger.info("V232 Backtest Starting (V229 + News Sentiment Factor)")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    # ==================== Step 1: 加载价格数据 ====================
    logger.info("\n[Step 1] Loading price data...")
    from src.backtest_engine import BacktestEngine
    
    engine = BacktestEngine(output_dir="reports")
    
    years = [2020, 2022, 2024]
    df = engine.load_data(years, warmup_year=2019, warmup_days=60)
    
    if df.empty:
        logger.error("No data loaded. Exiting.")
        return
    
    logger.info(f"Price data: {len(df)} rows, {df['symbol'].nunique()} symbols")
    
    # ==================== Step 2: 获取新闻数据 ====================
    logger.info("\n[Step 2] Fetching news data...")
    from src.data_sources.sina_news_fetcher import SinaNewsFetcher
    
    # 确定日期范围
    date_min = str(df['trade_date'].astype(str).str[:4].min()) + "-01-01"
    date_max = str(df['trade_date'].astype(str).str[:4].max()) + "-12-31"
    logger.info(f"News date range: {date_min} to {date_max}")
    
    # 获取新闻（首次运行建议限制范围）
    fetcher = SinaNewsFetcher()
    news_df = fetcher.fetch_news_batch(
        start_date=date_min,
        end_date=date_max,
        max_pages_per_day=5  # 每天最多 5 页（250 条新闻）
    )
    
    if news_df.empty:
        logger.warning("No news data fetched. Running V229 baseline instead.")
        run_v229_baseline(df, engine)
        return
    
    logger.info(f"News data: {len(news_df)} articles")
    
    # ==================== Step 3: 情绪分析 ====================
    logger.info("\n[Step 3] Running sentiment analysis...")
    from src.sentiment_analyzer import FinSentimentAnalyzer
    
    analyzer = FinSentimentAnalyzer(
        model_name="warrenxin/FinBERT",
        batch_size=32,
    )
    
    # 对新闻进行情绪分析
    news_df = analyzer.analyze_news_dataframe(
        news_df,
        text_cols=["title", "summary"],
        output_col="sentiment",
    )
    
    # 保存情绪分析结果
    news_sentiment = news_df[['news_id', 'title', 'fetch_date',
                               'stock_codes', 'sentiment_score',
                               'sentiment_label', 'sentiment_confidence']].copy()
    news_sentiment.to_csv("reports/news_sentiment_results.csv", index=False, encoding="utf-8-sig")
    logger.info(f"Sentiment results saved: {len(news_sentiment)} rows")
    
    # ==================== Step 4: 构建情绪因子 ====================
    logger.info("\n[Step 4] Computing sentiment factors...")
    from src.sentiment_factor import SentimentFactorCalculator
    
    # 将新闻关联到股票
    # 只处理包含股票代码的新闻
    stock_news = news_df[news_df['stock_codes'].apply(lambda x: len(x) > 0)].copy()
    
    # 展开股票代码（一条新闻可能关联多只股票）
    rows = []
    for _, row in stock_news.iterrows():
        for code in row['stock_codes']:
            rows.append({
                'trade_date': int(row['fetch_date'].replace('-', '')),
                'symbol': code,
                'sentiment_score': row['sentiment_score'],
                'sentiment_confidence': row['sentiment_confidence'],
            })
    
    sentiment_input = pd.DataFrame(rows)
    
    if sentiment_input.empty:
        logger.warning("No stock-related news found. Using V229 baseline.")
        run_v229_baseline(df, engine)
        return
    
    logger.info(f"Stock-related news: {len(sentiment_input)} entries")
    
    # 计算情绪因子
    calc = SentimentFactorCalculator(short_window=5, long_window=20)
    sentiment_factors = calc.compute_factors(sentiment_input)
    
    # 合并到价格数据
    merged_df = calc.merge_with_price_data(sentiment_factors, df)
    
    # ==================== Step 5: 运行回测 ====================
    logger.info("\n[Step 5] Running backtest...")
    from src.alpha_model_v232 import AlphaModelV232
    
    model = AlphaModelV232()
    
    # 运行回测（使用与 V229 相同的方式）
    for year in years:
        year_str = str(year)
        year_data = merged_df[merged_df['trade_date'].astype(str).str.startswith(year_str)].copy()
        
        if year_data.empty:
            logger.warning(f"[Year {year}] No data")
            continue
        
        logger.info(f"[Year {year}] Computing scores...")
        score_df = model.compute_score(year_data)
        
        # 合并得分
        year_data = year_data.merge(score_df[['trade_date', 'symbol', 'score']],
                                    on=['trade_date', 'symbol'], how='left')
        
        # 计算 IC
        from src.alpha_model_v232 import AlphaModelV232
        ics = model.get_factor_ics(year_data)
        logger.info(f"[Year {year}] Factor ICs:")
        for factor, ic in ics.items():
            logger.info(f"  {factor}: {ic:.4f}")
    
    # ==================== Step 6: 生成报告 ====================
    logger.info("\n[Step 6] Generating comparison report...")
    
    elapsed = time.time() - start_time
    logger.info(f"Backtest completed in {elapsed:.0f} seconds ({elapsed/3600:.1f} hours)")
    
    logger.info("=" * 70)
    logger.info("V232 Backtest Complete")
    logger.info("=" * 70)


def run_v229_baseline(df: pd.DataFrame, engine):
    """运行 V229 基线回测作为对比"""
    logger.info("Running V229 baseline for comparison...")
    from src.alpha_model_v229 import AlphaModelV229
    
    model = AlphaModelV229()
    score_df = model.compute_score(df)
    
    # 计算 IC
    df_with_score = df.merge(score_df, on=['trade_date', 'symbol'], how='left')
    ics = model.get_factor_ics(df_with_score)
    
    logger.info("V229 Baseline Factor ICs:")
    for factor, ic in ics.items():
        logger.info(f"  {factor}: {ic:.4f}")


if __name__ == "__main__":
    main()
```

### 5.2 IC 对比分析脚本

创建 `scripts/compare_ic_v229_v232.py`：

```python
"""
V229 vs V232 IC 对比分析
========================
对比添加情绪因子前后的 IC 变化
"""

import pandas as pd
import numpy as np
from loguru import logger


def compare_ic_results():
    """对比两个版本的 IC 结果"""
    
    # 定义对比数据（需要从回测日志或结果中提取）
    v229_results = {
        2020: {"t1_ic": 0.0451, "ic_ir": 0.35, "return": 0.5267},
        2022: {"t1_ic": 0.0659, "ic_ir": 0.56, "return": 0.1957},
        2024: {"t1_ic": 0.0388, "ic_ir": 0.20, "return": -0.5537},
    }
    
    # V232 结果（运行回测后填入）
    v232_results = {
        2020: {"t1_ic": None, "ic_ir": None, "return": None},
        2022: {"t1_ic": None, "ic_ir": None, "return": None},
        2024: {"t1_ic": None, "ic_ir": None, "return": None},
    }
    
    # 打印对比表
    logger.info("=" * 70)
    logger.info("V229 vs V232 IC Comparison")
    logger.info("=" * 70)
    
    print("\n| Year | Metric | V229 | V232 | Change |")
    print("|------|--------|------|------|--------|")
    
    for year in [2020, 2022, 2024]:
        v29 = v229_results[year]
        v32 = v232_results[year]
        
        if v32["t1_ic"] is not None:
            ic_change = v32["t1_ic"] - v29["t1_ic"]
            print(f"| {year} | T+1 IC | {v29['t1_ic']:.4f} | {v32['t1_ic']:.4f} | {ic_change:+.4f} |")
        else:
            print(f"| {year} | T+1 IC | {v29['t1_ic']:.4f} | N/A | N/A |")
    
    # 平均 IC
    v29_avg = np.mean([v["t1_ic"] for v in v229_results.values()])
    v32_valid = [v["t1_ic"] for v in v232_results.values() if v["t1_ic"] is not None]
    
    if v32_valid:
        v32_avg = np.mean(v32_valid)
        print(f"\n| Avg IC | {v29_avg:.4f} | {v32_avg:.4f} | {v32_avg-v29_avg:+.4f} |")
    
    print("\n" + "=" * 70)
    print("结论判断:")
    print(f"  - V232 平均 IC >= 0.05? {'是 ✅' if v32_valid and v32_avg >= 0.05 else '否 ❌'}")
    print(f"  - 2024 年收益 > 0%? {'是 ✅' if v232_results[2024]['return'] is not None and v232_results[2024]['return'] > 0 else '否/待验证 ❌'}")


if __name__ == "__main__":
    compare_ic_results()
```

### 5.3 预期验证结果

基于理论分析，情绪因子的预期效果：

| 指标 | V229 当前值 | V232 预期值 | 预期变化 |
|------|------------|------------|---------|
| 2020 IC | 0.0451 | 0.045~0.055 | +0~+0.01 |
| 2022 IC | 0.0659 | 0.065~0.075 | +0~+0.01 |
| **2024 IC** | **0.0388** | **0.045~0.060** | **+0.006~+0.021** |
| Avg IC | 0.0499 | **0.052~0.063** | +0.002~+0.013 |
| 2024 收益 | -55.37% | **-20%~+10%** | +35~+65pp |

**核心判断标准**：
- 如果 V232 Avg IC ≥ 0.05 → 情绪因子有效，继续优化
- 如果 2024 年收益 > -20% → 情绪因子改善了牛市表现
- 如果两个条件都不满足 → 需要调整情绪因子的计算方式或权重

---

## 六、故障排除

### 6.1 模型加载慢 / 下载失败

**症状**：
- `FinBERT` 模型加载耗时超过 10 分钟
- 出现 `ConnectionError` 或 `Timeout`

**解决方案**：

```python
# 方案 1: 使用国内镜像
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# 设置 HuggingFace 镜像
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

# 方案 2: 手动下载模型
# 1. 访问 https://hf-mirror.com/warrenxin/FinBERT
# 2. 下载所有文件到本地目录: ./models/finbert
# 3. 加载时指定本地路径
tokenizer = AutoTokenizer.from_pretrained("./models/finbert")
model = AutoModelForSequenceClassification.from_pretrained("./models/finbert")

# 方案 3: 使用更小的模型
# 如果显存不足，可以使用蒸馏版模型
tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-jd-binary-chinese")
```

### 6.2 API 限流 / 请求被拒绝

**症状**：
- 新浪财经 API 返回 403 错误
- 东方财富 API 返回频率限制

**解决方案**：

```python
# 1. 增加请求间隔
class SinaNewsFetcher:
    REQUEST_INTERVAL = 2.0  # 从 0.5 增加到 2 秒
    MAX_RETRIES = 5

# 2. 使用代理
import requests
proxies = {
    "http": "http://127.0.0.1:7890",
    "https": "http://127.0.0.1:7890",
}
session = requests.Session()
session.proxies.update(proxies)

# 3. 使用缓存
import joblib
import hashlib

def get_cached_news(fetcher, date_str, cache_dir="cache/news"):
    import os
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = f"{cache_dir}/{date_str}.pkl"
    
    if os.path.exists(cache_path):
        return joblib.load(cache_path)
    
    df = fetcher.fetch_news_by_date(date_str)
    joblib.dump(df, cache_path)
    return df

# 4. 分批运行，避免一次性请求过多
for year in [2020, 2022, 2024]:
    for month in range(1, 13):
        date = f"{year}-{month:02d}-01"
        # 每月单独请求
        time.sleep(5)  # 月份间休息
```

### 6.3 情绪得分全部为 0 或 NaN

**症状**：
- `sentiment_score` 列全部为 0 或 NaN
- 情绪因子没有区分度

**排查步骤**：

```python
# 1. 检查输入文本是否为空
print(news_df['title'].isna().sum())
print(news_df['title'].apply(lambda x: len(str(x)) if x else 0).describe())

# 2. 检查模型输出
analyzer = FinSentimentAnalyzer()
test_result = analyzer.predict_single("测试文本")
print(test_result)

# 3. 检查模型标签映射
print(analyzer.id2label)
print(analyzer.label2id)

# 如果标签不是 positive/neutral/negative，需要调整代码
# 例如有些模型输出是 '1'/'0'/'-1'

# 4. 确保文本编码正确
news_df['title'] = news_df['title'].astype(str).str.strip()
news_df = news_df[news_df['title'].str.len() > 0]  # 过滤空文本
```

### 6.4 情绪因子与价格数据对齐失败

**症状**：
- 合并后 `sentiment_daily` 列大部分为 NaN
- 覆盖率 < 10%

**解决方案**：

```python
# 1. 检查日期格式是否一致
print(price_df['trade_date'].dtype)  # 应该是 int (YYYYMMDD)
print(sentiment_factors['trade_date'].dtype)

# 2. 统一转换
price_df['trade_date'] = pd.to_datetime(price_df['trade_date']).dt.strftime('%Y%m%d').astype(int)
sentiment_factors['trade_date'] = pd.to_datetime(sentiment_factors['trade_date']).dt.strftime('%Y%m%d').astype(int)

# 3. 检查股票代码格式
print(price_df['symbol'].unique()[:10])
print(sentiment_factors['symbol'].unique()[:10])

# A股代码可能需要补零
sentiment_factors['symbol'] = sentiment_factors['symbol'].str.zfill(6)

# 4. 检查重叠范围
price_dates = set(price_df['trade_date'].unique())
sentiment_dates = set(sentiment_factors['trade_date'].unique())
overlap = price_dates & sentiment_dates
print(f"Date overlap: {len(overlap)} / {len(price_dates)}")

price_symbols = set(price_df['symbol'].unique())
sentiment_symbols = set(sentiment_factors['symbol'].unique())
symbol_overlap = price_symbols & sentiment_symbols
print(f"Symbol overlap: {len(symbol_overlap)} / {len(price_symbols)}")
```

### 6.5 内存不足 (OOM)

**症状**：
- `CUDA out of memory` 错误
- Python 进程被系统杀死

**解决方案**：

```python
# 1. 减小 batch_size
analyzer = FinSentimentAnalyzer(batch_size=8)  # 从 32 减到 8

# 2. 减小 max_length
analyzer = FinSentimentAnalyzer(max_length=128)  # 从 512 减到 128

# 3. 分批处理，释放中间结果
import gc

for batch_start in range(0, len(texts), batch_size):
    batch_texts = texts[batch_start:batch_start + batch_size]
    results = analyzer.predict_batch(batch_texts)
    # 处理结果...
    del batch_texts
    gc.collect()

# 4. 使用 CPU 而非 GPU（显存不足时）
analyzer = FinSentimentAnalyzer(device="cpu")

# 5. 减少数据加载量
# 只加载回测需要的年份，不要全量加载
```

### 6.6 数据库连接失败

**症状**：
- `sqlalchemy.exc.OperationalError`
- 无法连接到 MySQL

**解决方案**：

```python
# 1. 检查 .env 文件配置
# DATABASE_URL=mysql+pymysql://user:password@host:3306/dbname

# 2. 测试连接
from sqlalchemy import create_engine, text
engine = create_engine(os.getenv("DATABASE_URL"))
with engine.connect() as conn:
    result = conn.execute(text("SELECT 1"))
    print("Connection OK:", result.fetchone())

# 3. 增加连接超时
engine = create_engine(
    os.getenv("DATABASE_URL"),
    connect_args={"connect_timeout": 30}
)

# 4. 检查 MySQL 服务状态
# Windows: services.msc -> MySQL
# Linux: systemctl status mysql
```

### 6.7 回测结果为空

**症状**：
- 回测输出为空 DataFrame
- 没有 IC 计算结果

**排查清单**：

```python
# 1. 检查输入数据是否为空
print(f"Input data: {len(df)} rows")
print(f"Unique dates: {df['trade_date'].nunique()}")
print(f"Unique symbols: {df['symbol'].nunique()}")

# 2. 检查因子计算是否产生有效值
print(f"sentiment_daily null: {df['sentiment_daily'].isna().sum()}")
print(f"score null: {df['score'].isna().sum()}")

# 3. 检查日期过滤是否正确
year_data = df[df['trade_date'].astype(str).str.startswith('2024')]
print(f"2024 data: {len(year_data)} rows")

# 4. 检查 IC 计算是否有足够样本
for date in sorted(df['trade_date'].unique())[:5]:
    day_data = df[df['trade_date'] == date]
    print(f"Date {date}: {len(day_data)} stocks")
```

---

## 七、时间成本估算

| 步骤 | 任务 | 预计时间 | 备注 |
|------|------|---------|------|
| **1** | 环境配置 & 依赖安装 | 0.5 小时 | 包括验证环境 |
| **2** | 模型下载 & 测试 | 1 小时 | 400MB 模型下载 + 验证推理 |
| **3** | 新闻数据获取（2020-2024） | 3-5 小时 | 取决于 API 速度和反爬限制 |
| **4** | 情绪分析推理 | 2-4 小时 | 取决于硬件（GPU 2h，CPU 4h+） |
| **5** | 情绪因子计算 & 整合 | 1 小时 | 数据处理和对齐 |
| **6** | 回测运行（3个年份） | 1-2 小时 | 与 V229 相同规模 |
| **7** | 结果分析 & IC 对比 | 0.5 小时 | 编写对比报告 |
| **8** | 调参优化（可选） | 2-4 小时 | 调整因子权重 |
| **总计** | | **11-17 小时** | **约 1.5-2 个工作日** |

### 关键路径优化建议

1. **并行获取新闻数据**：可以先获取新闻数据，同时在另一台机器上下载模型
2. **缓存新闻和情绪结果**：首次运行后将结果保存到数据库/文件，后续只需增量更新
3. **优先测试 2024 年**：2024 年是判断成败的关键年份，可以先跑通 2024 年再跑其他年份

---

## 八、后续优化方向

如果 V232 回测结果达标（Avg IC ≥ 0.05），可以考虑以下进一步优化：

### 8.1 情绪因子改进

```python
# 1. 新闻来源加权
# - 权威媒体（证券时报、中国证券报）权重更高
# - 自媒体权重较低

# 2. 新闻热度因子
# - 同一股票短时间内大量新闻 = 关注度高
# - 新闻数量变化率 = 市场关注度变化

# 3. 情绪极性分解
# - 正面新闻比例
# - 负面新闻比例
# - 情绪分歧度（正负新闻都多 = 分歧大）
```

### 8.2 多源数据融合

```python
# 1. 分析师评级情绪
# - 评级上调 = 积极信号
# - 目标价上调 = 积极信号

# 2. 社交媒体情绪
# - 雪球/股吧讨论热度
# - 散户情绪指标（反向指标？）

# 3. 期权隐含波动率
# - IV 变化 = 市场预期变化
# - Put/Call ratio = 情绪指标
```

### 8.3 动态权重调整

```python
# 根据市场状态调整情绪因子权重
# - 牛市：情绪权重更高（0.3-0.4）
# - 熊市：情绪权重较低（0.1-0.2）
# - 震荡市：保持当前权重（0.25）
```

---

## 九、检查清单

在开始实施前，请确认以下准备项：

- [ ] Python 环境已配置（3.13.x）
- [ ] 所有依赖库已安装
- [ ] HuggingFace 可访问（或使用镜像）
- [ ] MySQL 数据库可连接
- [ ] .env 配置文件正确
- [ ] 磁盘空间充足（模型 400MB + 缓存 2-5GB）
- [ ] 有足够的计算资源（推荐 GPU，至少 8GB RAM）
- [ ] 已备份 V229 基线结果

---

*本指南基于 VFINAL 跨年度迭代报告编写，旨在为团队提供清晰的操作路径。*  
*编写时间: 2026-04-29*  
*项目路径: d:\PythonProject\Quantitative-Trading*
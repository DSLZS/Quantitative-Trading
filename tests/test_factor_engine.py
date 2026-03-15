"""
Unit tests for FactorEngine module.

【Iteration 17 更新】:
- 测试数据必须包含 symbol 和 trade_date 列
- 模拟真实的跨期、多标的数据环境
- 测试 compute_hist_sharpe 的分标的计算逻辑
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path
from datetime import datetime

from src.factor_engine import FactorEngine


class TestFactorEngine:
    """Tests for FactorEngine class."""

    @pytest.fixture
    def sample_config(self, tmp_path):
        """Create a temporary factor config file."""
        config_content = """
factors:
  - name: momentum_5
    description: "5-day momentum"
    expression: "close / close.shift(5) - 1"
    window: 5
    
  - name: volatility_5
    description: "5-day volatility"
    expression: "close.rolling_std(5)"
    window: 5

label:
  name: future_return_5
  description: "5-day forward return"
  expression: "close.shift(-5) / close - 1"
  window: 5
"""
        config_file = tmp_path / "factors.yaml"
        config_file.write_text(config_content)
        return str(config_file)

    @pytest.fixture
    def sample_ohlcv_data(self):
        """Create sample OHLCV data for testing with symbol and trade_date columns.
        
        【Iteration 17 更新】:
        - 包含多只股票数据 (跨标的)
        - 包含连续的交易日 (跨期)
        - 模拟真实的量价数据
        """
        np.random.seed(42)
        n_symbols = 5  # 5 只股票
        n_days = 50    # 50 个交易日
        n_rows = n_symbols * n_days
        
        # 生成股票和日期组合
        symbols = ["STOCK_A", "STOCK_B", "STOCK_C", "STOCK_D", "STOCK_E"]
        trade_dates = pl.date_range(
            start=datetime(2024, 1, 1),
            end=datetime(2024, 3, 20),
            eager=True
        )[:n_days]
        
        # 创建多股票数据
        symbol_list = []
        trade_date_list = []
        close_list = []
        high_list = []
        low_list = []
        volume_list = []
        
        for sym_idx, symbol in enumerate(symbols):
            # 每只股票有不同的基础价格
            base_price = 100 + sym_idx * 10
            
            # 生成价格序列 (带趋势)
            returns = np.random.randn(n_days) * 0.02
            close = base_price * np.cumprod(1 + returns)
            
            for day_idx in range(n_days):
                symbol_list.append(symbol)
                trade_date_list.append(trade_dates[day_idx])
                close_list.append(close[day_idx])
                high_list.append(close[day_idx] * (1 + abs(np.random.randn() * 0.01)))
                low_list.append(close[day_idx] * (1 - abs(np.random.randn() * 0.01)))
                volume_list.append(np.random.randint(1000000, 10000000))
        
        return pl.DataFrame({
            "symbol": symbol_list,
            "trade_date": trade_date_list,
            "open": close_list,  # 简化：open=前一日 close
            "high": high_list,
            "low": low_list,
            "close": close_list,
            "volume": volume_list,
        })

    def test_load_config(self, sample_config):
        """Test loading factor configuration."""
        engine = FactorEngine(sample_config)
        
        assert len(engine.factors) == 2
        assert engine.label_config is not None
        assert engine.label_config["name"] == "future_return_5"

    def test_get_factor_names(self, sample_config):
        """Test getting factor names."""
        engine = FactorEngine(sample_config)
        names = engine.get_factor_names()
        
        assert names == ["momentum_5", "volatility_5"]

    def test_get_feature_columns(self, sample_config):
        """Test getting all feature columns."""
        engine = FactorEngine(sample_config)
        columns = engine.get_feature_columns()
        
        assert "momentum_5" in columns
        assert "volatility_5" in columns
        assert "future_return_5" in columns

    def test_compute_factors(self, sample_config, sample_ohlcv_data):
        """Test computing factors on sample data."""
        engine = FactorEngine(sample_config)
        
        # Add trade_date column to avoid ValueError in compute_label
        sample_ohlcv_data = sample_ohlcv_data.with_columns([
            pl.Series("trade_date", ["2024-01-01"] * len(sample_ohlcv_data))
        ])
        
        result = engine.compute_factors(sample_ohlcv_data)
        
        # Check that factor columns were added
        assert "momentum_5" in result.columns
        assert "volatility_5" in result.columns
        
        # Check that original columns are preserved
        assert "close" in result.columns
        assert "volume" in result.columns
        
        # Check row count is preserved
        assert len(result) == len(sample_ohlcv_data)

    def test_compute_label(self, sample_config, sample_ohlcv_data):
        """Test computing label on sample data with proper symbol and trade_date columns."""
        engine = FactorEngine(sample_config)
        
        # sample_ohlcv_data already has symbol and trade_date columns
        result = engine.compute_label(sample_ohlcv_data)
        
        # Check that label column was added (new naming: future_return_5d)
        assert "future_return_5d" in result.columns
        assert "label_5d" in result.columns
        
        # Check row count is preserved
        assert len(result) == len(sample_ohlcv_data)
        
        # Check that label_5d has correct categories (0, 1, 2)
        unique_labels = result["label_5d"].unique().sort().to_list()
        assert all(label in [0, 1, 2] for label in unique_labels)

    def test_compute_factors_and_label(self, sample_config, sample_ohlcv_data):
        """Test computing both factors and labels with proper multi-stock data."""
        engine = FactorEngine(sample_config)
        
        result = engine.compute_factors(sample_ohlcv_data)
        result = engine.compute_label(result)
        
        # Check all expected columns exist
        expected_cols = ["symbol", "trade_date", "open", "high", "low", "close", "volume", 
                         "momentum_5", "volatility_5", "future_return_5d", "label_5d"]
        for col in expected_cols:
            assert col in result.columns
        
        # Check that all symbols are present
        unique_symbols = result["symbol"].unique().sort().to_list()
        assert len(unique_symbols) == 5
        
        # Check that trade_date grouping works correctly
        unique_dates = result["trade_date"].unique().sort().to_list()
        assert len(unique_dates) == 50

    def test_missing_label_config(self, tmp_path):
        """Test that compute_label works without label config (uses label_5d as default)."""
        config_content = """
factors:
  - name: momentum_5
    expression: "close / close.shift(5) - 1"
    window: 5
"""
        config_file = tmp_path / "factors.yaml"
        config_file.write_text(config_content)
        
        engine = FactorEngine(str(config_file))
        
        # compute_label requires trade_date column to avoid look-ahead bias
        # Test that it raises ValueError without trade_date
        with pytest.raises(ValueError, match="trade_date"):
            engine.compute_label(pl.DataFrame({"close": [1.0, 2.0, 3.0, 4.0, 5.0]}))
        
        # Test with trade_date column - should work correctly
        result = engine.compute_label(pl.DataFrame({
            "close": [1.0, 2.0, 3.0, 4.0, 5.0],
            "trade_date": ["2024-01-01"] * 5
        }))
        
        # Check that label_5d was added (new default label)
        assert "label_5d" in result.columns
        assert "future_return_5d" in result.columns

    def test_compute_hist_sharpe_requires_symbol(self, sample_config):
        """Test that compute_hist_sharpe raises ValueError without symbol column.
        
        【Iteration 17 修复验证】:
        - 没有 symbol 列时应抛出 ValueError，而不是使用全局计算
        """
        engine = FactorEngine(sample_config)
        
        # Test without symbol column - should raise ValueError
        df_no_symbol = pl.DataFrame({
            "close": [100.0, 102.0, 101.0, 103.0, 105.0] * 10,  # 50 行数据
            "trade_date": ["2024-01-01"] * 50,
        })
        
        with pytest.raises(ValueError, match="symbol"):
            engine.compute_hist_sharpe(df_no_symbol, window=5)
    
    def test_compute_hist_sharpe_per_symbol(self, sample_config):
        """Test that compute_hist_sharpe computes correctly per symbol.
        
        【Iteration 17 修复验证】:
        - 验证不同股票的夏普比率是独立计算的
        - 验证 symbol 分组计算的正确性
        """
        engine = FactorEngine(sample_config)
        
        # 创建两只股票的数据，一只股票上涨，一只股票下跌
        n_days = 30
        trade_dates = list(pl.date_range(datetime(2024, 1, 1), datetime(2024, 1, 30), eager=True))
        
        df = pl.DataFrame({
            "symbol": ["STOCK_UP"] * n_days + ["STOCK_DOWN"] * n_days,
            "trade_date": trade_dates + trade_dates,  # 每个股票都有完整的日期序列
            "close": [100.0 * (1.02 ** i) for i in range(n_days)] +  # 上涨股票
                     [100.0 * (0.98 ** i) for i in range(n_days)],   # 下跌股票
        })
        
        result = engine.compute_hist_sharpe(df, window=10)
        
        # 检查 hist_sharpe_20d 列已添加
        assert "hist_sharpe_20d" in result.columns
        
        # 检查行数量
        assert len(result) == 2 * n_days
        
        # 验证上涨股票的夏普比率应该更高
        stock_up_sharpe = result.filter(pl.col("symbol") == "STOCK_UP")["hist_sharpe_20d"]
        stock_down_sharpe = result.filter(pl.col("symbol") == "STOCK_DOWN")["hist_sharpe_20d"]
        
        # 过滤掉 null 值
        stock_up_sharpe_valid = [s for s in stock_up_sharpe if s is not None and s != 0]
        stock_down_sharpe_valid = [s for s in stock_down_sharpe if s is not None and s != 0]
        
        # 上涨股票的平均夏普比率应该高于下跌股票
        if stock_up_sharpe_valid and stock_down_sharpe_valid:
            avg_up = sum(stock_up_sharpe_valid) / len(stock_up_sharpe_valid)
            avg_down = sum(stock_down_sharpe_valid) / len(stock_down_sharpe_valid)
            # 上涨股票的夏普比率应该显著更高
            assert avg_up > avg_down, f"Expected UP stock Sharpe ({avg_up:.4f}) > DOWN stock Sharpe ({avg_down:.4f})"
    
    def test_compute_hist_sharpe_multi_symbol_alignment(self, sample_config):
        """Test that compute_hist_sharpe correctly aligns data by symbol.
        
        【Iteration 17 修复验证】:
        - 验证交错存储的多股票数据能正确对齐计算
        """
        engine = FactorEngine(sample_config)
        
        # 创建交错存储的多股票数据 (真实场景)
        n_days = 25
        symbols = ["A", "B", "C"]
        
        symbol_list = []
        trade_date_list = []
        close_list = []
        
        for day in range(n_days):
            for sym in symbols:
                symbol_list.append(sym)
                trade_date_list.append(f"2024-01-{day+1:02d}")
                # 不同股票有不同的价格趋势
                base_price = {"A": 100, "B": 50, "C": 200}[sym]
                close_list.append(base_price * (1.01 ** day))
        
        df = pl.DataFrame({
            "symbol": symbol_list,
            "trade_date": trade_date_list,
            "close": close_list,
        })
        
        # 应该正常工作，不抛出异常
        result = engine.compute_hist_sharpe(df, window=5)
        
        assert "hist_sharpe_20d" in result.columns
        assert len(result) == n_days * len(symbols)
        
        # 验证每个股票都有计算结果
        for sym in symbols:
            sym_data = result.filter(pl.col("symbol") == sym)
            assert len(sym_data) == n_days
            # 检查有非空的夏普比率值 (排除前 window 行)
            valid_sharpe = sym_data.filter(pl.col("hist_sharpe_20d").is_not_null())
            assert len(valid_sharpe) > 0, f"Symbol {sym} should have valid Sharpe values"

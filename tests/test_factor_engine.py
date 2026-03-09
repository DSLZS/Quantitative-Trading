"""
Unit tests for FactorEngine module.
"""

import pytest
import polars as pl
import numpy as np
from pathlib import Path

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
        """Create sample OHLCV data for testing."""
        np.random.seed(42)
        n_rows = 100
        
        # Generate realistic price data
        close = 100 * np.cumprod(1 + np.random.randn(n_rows) * 0.02)
        high = close * (1 + np.abs(np.random.randn(n_rows) * 0.01))
        low = close * (1 - np.abs(np.random.randn(n_rows) * 0.01))
        open_price = np.roll(close, 1)
        open_price[0] = close[0]
        volume = np.random.randint(1000000, 10000000, n_rows)
        
        return pl.DataFrame({
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
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
        """Test computing label on sample data."""
        engine = FactorEngine(sample_config)
        
        result = engine.compute_label(sample_ohlcv_data)
        
        # Check that label column was added
        assert "future_return_5" in result.columns
        
        # Check row count is preserved
        assert len(result) == len(sample_ohlcv_data)

    def test_compute_factors_and_label(self, sample_config, sample_ohlcv_data):
        """Test computing both factors and labels."""
        engine = FactorEngine(sample_config)
        
        result = engine.compute_factors(sample_ohlcv_data)
        result = engine.compute_label(result)
        
        # Check all expected columns exist
        expected_cols = ["open", "high", "low", "close", "volume", 
                         "momentum_5", "volatility_5", "future_return_5"]
        for col in expected_cols:
            assert col in result.columns

    def test_missing_label_config(self, tmp_path):
        """Test error when label config is missing."""
        config_content = """
factors:
  - name: momentum_5
    expression: "close / close.shift(5) - 1"
    window: 5
"""
        config_file = tmp_path / "factors.yaml"
        config_file.write_text(config_content)
        
        engine = FactorEngine(str(config_file))
        
        with pytest.raises(ValueError, match="No label configuration"):
            engine.compute_label(pl.DataFrame({"close": [1, 2, 3]}))
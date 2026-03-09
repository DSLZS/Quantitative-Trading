"""
Unit tests for DatabaseManager module.
"""

import os
import pytest
from unittest.mock import patch, MagicMock
from sqlalchemy import create_engine
from sqlalchemy.pool import QueuePool

from src.db_manager import DatabaseManager


class TestDatabaseManager:
    """Tests for DatabaseManager singleton and methods."""

    def test_singleton_pattern(self):
        """Test that DatabaseManager follows singleton pattern."""
        db1 = DatabaseManager()
        db2 = DatabaseManager()
        assert db1 is db2, "DatabaseManager should be a singleton"

    def test_build_connection_url_default(self):
        """Test connection URL building with default values."""
        db = DatabaseManager()
        with patch.dict(os.environ, {}, clear=True):
            url = db._build_connection_url()
            assert "localhost" in url
            assert "3306" in url
            assert "quant_trading" in url

    def test_build_connection_url_custom(self):
        """Test connection URL building with custom environment."""
        db = DatabaseManager()
        with patch.dict(
            os.environ,
            {
                "MYSQL_HOST": "127.0.0.1",
                "MYSQL_PORT": "3307",
                "MYSQL_USER": "test_user",
                "MYSQL_PASSWORD": "test_pass",
                "MYSQL_DATABASE": "test_db",
            },
        ):
            url = db._build_connection_url()
            assert "127.0.0.1" in url
            assert "3307" in url
            assert "test_user" in url
            assert "test_pass" in url
            assert "test_db" in url

    def test_connect_creates_engine(self):
        """Test that connect() creates engine with proper settings."""
        db = DatabaseManager()
        db._engine = None  # Reset for testing
        
        with patch("src.db_manager.create_engine") as mock_create:
            mock_engine = MagicMock()
            mock_create.return_value = mock_engine
            
            with patch.dict(
                os.environ,
                {"MYSQL_HOST": "localhost", "MYSQL_PASSWORD": "test"},
            ):
                db.connect()
                
                assert mock_create.called
                call_args = mock_create.call_args
                
                # Check pool settings
                assert call_args.kwargs["poolclass"] == QueuePool
                assert call_args.kwargs["pool_size"] == 10
                assert call_args.kwargs["max_overflow"] == 20

    def test_close_disposes_engine(self):
        """Test that close() properly disposes engine."""
        db = DatabaseManager()
        mock_engine = MagicMock()
        db._engine = mock_engine
        
        db.close()
        
        mock_engine.dispose.assert_called_once()
        assert db._engine is None

    def test_get_connection_context_manager(self):
        """Test context manager for database connections."""
        db = DatabaseManager()
        mock_engine = MagicMock()
        mock_conn = MagicMock()
        mock_engine.connect.return_value = mock_conn
        db._engine = mock_engine
        
        with db.get_connection() as conn:
            assert conn is mock_conn
        
        mock_conn.close.assert_called_once()



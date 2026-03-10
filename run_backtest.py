#!/usr/bin/env python3
"""
Backtest Runner Script

This script runs the complete backtesting pipeline:
1. Load features from Parquet file
2. Run backtest with configurable parameters
3. Generate performance reports and visualizations

Usage:
    python run_backtest.py [options]
    
Examples:
    python run_backtest.py  # Run with default parameters
    python run_backtest.py --threshold 0.01 --capital 500000
    python run_backtest.py --parquet data/parquet/features_latest.parquet
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from dotenv import load_dotenv
from loguru import logger

from backtester import Backtester, run_backtest
from visualizer import Visualizer, generate_backtest_report

# Load environment variables
load_dotenv()

# Configure logging
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
)
logger.add(
    "logs/backtest_{time:YYYY-MM-DD}.log",
    rotation="00:00",
    retention="7 days",
    level="DEBUG",
)


def main() -> None:
    """Main entry point for backtest runner."""
    parser = argparse.ArgumentParser(description="Run quantitative strategy backtest")
    parser.add_argument(
        "--parquet",
        type=str,
        default="data/parquet/features_latest.parquet",
        help="Path to features Parquet file",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="data/models/stock_model.txt",
        help="Path to trained model file",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.005,
        help="Prediction threshold for buy signal (default: 0.005 = 0.5%)",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=1_000_000.0,
        help="Initial capital (default: 1,000,000)",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=10,
        help="Maximum number of concurrent positions (default: 10)",
    )
    parser.add_argument(
        "--position-size",
        type=float,
        default=0.1,
        help="Position size as fraction of capital (default: 0.1 = 10%)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/plots",
        help="Output directory for plots (default: data/plots)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plots",
    )
    
    args = parser.parse_args()
    
    # Check if input files exist
    parquet_path = Path(args.parquet)
    model_path = Path(args.model)
    
    if not parquet_path.exists():
        logger.error(f"Features file not found: {args.parquet}")
        logger.info("Please run feature pipeline first: python src/feature_pipeline.py")
        sys.exit(1)
    
    logger.info("=" * 60)
    logger.info("Quantitative Strategy Backtest")
    logger.info("=" * 60)
    logger.info(f"Features file: {args.parquet}")
    logger.info(f"Model file: {args.model}")
    logger.info(f"Initial capital: ${args.capital:,.0f}")
    logger.info(f"Buy threshold: {args.threshold:.2%}")
    logger.info(f"Max positions: {args.max_positions}")
    logger.info(f"Position size: {args.position_size:.0%}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=args.capital,
        prediction_threshold=args.threshold,
        max_positions=args.max_positions,
        position_size_pct=args.position_size,
    )
    
    # Run backtest
    try:
        results = backtester.run(
            parquet_path=args.parquet,
            model_path=args.model,
        )
    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Extract results
    records_df = results["records"]
    metrics = results["metrics"]
    equity_curve = results["equity_curve"]
    
    # Print metrics
    logger.info("=" * 60)
    logger.info("Backtest Results")
    logger.info("=" * 60)
    logger.info(f"Total Return: {metrics['total_return']:.2%}")
    logger.info(f"Annualized Return: {metrics['annualized_return']:.2%}")
    logger.info(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
    logger.info(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    logger.info(f"Calmar Ratio: {metrics.get('calmar_ratio', 0):.2f}")
    logger.info(f"Volatility: {metrics.get('volatility', 0):.2%}")
    logger.info(f"Final Value: ${metrics['final_value']:,.0f}")
    logger.info(f"Trading Days: {metrics['num_trading_days']}")
    logger.info(f"Total Trades: {metrics.get('num_trades', 0)}")
    
    # Generate visualizations
    if not args.no_plot:
        logger.info("=" * 60)
        logger.info("Generating Visualizations")
        logger.info("=" * 60)
        
        try:
            viz = Visualizer()
            
            report = viz.generate_report(
                equity_curve=equity_curve,
                trade_records=records_df,
                initial_capital=args.capital,
                save_dir=args.output,
            )
            
            logger.info(f"Plots saved to: {args.output}")
            logger.info(f"  - equity_curve.png")
            logger.info(f"  - drawdown_curve.png")
            logger.info(f"  - returns_distribution.png")
            logger.info(f"  - backtest_result.png")
            
        except Exception as e:
            logger.error(f"Failed to generate plots: {e}")
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Backtest Complete!")
    logger.info("=" * 60)
    
    # Performance rating
    if metrics['annualized_return'] > 0.2:
        rating = "EXCELLENT"
    elif metrics['annualized_return'] > 0.1:
        rating = "GOOD"
    elif metrics['annualized_return'] > 0:
        rating = "POSITIVE"
    else:
        rating = "NEEDS IMPROVEMENT"
    
    logger.info(f"Strategy Rating: {rating}")
    
    # Save metrics to file
    metrics_file = Path(args.output) / "backtest_metrics.txt"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(metrics_file, "w", encoding="utf-8") as f:
        f.write("Backtest Performance Metrics\n")
        f.write("=" * 40 + "\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"Total Return:      {metrics['total_return']:.4f} ({metrics['total_return']:.2%})\n")
        f.write(f"Annualized Return: {metrics['annualized_return']:.4f} ({metrics['annualized_return']:.2%})\n")
        f.write(f"Max Drawdown:      {metrics['max_drawdown']:.4f} ({metrics['max_drawdown']:.2%})\n")
        f.write(f"Sharpe Ratio:      {metrics['sharpe_ratio']:.4f}\n")
        f.write(f"Calmar Ratio:      {metrics.get('calmar_ratio', 0):.4f}\n")
        f.write(f"Volatility:        {metrics.get('volatility', 0):.4f} ({metrics.get('volatility', 0):.2%})\n")
        f.write(f"Final Value:       ${metrics['final_value']:,.0f}\n")
        f.write(f"Initial Capital:   ${args.capital:,.0f}\n")
        f.write(f"Trading Days:      {metrics['num_trading_days']}\n")
        f.write(f"Total Trades:      {metrics.get('num_trades', 0)}\n")
        f.write(f"Strategy Rating:   {rating}\n")
    
    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
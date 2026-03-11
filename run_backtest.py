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
    parser.add_argument(
        "--min-hold-days",
        type=int,
        default=1,
        help="Minimum holding days before selling (default: 1)",
    )
    parser.add_argument(
        "--min-avg-return-to-enter",
        type=float,
        default=0.01,
        help="Minimum average predicted return to open position (default: 1%)",
    )
    parser.add_argument(
        "--min-prediction-diff",
        type=float,
        default=0.005,
        help="Minimum prediction difference for turnover (default: 0.5%)",
    )
    parser.add_argument(
        "--stop-loss",
        type=float,
        default=-0.02,
        help="Stop loss threshold (default: -2%)",
    )
    parser.add_argument(
        "--no-stop-loss",
        action="store_true",
        help="Disable stop loss",
    )
    parser.add_argument(
        "--turnover-control",
        action="store_true",
        default=True,
        help="Enable turnover control (default: True)",
    )
    parser.add_argument(
        "--no-turnover-control",
        action="store_true",
        help="Disable turnover control",
    )
    parser.add_argument(
        "--dynamic-position",
        action="store_true",
        default=False,
        help="Use dynamic position sizing based on prediction scores",
    )
    parser.add_argument(
        "--max-single-position",
        type=float,
        default=0.3,
        help="Maximum single position weight (default: 0.3 = 30%)",
    )
    
    args = parser.parse_args()
    
    # Handle turnover control flag
    if args.no_turnover_control:
        args.turnover_control = False
    
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
    logger.info(f"Min hold days: {args.min_hold_days}")
    logger.info(f"Min avg return to enter: {args.min_avg_return_to_enter:.2%}")
    logger.info(f"Min prediction diff: {args.min_prediction_diff:.2%}")
    logger.info(f"Turnover control: {args.turnover_control}")
    logger.info(f"Dynamic position: {args.dynamic_position}")
    logger.info(f"Max single position: {args.max_single_position:.0%}")
    logger.info(f"Output directory: {args.output}")
    logger.info("=" * 60)
    
    # Initialize backtester
    backtester = Backtester(
        initial_capital=args.capital,
        prediction_threshold=args.threshold,
        max_positions=args.max_positions,
        position_size_pct=args.position_size,
        min_hold_days=args.min_hold_days,
        min_avg_return_to_enter=args.min_avg_return_to_enter,
        min_prediction_diff=args.min_prediction_diff,
        stop_loss_threshold=args.stop_loss,
        enable_stop_loss=not args.no_stop_loss,
        turnover_control=args.turnover_control,
        use_dynamic_position=args.dynamic_position,
        max_single_position=args.max_single_position,
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
    logger.info(f"Win Rate: {metrics.get('win_rate', 0):.2%}")
    logger.info(f"Profit Factor: {metrics.get('profit_factor', 0):.2f}")
    logger.info(f"Stop Loss Triggered: {metrics.get('stop_loss_count', 0)} times")
    
    # Cost analysis
    if 'cost_analysis' in results:
        cost_info = results['cost_analysis']
        logger.info(f"Total Transaction Cost: ${cost_info['total_cost']:,.2f}")
        logger.info(f"Cost Ratio: {cost_info['cost_ratio']*100:.3f}% of initial capital")
    
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
        f.write(f"Win Rate:          {metrics.get('win_rate', 0):.2%}\n")
        f.write(f"Profit Factor:     {metrics.get('profit_factor', 0):.2f}\n")
        f.write(f"Stop Loss Count:   {metrics.get('stop_loss_count', 0)}\n")
        f.write(f"Strategy Rating:   {rating}\n")
        
        # Add cost analysis if available
        if 'cost_analysis' in results:
            cost_info = results['cost_analysis']
            f.write(f"\nTransaction Cost Analysis:\n")
            f.write(f"  Total Cost:        ${cost_info['total_cost']:,.2f}\n")
            f.write(f"  Cost Ratio:        {cost_info['cost_ratio']*100:.3f}%\n")
    
    logger.info(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
"""
V20 Backtest Runner - 铁血实盘会计逻辑回测

【使用说明】
1. 从数据库获取股票数据和信号
2. 使用 V20 会计引擎执行真实回测
3. 生成 V20 铁血报告

【核心特性】
- 真实账户管理：现金、持仓、冻结资金
- 真实手续费计算：滑点 + 规费 + 印花税 + 最低佣金
- T+1 锁定：今日买入明日才能卖
- 持仓生命周期：止损、止盈、信号排名

作者：资深量化交易系统专家
版本：V20.0
日期：2026-03-17
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

import polars as pl
import numpy as np
from loguru import logger

try:
    from .db_manager import DatabaseManager
    from .v20_accounting_engine import (
        V20AccountingEngine,
        V20BacktestExecutor,
        V20PositionManager,
        V20MarketRegimeFilter,
        BacktestResult,
        generate_v20_report,
        INITIAL_CAPITAL,
        TOP_K_STOCKS,
    )
except ImportError:
    from db_manager import DatabaseManager
    from v20_accounting_engine import (
        V20AccountingEngine,
        V20BacktestExecutor,
        V20PositionManager,
        V20MarketRegimeFilter,
        BacktestResult,
        generate_v20_report,
        INITIAL_CAPITAL,
        TOP_K_STOCKS,
    )


# ===========================================
# V20 信号生成器 - 整合 V19 因子体系
# ===========================================

class V20SignalGenerator:
    """
    V20 信号生成器 - 基于 V19 因子体系
    
    【因子库】
    1. 量价相关性 (vol_price_corr) - 最强因子
    2. 短线反转 (reversal_st) - 正交化
    3. 波动风险 (vol_risk) - 低波异常
    4. 异常换手 (turnover_signal)
    5. 动量因子 (momentum) - V19 新增
    6. 低波动因子 (low_vol) - V19 新增
    """
    
    EPSILON = 1e-6
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db or DatabaseManager.get_instance()
        logger.info("V20SignalGenerator initialized")
    
    def compute_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算所有因子"""
        result = df.clone().with_columns([
            pl.col("close").cast(pl.Float64, strict=False),
            pl.col("volume").cast(pl.Float64, strict=False),
            pl.col("turnover_rate").cast(pl.Float64, strict=False),
        ])
        
        # 1. 量价相关性
        volume_change = (pl.col("volume") / pl.col("volume").shift(1) - 1).fill_null(0)
        returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        
        window = 20
        vol_mean = volume_change.rolling_mean(window_size=window).shift(1)
        ret_mean = returns.rolling_mean(window_size=window).shift(1)
        cov_xy = ((volume_change - vol_mean) * (returns - ret_mean)).rolling_mean(window_size=window).shift(1)
        vol_std = volume_change.rolling_std(window_size=window, ddof=1).shift(1)
        ret_std = returns.rolling_std(window_size=window, ddof=1).shift(1)
        vol_price_corr = cov_xy / (vol_std * ret_std + self.EPSILON)
        
        result = result.with_columns([
            vol_price_corr.alias("vol_price_corr"),
        ])
        
        # 2. 短线反转
        momentum_5 = pl.col("close") / (pl.col("close").shift(6) + self.EPSILON) - 1
        reversal_st = -momentum_5.shift(1)
        result = result.with_columns([reversal_st.alias("reversal_st")])
        
        # 3. 波动风险
        stock_returns = (pl.col("close") / pl.col("close").shift(1) - 1).fill_null(0)
        volatility_20 = stock_returns.rolling_std(window_size=20, ddof=1).shift(1) * np.sqrt(252)
        vol_risk = -volatility_20
        result = result.with_columns([vol_risk.alias("vol_risk")])
        
        # 4. 异常换手
        turnover_ma20 = pl.col("turnover_rate").rolling_mean(window_size=20).shift(1)
        turnover_ratio = pl.col("turnover_rate") / (turnover_ma20 + self.EPSILON)
        turnover_signal = (turnover_ratio - 1).clip(-0.9, 2.0)
        result = result.with_columns([turnover_signal.alias("turnover_signal")])
        
        # 5. 动量因子
        ma20 = pl.col("close").rolling_mean(window_size=20).shift(1)
        momentum = (pl.col("close") - ma20) / (ma20 + self.EPSILON)
        momentum_signal = -momentum
        result = result.with_columns([
            ma20.alias("ma20"),
            momentum_signal.alias("momentum"),
        ])
        
        # 6. 低波动因子
        std_20d = stock_returns.rolling_std(window_size=20, ddof=1).shift(1) * np.sqrt(252)
        low_vol = -std_20d
        result = result.with_columns([low_vol.alias("low_vol")])
        
        logger.info(f"Computed 6 factors for {len(result)} records")
        return result
    
    def normalize_factors(self, df: pl.DataFrame, factor_names: List[str]) -> pl.DataFrame:
        """截面标准化"""
        result = df.clone()
        
        for factor in factor_names:
            if factor not in result.columns:
                continue
            
            # 计算每日均值和标准差
            stats = result.group_by("trade_date").agg([
                pl.col(factor).mean().alias("mean"),
                pl.col(factor).std().alias("std"),
            ])
            
            result = result.join(stats, on="trade_date", how="left")
            result = result.with_columns([
                ((pl.col(factor) - pl.col("mean")) / (pl.col("std") + self.EPSILON)).alias(f"{factor}_std")
            ]).drop(["mean", "std"])
        
        return result
    
    def compute_composite_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算综合信号"""
        factor_names = ["vol_price_corr", "reversal_st", "vol_risk", "turnover_signal", "momentum", "low_vol"]
        std_factors = [f"{f}_std" for f in factor_names if f"{f}_std" in df.columns]
        
        if not std_factors:
            # 如果标准化失败，使用原始因子
            std_factors = factor_names
        
        # 等权重综合信号
        result = df.with_columns([
            (
                sum(pl.col(f) for f in std_factors) / len(std_factors)
            ).alias("composite_signal")
        ])
        
        return result
    
    def generate_signals(
        self,
        start_date: str,
        end_date: str,
    ) -> pl.DataFrame:
        """生成信号"""
        logger.info(f"Generating signals from {start_date} to {end_date}...")
        
        # 从数据库获取数据
        query = f"""
            SELECT 
                symbol, trade_date, open, high, low, close, pre_close,
                volume, amount, turnover_rate
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        
        df = self.db.read_sql(query)
        
        if df.is_empty():
            logger.error("No data found")
            return pl.DataFrame()
        
        logger.info(f"Loaded {len(df)} records, {df['symbol'].n_unique()} stocks")
        
        # 计算因子
        df = self.compute_factors(df)
        
        # 标准化因子
        factor_names = ["vol_price_corr", "reversal_st", "vol_risk", "turnover_signal", "momentum", "low_vol"]
        df = self.normalize_factors(df, factor_names)
        
        # 计算综合信号
        df = self.compute_composite_signal(df)
        
        # 提取信号
        signals = df.select(["symbol", "trade_date", "composite_signal"]).rename({
            "composite_signal": "signal"
        })
        
        logger.info(f"Generated signals for {len(signals)} records")
        return signals
    
    def get_prices(self, start_date: str, end_date: str) -> pl.DataFrame:
        """获取价格数据"""
        query = f"""
            SELECT symbol, trade_date, close
            FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY symbol, trade_date
        """
        return self.db.read_sql(query)
    
    def get_index_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """获取指数数据"""
        query = f"""
            SELECT symbol, trade_date, open, high, low, close, pre_close
            FROM stock_daily
            WHERE symbol = '000300.SH'
            AND trade_date >= '{start_date}'
            AND trade_date <= '{end_date}'
            ORDER BY trade_date
        """
        return self.db.read_sql(query)


# ===========================================
# V20 回测运行器
# ===========================================

class V20BacktestRunner:
    """
    V20 回测运行器 - 整合信号生成和会计引擎
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        top_k_stocks: int = TOP_K_STOCKS,
        db: Optional[DatabaseManager] = None,
    ):
        self.initial_capital = initial_capital
        self.top_k_stocks = top_k_stocks
        self.db = db or DatabaseManager.get_instance()
        self.signal_generator = V20SignalGenerator(db=self.db)
        
        logger.info(f"V20BacktestRunner initialized")
        logger.info(f"  Initial Capital: {initial_capital:,.0f}")
        logger.info(f"  Top K Stocks: {top_k_stocks}")
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
    ) -> BacktestResult:
        """运行回测"""
        logger.info("=" * 70)
        logger.info("V20 BACKTEST RUNNER")
        logger.info("=" * 70)
        logger.info(f"Period: {start_date} to {end_date}")
        
        # 1. 生成信号
        logger.info("\nStep 1: Generating signals...")
        signals_df = self.signal_generator.generate_signals(start_date, end_date)
        
        if signals_df.is_empty():
            logger.error("Signal generation failed")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
            )
        
        # 2. 获取价格数据
        logger.info("\nStep 2: Getting price data...")
        prices_df = self.signal_generator.get_prices(start_date, end_date)
        
        if prices_df.is_empty():
            logger.error("Price data not found")
            return BacktestResult(
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital,
            )
        
        logger.info(f"Loaded {len(prices_df)} price records")
        
        # 3. 获取指数数据（用于市场熔断）
        logger.info("\nStep 3: Getting index data...")
        index_df = self.signal_generator.get_index_data(start_date, end_date)
        
        if index_df.is_empty():
            logger.warning("Index data not found, skipping market regime filter")
            index_df = None
        else:
            logger.info(f"Loaded {len(index_df)} index records")
        
        # 4. 运行回测
        logger.info("\nStep 4: Running V20 backtest...")
        executor = V20BacktestExecutor(
            initial_capital=self.initial_capital,
            top_k_stocks=self.top_k_stocks,
            db=self.db,
        )
        
        result = executor.run_backtest(
            signals_df=signals_df,
            prices_df=prices_df,
            index_df=index_df,
        )
        
        return result
    
    def generate_report(self, result: BacktestResult) -> str:
        """生成报告"""
        report_path = generate_v20_report(result)
        return report_path


# ===========================================
# 主函数
# ===========================================

def main():
    """主入口函数"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )
    
    logger.info("=" * 70)
    logger.info("V20 Iron Blood Backtest Runner")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 默认参数
    start_date = "2025-01-01"
    end_date = "2026-03-31"
    initial_capital = INITIAL_CAPITAL
    top_k_stocks = TOP_K_STOCKS
    
    # 解析命令行参数（如果提供）
    if len(sys.argv) > 1:
        start_date = sys.argv[1]
    if len(sys.argv) > 2:
        end_date = sys.argv[2]
    if len(sys.argv) > 3:
        initial_capital = float(sys.argv[3])
    if len(sys.argv) > 4:
        top_k_stocks = int(sys.argv[4])
    
    logger.info(f"Parameters:")
    logger.info(f"  Start Date: {start_date}")
    logger.info(f"  End Date: {end_date}")
    logger.info(f"  Initial Capital: {initial_capital:,.0f}")
    logger.info(f"  Top K Stocks: {top_k_stocks}")
    
    # 运行回测
    runner = V20BacktestRunner(
        initial_capital=initial_capital,
        top_k_stocks=top_k_stocks,
    )
    
    result = runner.run_backtest(start_date, end_date)
    
    # 生成报告
    logger.info("\nStep 5: Generating report...")
    report_path = runner.generate_report(result)
    
    # 保存 JSON 结果
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = output_dir / f"V20_backtest_result_{timestamp}.json"
    
    # 序列化结果
    result_dict = {
        "start_date": result.start_date,
        "end_date": result.end_date,
        "initial_capital": result.initial_capital,
        "total_return": result.total_return,
        "annual_return": result.annual_return,
        "sharpe_ratio": result.sharpe_ratio,
        "max_drawdown": result.max_drawdown,
        "win_rate": result.win_rate,
        "total_trades": result.total_trades,
        "total_buy_amount": result.total_buy_amount,
        "total_sell_amount": result.total_sell_amount,
        "total_commission": result.total_commission,
        "total_stamp_duty": result.total_stamp_duty,
        "total_slippage": result.total_slippage,
        "turnover_rate": result.turnover_rate,
        "avg_holding_days": result.avg_holding_days,
        "avg_position_count": result.avg_position_count,
        "report_path": report_path,
    }
    
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2, ensure_ascii=False)
    
    logger.info(f"\nJSON result saved to: {json_path}")
    
    logger.info("\n" + "=" * 70)
    logger.info("V20 BACKTEST COMPLETE")
    logger.info("=" * 70)
    logger.info(f"Report Path: {report_path}")
    logger.info(f"JSON Path: {json_path}")
    
    # 输出最终结果
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULT SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.total_return:.2%}")
    logger.info(f"Annual Return: {result.annual_return:.2%}")
    logger.info(f"Sharpe Ratio: {result.sharpe_ratio:.3f}")
    logger.info(f"Max Drawdown: {result.max_drawdown:.2%}")
    logger.info(f"Total Trades: {result.total_trades}")
    logger.info(f"Total Commission: {result.total_commission:,.2f} 元")
    logger.info(f"Turnover Rate: {result.turnover_rate:.2%}")


if __name__ == "__main__":
    main()
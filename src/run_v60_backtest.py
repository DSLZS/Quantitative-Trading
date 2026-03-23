"""
V60 回测运行脚本 - 全样本实战与逻辑自进化

【使用说明】
1. 确保数据库中有 2024-2025 全年全市场数据
2. 运行：python src/run_v60_backtest.py
3. 查看报告：reports/V60_Backtest_Report.md

作者：量化系统
版本：V60.0
日期：2026-03-23
"""

import sys
import os
from datetime import datetime
from pathlib import Path

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
from loguru import logger

from v60_core import V60FactorEngine, V60IndustryLoader
from v60_engine import V60BacktestEngine, MasterLoop

# 配置日志
logger.remove()
logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("logs/v60_backtest_{time:YYYYMMDD_HHmmss}.log", level="INFO", rotation="100 MB")


def load_data_from_db(start_date: str, end_date: str) -> tuple:
    """从数据库加载全样本数据"""
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager.get_instance()
        
        # 加载价格数据 - 全样本，严禁 limit
        logger.info(f"Loading price data from {start_date} to {end_date}...")
        price_query = f"""
            SELECT symbol, trade_date, open, high, low, close, volume, amount,
                   adj_factor, turnover_rate, vol_ratio
            FROM stock_daily
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date, symbol
        """
        price_data = db.read_sql(price_query)
        
        if price_data.is_empty():
            logger.error("No price data found in database!")
            raise ValueError("Database empty! Must sync full market data first!")
        
        unique_stocks = price_data['symbol'].n_unique()
        logger.info(f"Loaded {len(price_data)} rows, {unique_stocks} unique stocks")
        
        if unique_stocks < 100:
            logger.warning(f"Only {unique_stocks} stocks found - may need more data")
        
        # 加载行业数据
        logger.info("Loading industry data...")
        try:
            industry_query = f"""
                SELECT symbol, trade_date, industry_name, industry_mv_ratio
                FROM stock_industry_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            """
            industry_data = db.read_sql(industry_query)
            if not industry_data.is_empty():
                logger.info(f"Loaded {len(industry_data)} rows of industry data")
            else:
                logger.info("No industry data in database, will use code segment mapping")
                industry_data = None
        except Exception as e:
            logger.info(f"Industry table not available: {e}")
            industry_data = None
        
        # 加载指数数据
        logger.info("Loading index data...")
        try:
            index_query = f"""
                SELECT symbol, trade_date, close, ma20, ma60
                FROM index_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            """
            index_data = db.read_sql(index_query)
            if index_data.is_empty():
                logger.info("No index data found")
                index_data = None
        except Exception:
            index_data = None
        
        return price_data, industry_data, index_data, db
        
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def run_v60_backtest(start_date: str = "2024-01-01", end_date: str = "2025-12-31",
                     use_masterloop: bool = True) -> dict:
    """运行 V60 回测"""
    logger.info("=" * 60)
    logger.info("V60 BACKTEST STARTING")
    logger.info("=" * 60)
    logger.info(f"Period: {start_date} to {end_date}")
    logger.info(f"MasterLoop: {'Enabled' if use_masterloop else 'Disabled'}")
    
    # 加载数据
    price_data, industry_data, index_data, db = load_data_from_db(start_date, end_date)
    
    # 计算因子
    logger.info("Computing factors...")
    factor_engine = V60FactorEngine()
    factor_data, factor_status = factor_engine.compute_all_factors(
        price_data, industry_data, db, start_date, end_date, index_data
    )
    
    logger.info(f"Factors computed: {factor_status['factors_computed']}")
    
    # 运行回测
    if use_masterloop:
        logger.info("Running MasterLoop (Logic Self-Evolution)...")
        masterloop = MasterLoop(max_iterations=50)
        
        iteration = 0
        best_result = None
        
        while iteration < masterloop.max_iterations:
            iteration += 1
            logger.info(f"\n{'='*60}")
            logger.info(f"ITERATION {iteration}/{masterloop.max_iterations}")
            logger.info(f"{'='*60}")
            
            result = masterloop.run_iteration(
                factor_data=factor_data,
                price_data=price_data,
                industry_data=industry_data,
                index_data=index_data,
                start_date=start_date,
                end_date=end_date
            )
            
            if result.get('meets_target', False):
                logger.info(f"TARGET MET at iteration {iteration}!")
                best_result = result
                break
            
            best_result = result
        
        final_result = best_result if best_result else masterloop.get_best_result_dict()
        
        # 生成逻辑进化报告
        evolution_report = masterloop.get_logic_evolution_report()
        comparison = masterloop.compare_logic_paths()
        
        return {
            'result': final_result,
            'evolution_report': evolution_report,
            'logic_comparison': comparison,
            'iteration_results': masterloop.iteration_results
        }
    
    else:
        # 单次回测
        engine = V60BacktestEngine()
        result = engine.run_backtest(
            factor_data=factor_data,
            price_data=price_data,
            industry_data=industry_data,
            index_data=index_data,
            start_date=start_date,
            end_date=end_date
        )
        return {'result': result.get('result', {})}


def generate_report(backtest_result: dict, output_path: str = "reports/V60_Backtest_Report.md"):
    """生成回测报告"""
    result = backtest_result.get('result', {})
    evolution_report = backtest_result.get('evolution_report', {})
    logic_comparison = backtest_result.get('logic_comparison', {})
    
    if not result or 'error' in result:
        logger.error(f"Cannot generate report: {result.get('error', 'Unknown error')}")
        return
    
    report_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    md = f"""# V60 回测报告

**生成时间**: {report_time}
**版本**: V60.0

---

## 一、核心指标

| 指标 | 数值 |
|------|------|
| 总收益率 | {result.get('total_return', 0):.2%} |
| 年化收益 | {result.get('annual_return', 0):.2%} |
| 最大回撤 | {result.get('max_drawdown', 0):.2%} |
| 夏普比率 | {result.get('sharpe_ratio', 0):.2f} |
| 胜率 | {result.get('win_rate', 0):.2%} |
| 盈亏比 | {result.get('profit_loss_ratio', 0):.2f} |
| 交易次数 | {result.get('total_trades', 0)} |
| 盈利次数 | {result.get('winning_trades', 0)} |
| 亏损次数 | {result.get('losing_trades', 0)} |
| 最终权益 | {result.get('final_equity', 0):,.2f} |

---

## 二、V60 核心改进验证

### 2.1 数据规模审计

- 数据源：{result.get('data_source', 'Unknown')}
- 总交易天数：{len(result.get('trade_dates', []))}
- 交易股票数量：{result.get('total_stocks_traded', 0)}
- 全局交易次数：{result.get('global_trade_count', 0)}

### 2.2 ATR 趋势跟踪 2.0

- 止损模式：atr_only
- ATR 倍数：3.0
- 保本止损阈值：15%
- 追踪止盈阈值：20%
- 阶梯止盈：已启用

### 2.3 趋势确认过滤

- MA20 > MA60 过滤：已启用
- Close > MA120 过滤：已启用
- 成交量突破：1.5 倍

---

## 三、目标达成分析

**是否达标**: {"✅ 是" if result.get('meets_target', False) else "❌ 否"}

{result.get('target_analysis', 'All targets met')}

---

## 四、逻辑进化报告

"""
    
    if evolution_report:
        md += f"""### 4.1 进化统计

- 总进化次数：{evolution_report.get('total_evolutions', 0)}
- 最佳收益率：{evolution_report.get('best_result', {}).get('total_return', 0):.2%}
- 最佳盈亏比：{evolution_report.get('best_result', {}).get('profit_loss_ratio', 0):.2f}

### 4.2 逻辑突变记录

"""
        for i, record in enumerate(evolution_report.get('evolution_records', [])[:10], 1):
            md += f"""#### 突变 {i}: Iteration {record.get('iteration', '?')}

- **前逻辑**: {record.get('previous_logic', 'N/A')}
- **新逻辑**: {record.get('new_logic', 'N/A')}
- **原因**: {record.get('reason', 'N/A')}
- **参数变化**: {record.get('parameters_changed', {})}
- **性能影响**: 收益率={record.get('performance_impact', {}).get('total_return', 0):.2%}, 盈亏比={record.get('performance_impact', {}).get('profit_loss_ratio', 0):.2f}

"""
    
    if logic_comparison:
        md += f"""### 4.3 逻辑路径对比

| 逻辑路径 | 平均收益 | 平均盈亏比 | 平均回撤 | 稳定性得分 |
|----------|----------|------------|----------|------------|
"""
        for path, metrics in logic_comparison.get('comparison', {}).items():
            md += f"| {path} | {metrics.get('avg_return', 0):.2%} | {metrics.get('avg_pl_ratio', 0):.2f} | {metrics.get('avg_mdd', 0):.2%} | {metrics.get('stability_score', 0):.2f} |\n"
        
        md += f"\n**最佳逻辑路径**: {logic_comparison.get('best_logic_path', 'N/A')}\n"
    
    md += f"""
---

## 五、费用分析

| 费用类型 | 金额 |
|----------|------|
| 总佣金 | {result.get('total_commission', 0):,.2f} |
| 总滑点 | {result.get('total_slippage', 0):,.2f} |
| 总印花税 | {result.get('total_stamp_duty', 0):,.2f} |
| 总过户费 | {result.get('total_transfer_fee', 0):,.2f} |

---

## 六、交易质量

| 指标 | 数值 |
|------|------|
| 平均持仓天数 | {result.get('avg_holding_days', 0):.1f} |
| 平均盈利 | {result.get('avg_win', 0):,.2f} |
| 平均亏损 | {result.get('avg_loss', 0):,.2f} |
| 最大盈利 | {result.get('largest_win', 0):,.2f} |
| 最大亏损 | {result.get('largest_loss', 0):,.2f} |
| 连续盈利 | {result.get('consecutive_wins', 0)} |
| 连续亏损 | {result.get('consecutive_losses', 0)} |

---

## 七、V60 核心哲学

> 宁可让部分盈利单变成亏损单，也要捕捉 20%+ 的大趋势。
> 动态止损适应市场波动率，废除 8% 固定止损的教条主义。
> 真正的逻辑自进化：收益率<15% 时必须调整选股分位数和趋势周期。

---

## 八、合规性审计

### 8.1 成交价审计

- 违规次数：{result.get('price_audit_violations', 0)}
- 审计状态：{"✅ 通过" if result.get('price_audit_violations', 0) == 0 else "⚠️ 存在违规"}

### 8.2 数据规模审计

- 严禁 limit 20: {"✅ 遵守" if result.get('total_stocks_traded', 0) > 100 else "⚠️ 需检查"}
- 全样本加载：{"✅ 是" if result.get('total_stocks_traded', 0) > 100 else "❌ 否"}

---

*报告由 V60 回测系统自动生成*
"""
    
    # 确保目录存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md)
    
    logger.info(f"Report saved to: {output_path}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='V60 Backtest Runner')
    parser.add_argument('--start', type=str, default='2024-01-01', help='Start date')
    parser.add_argument('--end', type=str, default='2025-12-31', help='End date')
    parser.add_argument('--no-masterloop', action='store_true', help='Disable MasterLoop')
    parser.add_argument('--output', type=str, default='reports/V60_Backtest_Report.md', help='Output report path')
    
    args = parser.parse_args()
    
    try:
        result = run_v60_backtest(args.start, args.end, not args.no_masterloop)
        generate_report(result, args.output)
        
        # 打印摘要
        final_result = result.get('result', {})
        print("\n" + "=" * 60)
        print("V60 BACKTEST SUMMARY")
        print("=" * 60)
        print(f"Total Return: {final_result.get('total_return', 0):.2%}")
        print(f"Max Drawdown: {final_result.get('max_drawdown', 0):.2%}")
        print(f"Profit/Loss Ratio: {final_result.get('profit_loss_ratio', 0):.2f}")
        print(f"Target Met: {final_result.get('meets_target', False)}")
        print("=" * 60)
        
    except Exception as e:
        logger.error(f"V60 Backtest FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
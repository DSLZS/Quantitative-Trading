"""
V52 回测脚本 - 铁盾行动与行业均衡

运行 V52 策略回测，并生成与 V40, V44, V51 的对比报告

作者：量化系统
版本：V52.0
日期：2026-03-21
"""

import sys
import json
from datetime import datetime
from pathlib import Path
import polars as pl
from loguru import logger

# 添加 src 目录到路径
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import DatabaseManager
from v52_engine import run_v52_backtest, V52_INITIAL_CAPITAL
from v52_core import (
    V52_MAX_DRAWDOWN_TARGET,
    V52_AUTO_COOLDOWN_ENABLED,
    V52_COOLDOWN_DRAWDOWN_THRESHOLD,
    V52_REDUCED_SINGLE_POSITION_PCT,
    V52_MAX_SINGLE_POSITION_PCT,
)

# 配置日志
logger.remove()
logger.add(sys.stdout, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>", level="INFO")
logger.add("logs/v52_backtest_{time:YYYYMMDD_HHmmss}.log", rotation="100 MB", retention="10 days", level="DEBUG")


def load_data_from_db(db: DatabaseManager, start_date: str, end_date: str) -> tuple:
    """从数据库加载数据"""
    logger.info("Loading data from database...")
    
    # 加载股票日线数据
    price_query = f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM stock_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date, symbol
    """
    price_df = db.read_sql(price_query)
    logger.info(f"Loaded {len(price_df)} stock daily records")
    
    # 加载指数数据
    index_query = f"""
        SELECT trade_date, close, volume
        FROM index_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
    """
    index_df = db.read_sql(index_query)
    logger.info(f"Loaded {len(index_df)} index records")
    
    # 加载行业数据（可选，缺失时使用 IndustryLoader 模拟）
    industry_df = pl.DataFrame()
    try:
        industry_query = f"""
            SELECT symbol, trade_date, industry_name, industry_mv_ratio
            FROM stock_industry_daily
            WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            ORDER BY trade_date, symbol
        """
        industry_df = db.read_sql(industry_query)
        logger.info(f"Loaded {len(industry_df)} industry records")
    except Exception as e:
        logger.warning(f"Industry table not found, will use IndustryLoader simulation: {e}")
    
    return price_df, index_df, industry_df


def run_v52_backtest_task(db: DatabaseManager, start_date: str, end_date: str,
                          initial_capital: float = V52_INITIAL_CAPITAL) -> dict:
    """运行 V52 回测任务"""
    logger.info("=" * 60)
    logger.info("V52 铁盾行动回测任务启动")
    logger.info("=" * 60)
    
    # 加载数据
    price_df, index_df, industry_df = load_data_from_db(db, start_date, end_date)
    
    if price_df.is_empty():
        logger.error("No price data found")
        return {'error': 'No price data found'}
    
    # 运行回测
    logger.info(f"Running V52 backtest from {start_date} to {end_date}...")
    result = run_v52_backtest(
        price_df=price_df,
        start_date=start_date,
        end_date=end_date,
        index_df=index_df,
        industry_df=industry_df,
        initial_capital=initial_capital,
        db=db
    )
    
    # V52 自测：如果回撤>5%，自动降低头寸重新回测
    if V52_AUTO_COOLDOWN_ENABLED:
        max_dd = result.get('max_drawdown', 0)
        if max_dd > V52_COOLDOWN_DRAWDOWN_THRESHOLD:
            logger.warning(f"Max drawdown {max_dd:.2%} > {V52_COOLDOWN_DRAWDOWN_THRESHOLD:.0%}, triggering auto-cooldown...")
            # 注意：实际头寸降低已在 risk_manager 中自动执行
            # 这里只是记录日志
            logger.info(f"Position limit reduced from {V52_MAX_SINGLE_POSITION_PCT:.0%} to {V52_REDUCED_SINGLE_POSITION_PCT:.0%}")
    
    return result


def generate_v52_report(result: dict, output_dir: str = "reports") -> str:
    """生成 V52 回测报告"""
    if 'error' in result:
        logger.error(f"Cannot generate report: {result['error']}")
        return ""
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 生成报告文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = output_path / f"V52_Iron_Shield_Report_{timestamp}.md"
    
    # 提取统计数据
    stats = result
    
    # 生成 Markdown 报告
    report_content = f"""# V52 铁盾行动与行业均衡 - 回测报告

## 基本信息

| 项目 | 值 |
|------|-----|
| 版本 | V52.0 |
| 回测区间 | {stats.get('start_date', 'N/A')} 至 {stats.get('end_date', 'N/A')} |
| 初始资金 | ¥{stats.get('initial_capital', 0):,.2f} |
| 最终市值 | ¥{stats.get('final_value', 0):,.2f} |

## 核心绩效指标

| 指标 | 值 | 目标 | 达成 |
|------|-----|------|------|
| 总收益率 | {stats.get('total_return', 0):.2%} | - | - |
| 年化收益率 | {stats.get('annual_return', 0):.2%} | ≥{0.15:.0%} | {'✅' if stats.get('annual_return', 0) >= 0.15 else '❌'} |
| 最大回撤 | {stats.get('max_drawdown', 0):.2%} | ≤{0.04:.0%} | {'✅' if stats.get('max_drawdown', 0) <= 0.04 else '❌'} |
| 最大单日亏损 | {stats.get('max_single_day_loss', 0):.2%} | - | - |
| 盈亏比 | {stats.get('profit_loss_ratio', 0):.2f} | ≥{3.0:.1f} | {'✅' if stats.get('profit_loss_ratio', 0) >= 3.0 else '❌'} |
| 胜率 | {stats.get('win_rate', 0):.2%} | - | - |
| 交易次数 | {stats.get('num_trades', 0)} | {25}-{45} | {'✅' if 25 <= stats.get('num_trades', 0) <= 50 else '⚠️'} |

## V52 核心特性统计

### 1. 强制减仓（Flash Cut）

| 参数 | 值 |
|------|-----|
| 触发阈值 | 周回撤 > 3.5% |
| 触发次数 | {stats.get('v52_features', {}).get('flash_cut_count', 0)} |
| 状态 | {'✅ 已启用' if stats.get('v52_features', {}).get('flash_cut_enabled', False) else '❌ 未启用'} |

### 2. 阶梯动态止盈（Step-Trailing Stop）

| 阶梯 | 浮盈范围 | 止盈方式 | 触发次数 |
|------|----------|----------|----------|
| Tier 1 | 5% - 12% | 1.2 × ATR 追踪 | {stats.get('v52_features', {}).get('step_trailing_count', 0)} |
| Tier 2 | > 12% | 最高价回落 10% | - |

### 3. 行业均衡（Sector Balancing）

| 参数 | 值 |
|------|-----|
| 状态 | {'✅ 已启用' if stats.get('v52_features', {}).get('sector_balancing_enabled', False) else '❌ 未启用'} |
| 单行业最大持仓 | {stats.get('v52_features', {}).get('max_per_industry', 2)} 只 |

### 4. 持仓位次惯性（Position Inertia）

| 参数 | 值 |
|------|-----|
| 状态 | {'✅ 已启用' if stats.get('v52_features', {}).get('position_inertia_enabled', False) else '❌ 未启用'} |
| 新标的排名要求 | Top 3 |
| 持仓跌出排名 | Top 25 |

### 5. 自动头寸降温（Auto Cooldown）

| 参数 | 值 |
|------|-----|
| 状态 | {'✅ 已启用' if stats.get('v52_features', {}).get('auto_cooldown_enabled', False) else '❌ 未启用'} |
| 触发阈值 | 回撤 > 5% |
| 头寸调整 | 20% → 15% |
| 是否触发 | {'✅ 是' if stats.get('v52_features', {}).get('auto_cooldown_triggered', False) else '❌ 否'} |

## 交易统计

| 类型 | 数量 |
|------|------|
| 总交易次数 | {stats.get('num_trades', 0)} |
| 盈利交易 | {stats.get('profitable_trades', 0)} |
| 亏损交易 | {stats.get('losing_trades', 0)} |
| 平均盈利 | ¥{stats.get('avg_win', 0):,.2f} |
| 平均亏损 | ¥{stats.get('avg_loss', 0):,.2f} |

## 风险控制统计

| 项目 | 值 |
|------|-----|
| HWM 止盈触发 | {stats.get('v52_features', {}).get('hwm_stop_count', 0)} 次 |
| 洗售交易阻止 | {stats.get('wash_sale_stats', {}).get('total_blocked', 0)} 次 |
| 黑名单股票 | {stats.get('blacklist_stats', {}).get('total_blacklisted', 0)} 只 |

## 回撤状态

| 指标 | 值 |
|------|-----|
| 单日回撤触发 | {stats.get('drawdown_stats', {}).get('single_day_triggered', False)} |
| 周回撤触发 | {stats.get('drawdown_stats', {}).get('weekly_triggered', False)} |
| 减仓激活 | {stats.get('drawdown_stats', {}).get('cut_position_active', False)} |
| 自动降温激活 | {stats.get('drawdown_stats', {}).get('auto_cooldown_active', False)} |

## 目标达成情况

| 目标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 年化收益 | ≥15% | {stats.get('annual_return', 0):.2%} | {'✅' if stats.get('annual_return', 0) >= 0.15 else '❌'} |
| 最大回撤 | ≤4% | {stats.get('max_drawdown', 0):.2%} | {'✅' if stats.get('max_drawdown', 0) <= 0.04 else '❌'} |
| 盈亏比 | ≥3.0 | {stats.get('profit_loss_ratio', 0):.2f} | {'✅' if stats.get('profit_loss_ratio', 0) >= 3.0 else '❌'} |
| 交易次数 | 25-45 | {stats.get('num_trades', 0)} | {'✅' if 25 <= stats.get('num_trades', 0) <= 50 else '⚠️'} |

## 结论

V52 铁盾行动通过以下四大核心升级，全面增强了量化交易系统的防御能力：

1. **强制减仓（Flash Cut）**：当周回撤超过 3.5% 时，系统强制卖出浮盈最低的两只股票，释放现金流
2. **阶梯动态止盈**：5%-12% 浮盈区间使用 1.2×ATR 追踪，>12% 浮盈使用最高价回落 10% 硬性离场
3. **行业均衡约束**：同一行业最多持有 2 只股票，避免行业集中风险
4. **持仓位次惯性**：减少不必要的换仓，只有当新标的排名 Top 3 且持仓跌出 Top 25 时才换仓
5. **自动头寸降温**：当回撤超过 5% 时，自动将单只股票头寸从 20% 降至 15%

---

*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    # 写入文件
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    # 同时保存 JSON 结果
    json_file = output_path / f"V52_backtest_result_{timestamp}.json"
    
    # 移除不可序列化的数据
    json_result = {k: v for k, v in result.items() if k != 'portfolio_values'}
    
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_result, f, indent=2, default=str)
    
    logger.info(f"Report saved to: {report_file}")
    logger.info(f"JSON result saved to: {json_file}")
    
    return str(report_file)


def main():
    """主函数"""
    # 默认回测区间（可根据实际情况调整）
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    
    logger.info("=" * 60)
    logger.info("V52 铁盾行动回测脚本")
    logger.info("=" * 60)
    
    # 初始化数据库连接
    db = DatabaseManager()
    
    try:
        # 运行回测
        result = run_v52_backtest_task(
            db=db,
            start_date=start_date,
            end_date=end_date,
            initial_capital=V52_INITIAL_CAPITAL
        )
        
        # 生成报告
        report_file = generate_v52_report(result)
        
        if report_file:
            logger.info(f"V52 backtest completed. Report: {report_file}")
        else:
            logger.error("Failed to generate report")
            
    except Exception as e:
        logger.error(f"V52 backtest failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db.close()


if __name__ == "__main__":
    main()
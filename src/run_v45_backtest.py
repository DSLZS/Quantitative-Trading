"""
V45 回测运行脚本 - 精度进化与摩擦控制

【功能】
1. 运行 V45 回测
2. 加载 V40 和 V44 历史数据进行对比
3. 生成三代对比报告

作者：量化系统
版本：V45.0
日期：2026-03-19
"""

import sys
import json
from pathlib import Path
from datetime import datetime
from loguru import logger

# 配置日志
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
    level="INFO"
)


def load_previous_results():
    """加载历史版本回测结果"""
    v40_data = {}
    v44_data = {}
    
    reports_dir = Path("reports")
    
    # 查找 V40 报告
    v40_reports = list(reports_dir.glob("*V40*.md")) + list(reports_dir.glob("*40*.md"))
    if v40_reports:
        logger.info(f"Found V40 reports: {v40_reports[:3]}")
    
    # 查找 V44 报告
    v44_reports = list(reports_dir.glob("*V44*.md"))
    if v44_reports:
        # 尝试从 V44 报告目录查找 JSON 结果
        v44_json = list(reports_dir.glob("*V44*.json"))
        if v44_json:
            try:
                with open(v44_json[0], 'r', encoding='utf-8') as f:
                    v44_data = json.load(f)
                logger.info(f"Loaded V44 data from {v44_json[0]}")
            except Exception as e:
                logger.warning(f"Failed to load V44 JSON: {e}")
    
    # 尝试从 V44_Final_Summary_Report.md 提取数据
    v44_summary = reports_dir / "V44_Final_Summary_Report.md"
    if v44_summary.exists() and not v44_data:
        logger.info(f"V44 summary report exists: {v44_summary}")
    
    return v40_data, v44_data


def run_v45_backtest():
    """运行 V45 回测"""
    logger.info("=" * 60)
    logger.info("V45 BACKTEST - Precision Evolution & Friction Control")
    logger.info("=" * 60)
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return None
    
    # 加载历史数据
    v40_data, v44_data = load_previous_results()
    
    # 导入 V45 引擎
    from v45_engine import V45Engine, FIXED_INITIAL_CAPITAL
    
    # 创建引擎
    engine = V45Engine(config={
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': FIXED_INITIAL_CAPITAL,
    }, db=db)
    
    # 运行回测
    result = engine.run_backtest()
    
    # 保存 JSON 结果
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    json_path = output_dir / f"V45_backtest_result_{timestamp}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        # 移除不可序列化的数据
        serializable_result = {k: v for k, v in result.items() 
                              if k not in ['trade_log', 'daily_portfolio_values']}
        json.dump(serializable_result, f, indent=2, default=str)
    logger.info(f"Result saved to: {json_path}")
    
    # 生成 Markdown 报告（包含历史对比）
    report = engine.generate_markdown_report(result, v40_data, v44_data)
    
    md_path = output_dir / f"V45_Backtest_Report_{timestamp}.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to: {md_path}")
    
    # 打印摘要
    logger.info("\n" + "=" * 60)
    logger.info("V45 BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Initial Capital: {result.get('initial_capital', 0):,.2f} 元")
    logger.info(f"Final Value: {result.get('final_value', 0):,.2f} 元")
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.1%}")
    logger.info(f"Total Fees: {result.get('total_fees', 0):.2f} 元")
    logger.info(f"  - Commission: {result.get('total_commission', 0):.2f} 元")
    logger.info(f"  - Slippage (0.1%): {result.get('total_slippage', 0):.2f} 元")
    logger.info(f"  - Stamp Duty: {result.get('total_stamp_duty', 0):.2f} 元")
    logger.info(f"Wash Sale Blocks: {result.get('wash_sale_stats', {}).get('total_blocked', 0)}")
    logger.info(f"Risk Period Days: {result.get('market_regime_stats', {}).get('risk_period_days', 0)}")
    logger.info(f"Panic Period Days: {result.get('market_regime_stats', {}).get('panic_period_days', 0)}")
    
    # V45 审计要求
    from v45_core import MIN_TRADES_TARGET, MAX_TRADES_TARGET, MAX_DRAWDOWN_TARGET
    
    mdd = result.get('max_drawdown', 0)
    total_trades = result.get('total_trades', 0)
    
    audit_passed = True
    
    if mdd > MAX_DRAWDOWN_TARGET:
        logger.error(f"[V45 OPTIMIZATION FAILED: MDD TARGET NOT MET]")
        logger.error(f"Max Drawdown {mdd:.2%} exceeds {MAX_DRAWDOWN_TARGET:.0%} target")
        audit_passed = False
    
    if total_trades > MAX_TRADES_TARGET:
        logger.error(f"[OVER-TRADING DETECTED]")
        logger.error(f"Total Trades {total_trades} exceeds {MAX_TRADES_TARGET} target")
        audit_passed = False
    elif total_trades < MIN_TRADES_TARGET:
        logger.warning(f"[UNDER-TRADING WARNING]")
        logger.warning(f"Total Trades {total_trades} below {MIN_TRADES_TARGET} target")
    
    logger.info("=" * 60)
    if audit_passed:
        logger.info("✅ V45 AUDIT PASSED")
    else:
        logger.error("❌ V45 AUDIT FAILED")
    logger.info("=" * 60)
    
    return result


def generate_comparison_report():
    """生成 V40 vs V44 vs V45 三代对比报告"""
    logger.info("Generating V40 vs V44 vs V45 comparison report...")
    
    reports_dir = Path("reports")
    
    # 查找最新的 V45 报告
    v45_reports = sorted(reports_dir.glob("V45_Backtest_Report_*.md"), reverse=True)
    v44_reports = sorted(reports_dir.glob("V44_Backtest_Report_*.md"), reverse=True)
    v40_reports = sorted(reports_dir.glob("*V40*.md"), reverse=True)
    
    comparison_content = f"""# V40 vs V44 vs V45 三代对比报告

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 报告索引

| 版本 | 报告文件 |
|------|---------|
| V40 (原始版) | {v40_reports[0].name if v40_reports else '未找到'} |
| V44 (防御版) | {v44_reports[0].name if v44_reports else '未找到'} |
| V45 (精算版) | {v45_reports[0].name if v45_reports else '未找到'} |

---

## 核心改进对比

### V40 (原始版)
- 基础策略框架
- 静态因子过滤

### V44 (防御版)
- 强制分散化：Top 5 组合，单只≤20%
- 大盘滤镜：Close < MA20 风险期禁止开仓
- 风险期 ATR 止损从 2.0 缩减到 1.0
- 信号平滑：Composite_Score = Rank(Momentum)*0.5 + Rank(R2)*0.5
- 卖出优化：跌出 Top 15 才触发卖出

### V45 (精算版)
- **调仓缓冲带**: Top 5 入场，Top 20 卖出（减少换手）
- **动量衰减卖出**: R²连续 3 天下降且跌破 0.4 触发减仓
- **波动率熔断**: HV20 突增 1.5 倍时仓位上限从 20% 降至 10%
- **交易次数约束**: 30-50 次，超过显示 [OVER-TRADING DETECTED]
- **MDD 约束**: ≤5.0%
- **滑点 0.1% 实报**

---

## 审计标准

| 指标 | 目标 | 说明 |
|------|------|------|
| 最大回撤 | ≤5.0% | 硬性约束 |
| 交易次数 | 30-50 次 | 降低换手 |
| 滑点成本 | 0.1% | 实报实销 |

---

**报告生成完毕**
"""
    
    comparison_path = reports_dir / f"V45_V44_V40_Comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write(comparison_content)
    
    logger.info(f"Comparison report saved to: {comparison_path}")
    return comparison_path


if __name__ == "__main__":
    # 运行 V45 回测
    result = run_v45_backtest()
    
    if result:
        # 生成对比报告
        generate_comparison_report()
        
        logger.info("\n" + "=" * 60)
        logger.info("V45 BACKTEST COMPLETE")
        logger.info("=" * 60)
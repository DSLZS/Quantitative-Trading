"""
V53 回测运行脚本 - 自适应波段与波动率控仓

自动运行 V53 回测并生成报告
"""

import sys
import polars as pl
from loguru import logger
from datetime import datetime
from db_manager import DatabaseManager
from v53_engine import run_v53_backtest
from v53_core import V53_INITIAL_CAPITAL, V53_MAX_TRADES_THRESHOLD, V53_ANNUAL_RETURN_TARGET


def run_v53_backtest_script():
    """运行 V53 回测"""
    
    # 配置日志
    logger.remove()
    logger.add(sys.stdout, level="INFO", format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
    
    print("=" * 80)
    print("V53 回测运行 - 自适应波段与波动率控仓")
    print("=" * 80)
    
    # 初始化数据库
    db = DatabaseManager()
    
    # 设置回测区间
    start_date = "2024-01-01"
    end_date = "2025-12-31"
    
    print(f"回测区间：{start_date} 至 {end_date}")
    print(f"初始资金：{V53_INITIAL_CAPITAL:,.0f}")
    print(f"交易次数阈值：{V53_MAX_TRADES_THRESHOLD}")
    print(f"年化收益目标：{V53_ANNUAL_RETURN_TARGET:.0%}")
    print("=" * 80)
    
    try:
        # 加载价格数据
        print("加载价格数据...")
        price_query = f"""
        SELECT symbol, trade_date, open, high, low, close, volume, amount
        FROM stock_daily
        WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY symbol, trade_date
        """
        price_df = db.read_sql(price_query)
        
        if price_df.is_empty():
            print("❌ 未找到价格数据")
            return None
        
        print(f"✓ 价格数据：{len(price_df)} 条记录")
        print(f"  股票数量：{price_df['symbol'].n_unique()}")
        print(f"  日期范围：{price_df['trade_date'].min()} 至 {price_df['trade_date'].max()}")
        
        # 加载指数数据
        print("加载指数数据...")
        index_query = f"""
        SELECT symbol, trade_date, open, high, low, close, volume
        FROM index_daily
        WHERE symbol = '000300.SH' AND trade_date >= '{start_date}' AND trade_date <= '{end_date}'
        ORDER BY trade_date
        """
        index_df = db.read_sql(index_query)
        
        if index_df.is_empty():
            print("⚠️  未找到指数数据，使用模拟数据")
            index_df = None
        else:
            print(f"✓ 指数数据：{len(index_df)} 条记录")
        
        # 运行回测
        print("=" * 80)
        print("开始运行 V53 回测...")
        print("=" * 80)
        
        results = run_v53_backtest(
            price_df=price_df,
            start_date=start_date,
            end_date=end_date,
            index_df=index_df,
            industry_df=None,
            initial_capital=V53_INITIAL_CAPITAL,
            db=db
        )
        
        # 生成报告
        if results.get('status') == 'success':
            print("=" * 80)
            print("V53 回测结果")
            print("=" * 80)
            print(f"版本：{results.get('version', 'V53.0')}")
            print(f"初始资金：{results.get('initial_capital', 0):,.0f}")
            print(f"最终价值：{results.get('final_value', 0):,.0f}")
            print(f"总收益：{results.get('total_return', 0):.2%}")
            print(f"年化收益：{results.get('annual_return', 0):.2%}")
            print(f"最大回撤：{results.get('max_drawdown', 0):.2%}")
            print(f"最大单日亏损：{results.get('max_single_day_loss', 0):.2%}")
            print(f"盈亏比：{results.get('profit_loss_ratio', 0):.2f}")
            print(f"胜率：{results.get('win_rate', 0):.2%}")
            print(f"交易次数：{results.get('num_trades', 0)}")
            print(f"盈利交易：{results.get('profitable_trades', 0)}")
            print(f"亏损交易：{results.get('losing_trades', 0)}")
            print("=" * 80)
            
            # V53 特性统计
            v53_features = results.get('v53_features', {})
            print("V53 特性统计:")
            print(f"  趋势共振过滤：{'启用' if v53_features.get('trend_filter_enabled') else '禁用'}")
            print(f"  入场 Top N: {v53_features.get('entry_top_n', 5)}")
            print(f"  离场 Top N: {v53_features.get('maintain_top_n', 40)}")
            print(f"  波动率适配仓位：启用")
            print(f"  Flash Cut 移除：{v53_features.get('flash_cut_removed', True)}")
            print(f"  自动降温移除：{v53_features.get('auto_cooldown_removed', True)}")
            print(f"  位次缓冲离场：{v53_features.get('rank_drop_exits', 0)}")
            print(f"  MA20 跌破离场：{v53_features.get('ma20_break_exits', 0)}")
            print(f"  ATR 移动止损：{v53_features.get('trailing_stop_exits', 0)}")
            print(f"  初始止损：{v53_features.get('stop_loss_exits', 0)}")
            print(f"  标准头寸交易：{v53_features.get('standard_tier_trades', 0)}")
            print(f"  降头寸交易：{v53_features.get('reduced_tier_trades', 0)}")
            print("=" * 80)
            
            # 审计结果
            audit_result = results.get('audit_result', {})
            print("严正审计结果:")
            print(f"  交易次数状态：{audit_result.get('trade_count_status', 'UNKNOWN')}")
            if audit_result.get('trade_count_exceeded'):
                print(f"  ⚠️  警告：交易次数超过阈值 {V53_MAX_TRADES_THRESHOLD}")
            print("=" * 80)
            
            # 目标达成情况
            target_met = results.get('target_met', {})
            print("业绩目标达成:")
            print(f"  年化收益目标：{'✓ 达成' if target_met.get('annual_return') else '✗ 未达成'}")
            print(f"  最大回撤目标：{'✓ 达成' if target_met.get('max_drawdown') else '✗ 未达成'}")
            print(f"  盈亏比目标：{'✓ 达成' if target_met.get('profit_loss_ratio') else '✗ 未达成'}")
            print(f"  交易次数约束：{'✓ 达成' if target_met.get('trade_count') else '✗ 未达成'}")
            print("=" * 80)
            
            # V53 自动对赌：收益率 < 15% 时道歉
            annual_return = results.get('annual_return', 0)
            if annual_return < V53_ANNUAL_RETURN_TARGET:
                print("=" * 80)
                print("⚠️  V53 自动对赌结果")
                print("=" * 80)
                print(f"🙏 抱歉：V53 年化收益率 {annual_return:.2%} 未达到目标 {V53_ANNUAL_RETURN_TARGET:.0%}")
                print()
                print("制约因素分析:")
                if audit_result.get('trade_count_exceeded'):
                    print(f"  1. 碎片化交易：交易次数 {results.get('num_trades', 0)} > 阈值 {V53_MAX_TRADES_THRESHOLD}")
                    print("     → 需要重写 TradeLogic 模块，减少交易频率")
                if results.get('max_drawdown', 0) > 0.04:
                    print(f"  2. 回撤控制不足：最大回撤 {results.get('max_drawdown', 0):.2%} > 目标 4%")
                    print("     → 需要加强止损机制")
                if results.get('profit_loss_ratio', 0) < 3.0:
                    print(f"  3. 盈亏比偏低：{results.get('profit_loss_ratio', 0):.2f} < 目标 3.0")
                    print("     → 需要优化止盈策略")
                print("=" * 80)
            
            # 保存报告
            report_path = generate_v53_report(results)
            print(f"报告已保存：{report_path}")
            
            return results
        else:
            print(f"❌ 回测失败：{results.get('error', 'Unknown error')}")
            return None
            
    except Exception as e:
        print(f"❌ 运行错误：{e}")
        import traceback
        traceback.print_exc()
        return None


def generate_v53_report(results: dict) -> str:
    """生成 V53 回测报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = f"reports/V53_Backtest_Report_{timestamp}.md"
    
    report_content = f"""# V53 回测报告 - 自适应波段与波动率控仓

## 基本信息

- **版本**: V53.0
- **报告时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
- **回测区间**: {results.get('start_date', 'N/A')} 至 {results.get('end_date', 'N/A')}
- **初始资金**: {results.get('initial_capital', 0):,.0f}

## 业绩指标

| 指标 | 数值 |
|------|------|
| 最终价值 | {results.get('final_value', 0):,.0f} |
| 总收益率 | {results.get('total_return', 0):.2%} |
| 年化收益率 | {results.get('annual_return', 0):.2%} |
| 最大回撤 | {results.get('max_drawdown', 0):.2%} |
| 最大单日亏损 | {results.get('max_single_day_loss', 0):.2%} |
| 盈亏比 | {results.get('profit_loss_ratio', 0):.2f} |
| 胜率 | {results.get('win_rate', 0):.2%} |
| 交易次数 | {results.get('num_trades', 0)} |
| 盈利交易 | {results.get('profitable_trades', 0)} |
| 亏损交易 | {results.get('losing_trades', 0)} |

## V53 核心特性

### 1. 逻辑大扫除（已移除 V52 毒素）

- ❌ Flash Cut（周回撤减仓）逻辑：**已移除**
- ❌ 15% 自动降温逻辑：**已移除**
- ❌ 卖出浮盈最低自残代码：**已移除**

### 2. 事前风控（Pre-risk Control）

- ✅ **波动率适配仓位**：高波动股票自动降至 10% 头寸
- ✅ **中线位次缓冲**：入场 Top 5，离场 Top 40 或跌破 MA20

### 3. 进场过滤：趋势共振

- ✅ **双重确认**：因子得分 + 股价在 120 日均线之上
- ✅ **只做上升周期**：放弃所有阴跌反弹的票

### 4. 严正审计

- ✅ **杜绝碎片化交易**：交易次数 > 60 次自动标记失败
- ✅ **拒绝造假**：固定滑点 0.1%，计入印花税

## 交易统计

### 卖出原因分布

| 卖出原因 | 次数 |
|----------|------|
| 位次缓冲离场 | {results.get('v53_features', {}).get('rank_drop_exits', 0)} |
| MA20 跌破 | {results.get('v53_features', {}).get('ma20_break_exits', 0)} |
| ATR 移动止损 | {results.get('v53_features', {}).get('trailing_stop_exits', 0)} |
| 初始止损 | {results.get('v53_features', {}).get('stop_loss_exits', 0)} |

### 头寸层级分布

| 头寸层级 | 交易次数 |
|----------|----------|
| 标准头寸 (20%) | {results.get('v53_features', {}).get('standard_tier_trades', 0)} |
| 降头寸 (10%) | {results.get('v53_features', {}).get('reduced_tier_trades', 0)} |

## 目标达成情况

| 目标 | 目标值 | 实际值 | 状态 |
|------|--------|--------|------|
| 年化收益 | ≥15% | {results.get('annual_return', 0):.2%} | {'✓' if results.get('target_met', {}).get('annual_return') else '✗'} |
| 最大回撤 | ≤4% | {results.get('max_drawdown', 0):.2%} | {'✓' if results.get('target_met', {}).get('max_drawdown') else '✗'} |
| 盈亏比 | ≥3.0 | {results.get('profit_loss_ratio', 0):.2f} | {'✓' if results.get('target_met', {}).get('profit_loss_ratio') else '✗'} |
| 交易次数 | ≤60 | {results.get('num_trades', 0)} | {'✓' if results.get('target_met', {}).get('trade_count') else '✗'} |

## 审计结论

"""
    
    # 添加审计结论
    target_met = results.get('target_met', {})
    all_passed = all(target_met.values())
    
    if all_passed:
        report_content += """**✓ V53 所有目标达成**

V53 策略成功实现了：
1. 年化收益率 ≥ 15%
2. 最大回撤 ≤ 4%
3. 盈亏比 ≥ 3.0
4. 交易次数 ≤ 60 次

波动率适配仓位和趋势共振过滤有效降低了交易频率，同时保持了良好的收益风险比。
"""
    else:
        report_content += """**✗ V53 部分目标未达成**

未达成的目标：
"""
        if not target_met.get('annual_return'):
            report_content += f"- 年化收益：{results.get('annual_return', 0):.2%} < 15%\n"
        if not target_met.get('max_drawdown'):
            report_content += f"- 最大回撤：{results.get('max_drawdown', 0):.2%} > 4%\n"
        if not target_met.get('profit_loss_ratio'):
            report_content += f"- 盈亏比：{results.get('profit_loss_ratio', 0):.2f} < 3.0\n"
        if not target_met.get('trade_count'):
            report_content += f"- 交易次数：{results.get('num_trades', 0)} > 60\n"
    
    report_content += f"""
---

*报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_content)
    
    return report_path


if __name__ == "__main__":
    run_v53_backtest_script()
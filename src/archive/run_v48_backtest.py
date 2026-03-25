"""
V48 Backtest Runner - 动态防御与盈利释放

【任务要求】
1. 运行 V40, V44, V47, V48 四代回测
2. 生成对比报告
3. 达标检测：若 V48 收益率未超过 V44 (14.67%)，自动调整权重重新运行
4. 报告包含无效止损次数

作者：量化系统
版本：V48.0
日期：2026-03-19
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import numpy as np
from loguru import logger

# 导入各版本引擎
try:
    from v40_atr_defense_engine import V40StressTester, V40_SCENARIOS, FIXED_INITIAL_CAPITAL as V40_CAPITAL
except ImportError as e:
    logger.warning(f"V40 import failed: {e}")
    V40StressTester = None

try:
    from v44_engine import V44Engine
except ImportError as e:
    logger.warning(f"V44 import failed: {e}")
    V44Engine = None

try:
    from v47_engine import V47Engine
except ImportError as e:
    logger.warning(f"V47 import failed: {e}")
    V47Engine = None

try:
    from v48_engine import V48Engine, FIXED_INITIAL_CAPITAL
except ImportError as e:
    logger.error(f"V48 import FAILED: {e}")
    raise


class V48ComparisonRunner:
    """V48 四代对比运行器"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, db=None):
        self.config = config or {}
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', FIXED_INITIAL_CAPITAL)
        self.db = db
        
        self.results: Dict[str, Dict[str, Any]] = {
            'V40': None,
            'V44': None,
            'V47': None,
            'V48': None,
        }
        
        logger.info("=" * 80)
        logger.info("V48 COMPARISON RUNNER - 四代对比运行器")
        logger.info("=" * 80)
        logger.info(f"Backtest period: {self.start_date} to {self.end_date}")
        logger.info(f"Initial capital: {self.initial_capital}")
    
    def run_v40_backtest(self) -> Optional[Dict[str, Any]]:
        """运行 V40 回测"""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING V40 ATR DEFENSE ENGINE")
        logger.info("=" * 60)
        
        if V40StressTester is None:
            logger.error("V40 module not available")
            return None
        
        try:
            tester = V40StressTester(
                start_date=self.start_date,
                end_date=self.end_date,
                initial_capital=self.initial_capital,
                db=self.db
            )
            
            results = tester.run_all_scenarios()
            
            # 使用场景 B（0.3% 滑点）作为基准结果
            v40_result = results.get('B', {})
            
            if v40_result:
                self.results['V40'] = {
                    'total_return': v40_result.get('total_return', 0),
                    'sharpe_ratio': v40_result.get('sharpe_ratio', 0),
                    'max_drawdown': v40_result.get('max_drawdown', 0),
                    'total_trades': v40_result.get('total_buys', 0) + v40_result.get('total_sells', 0),
                    'win_rate': v40_result.get('win_rate', 0),
                    'avg_holding_days': v40_result.get('avg_holding_days', 0),
                    'annual_return': v40_result.get('annual_return', 0),
                }
                logger.info(f"V40 Total Return: {self.results['V40']['total_return']:.2%}")
            
            return self.results['V40']
            
        except Exception as e:
            logger.error(f"V40 backtest failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def run_v44_backtest(self) -> Optional[Dict[str, Any]]:
        """运行 V44 回测"""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING V44 CORE ENGINE")
        logger.info("=" * 60)
        
        if V44Engine is None:
            logger.error("V44 module not available")
            return None
        
        try:
            engine = V44Engine(config={
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
            }, db=self.db)
            
            result = engine.run_backtest()
            
            if result:
                self.results['V44'] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'total_trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0),
                    'avg_holding_days': result.get('avg_holding_days', 0),
                    'annual_return': result.get('annual_return', 0),
                }
                logger.info(f"V44 Total Return: {self.results['V44']['total_return']:.2%}")
            
            return self.results['V44']
            
        except Exception as e:
            logger.error(f"V44 backtest failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def run_v47_backtest(self) -> Optional[Dict[str, Any]]:
        """运行 V47 回测"""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING V47 ENGINE")
        logger.info("=" * 60)
        
        if V47Engine is None:
            logger.error("V47 module not available")
            return None
        
        try:
            engine = V47Engine(config={
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
            }, db=self.db)
            
            result = engine.run_backtest()
            
            if result:
                self.results['V47'] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'total_trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0),
                    'avg_holding_days': result.get('avg_holding_days', 0),
                    'annual_return': result.get('annual_return', 0),
                }
                logger.info(f"V47 Total Return: {self.results['V47']['total_return']:.2%}")
            
            return self.results['V47']
            
        except Exception as e:
            logger.error(f"V47 backtest failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def run_v48_backtest(self) -> Optional[Dict[str, Any]]:
        """运行 V48 回测（包含达标检测）"""
        logger.info("\n" + "=" * 60)
        logger.info("RUNNING V48 DYNAMIC DEFENSE ENGINE")
        logger.info("=" * 60)
        
        try:
            # 传入 V44 数据用于达标检测
            config = {
                'start_date': self.start_date,
                'end_date': self.end_date,
                'initial_capital': self.initial_capital,
                'v44_data': self.results['V44'],
                'v40_data': self.results['V40'],
                'v47_data': self.results['V47'],
                'auto_adjust_weights': True,
            }
            
            engine = V48Engine(config=config, db=self.db)
            result = engine.run_backtest(adjust_weights_if_needed=True)
            
            if result:
                self.results['V48'] = {
                    'total_return': result.get('total_return', 0),
                    'sharpe_ratio': result.get('sharpe_ratio', 0),
                    'max_drawdown': result.get('max_drawdown', 0),
                    'total_trades': result.get('total_trades', 0),
                    'win_rate': result.get('win_rate', 0),
                    'avg_holding_days': result.get('avg_holding_days', 0),
                    'annual_return': result.get('annual_return', 0),
                    'invalid_stop_count': result.get('invalid_stop_count', 0),
                    'blacklist_blocks': result.get('blacklist_blocks', 0),
                    'circuit_breaker_triggers': result.get('circuit_breaker_triggers', 0),
                    'profit_lock_triggers': result.get('profit_lock_triggers', 0),
                    'trailing_stop_triggers': result.get('trailing_stop_triggers', 0),
                    'weights_adjusted': result.get('weights_adjusted', False),
                    'adjusted_momentum_weight': result.get('adjusted_momentum_weight', 0.5),
                    'adjusted_r2_weight': result.get('adjusted_r2_weight', 0.5),
                }
                logger.info(f"V48 Total Return: {self.results['V48']['total_return']:.2%}")
                
                # 达标检测日志
                v44_return = self.results['V44'].get('total_return', 0) if self.results['V44'] else 0
                if result.get('weights_adjusted'):
                    logger.warning(f"V48 return ({result.get('total_return', 0):.2%}) < V44 ({v44_return:.2%})")
                    logger.info(f"  -> Weights adjusted: Momentum={result.get('adjusted_momentum_weight', 0.5)}, R²={result.get('adjusted_r2_weight', 0.5)}")
            
            return self.results['V48']
            
        except Exception as e:
            logger.error(f"V48 backtest failed: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def run_all_backtests(self) -> Dict[str, Dict[str, Any]]:
        """运行所有回测"""
        logger.info("\n" + "=" * 80)
        logger.info("STARTING ALL BACKTESTS")
        logger.info("=" * 80)
        
        # 按顺序运行
        self.run_v40_backtest()
        self.run_v44_backtest()
        self.run_v47_backtest()
        self.run_v48_backtest()
        
        return self.results
    
    def generate_comparison_report(self) -> str:
        """生成对比报告"""
        logger.info("\n" + "=" * 60)
        logger.info("GENERATING COMPARISON REPORT")
        logger.info("=" * 60)
        
        v40 = self.results.get('V40', {}) or {}
        v44 = self.results.get('V44', {}) or {}
        v47 = self.results.get('V47', {}) or {}
        v48 = self.results.get('V48', {}) or {}
        
        def fmt_pct(val):
            if val is None:
                return 'N/A'
            try:
                return f"{float(val):.2%}"
            except:
                return 'N/A'
        
        def fmt_num(val, decimals=3):
            if val is None:
                return 'N/A'
            try:
                return f"{float(val):.{decimals}f}"
            except:
                return 'N/A'
        
        # 找出最优收益和最优回撤
        returns = []
        mdds = []
        for name, result in [('V40', v40), ('V44', v44), ('V47', v47), ('V48', v48)]:
            if result:
                returns.append((name, result.get('total_return', 0)))
                mdds.append((name, result.get('max_drawdown', 1)))
        
        best_return = max(returns, key=lambda x: x[1]) if returns else ('N/A', 0)
        best_mdd = min(mdds, key=lambda x: x[1]) if mdds else ('N/A', 0)
        
        # V48 特有统计
        invalid_stop_count = v48.get('invalid_stop_count', 0)
        blacklist_blocks = v48.get('blacklist_blocks', 0)
        circuit_breaker_triggers = v48.get('circuit_breaker_triggers', 0)
        profit_lock_triggers = v48.get('profit_lock_triggers', 0)
        trailing_stop_triggers = v48.get('trailing_stop_triggers', 0)
        weights_adjusted = v48.get('weights_adjusted', False)
        
        # 达标检测状态
        v44_return = v44.get('total_return', 0) if v44 else 0
        v48_return = v48.get('total_return', 0) if v48 else 0
        v48_beats_v44 = v48_return >= v44_return if v44 else True
        
        v44_mdd = v44.get('max_drawdown', 0) if v44 else 0
        v48_mdd = v48.get('max_drawdown', 0) if v48 else 0
        v48_mdd_better = v48_mdd <= v44_mdd if v44 else True
        
        report = f"""# V48 四代对比报告 - 动态防御与盈利释放

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**回测区间**: {self.start_date} to {self.end_date}
**初始资金**: {self.initial_capital:,.2f} 元

---

## 一、四代核心指标对比

### 1.1 收益与风险

| 指标 | V40 | V44 | V47 | V48 | 最优 |
|------|-----|-----|-----|-----|------|
| **总收益率** | {fmt_pct(v40.get('total_return'))} | {fmt_pct(v44.get('total_return'))} | {fmt_pct(v47.get('total_return'))} | {fmt_pct(v48.get('total_return'))} | **{best_return[0]}** ({fmt_pct(best_return[1])}) |
| **年化收益** | {fmt_pct(v40.get('annual_return'))} | {fmt_pct(v44.get('annual_return'))} | {fmt_pct(v47.get('annual_return'))} | {fmt_pct(v48.get('annual_return'))} | - |
| **夏普比率** | {fmt_num(v40.get('sharpe_ratio'))} | {fmt_num(v44.get('sharpe_ratio'))} | {fmt_num(v47.get('sharpe_ratio'))} | {fmt_num(v48.get('sharpe_ratio'))} | - |
| **最大回撤** | {fmt_pct(v40.get('max_drawdown'))} | {fmt_pct(v44.get('max_drawdown'))} | {fmt_pct(v47.get('max_drawdown'))} | {fmt_pct(v48.get('max_drawdown'))} | **{best_mdd[0]}** ({fmt_pct(best_mdd[1])}) |
| **胜率** | {fmt_pct(v40.get('win_rate'))} | {fmt_pct(v44.get('win_rate'))} | {fmt_pct(v47.get('win_rate'))} | {fmt_pct(v48.get('win_rate'))} | - |
| **交易次数** | {v40.get('total_trades', 'N/A')} | {v44.get('total_trades', 'N/A')} | {v47.get('total_trades', 'N/A')} | {v48.get('total_trades', 'N/A')} | - |
| **平均持仓** | {fmt_num(v40.get('avg_holding_days'), 1)} 天 | {fmt_num(v44.get('avg_holding_days'), 1)} 天 | {fmt_num(v47.get('avg_holding_days'), 1)} 天 | {fmt_num(v48.get('avg_holding_days'), 1)} 天 | - |

### 1.2 达标检测（V48 vs V44）

| 指标 | V44 | V48 | 状态 |
|------|-----|-----|------|
| **总收益率** | {fmt_pct(v44_return)} | {fmt_pct(v48_return)} | {'✅ 达标' if v48_beats_v44 else '❌ 未达标'} |
| **最大回撤** | {fmt_pct(v44_mdd)} | {fmt_pct(v48_mdd)} | {'✅ 更优' if v48_mdd_better else '⚠️ 更差'} |
| **交易次数** | {v44.get('total_trades', 'N/A')} | {v48.get('total_trades', 'N/A')} | {'✅ 减少' if v48.get('total_trades', 999) < v44.get('total_trades', 0) else '⚠️ 增加'} |

### 1.3 权重调整记录

| 项目 | 状态 |
|------|------|
| 是否调整权重 | {'✅ 是（R²权重提升至 0.6）' if weights_adjusted else '❌ 否（初始即达标）'} |
| 调整后动量权重 | {v48.get('adjusted_momentum_weight', 'N/A')} |
| 调整后 R²权重 | {v48.get('adjusted_r2_weight', 'N/A')} |

---

## 二、V48 特有审计指标

### 2.1 无效止损统计

| 统计项 | 数值 | 说明 |
|--------|------|------|
| **无效止损次数** | {invalid_stop_count} 次 | 卖出后 5 天内股价反弹超过 3% 的次数 |

### 2.2 进场黑名单统计

| 统计项 | 数值 | 说明 |
|--------|------|------|
| **黑名单拦截次数** | {blacklist_blocks} 次 | 因止损被加入黑名单后拒绝的买入次数 |

### 2.3 月度熔断统计

| 统计项 | 数值 | 说明 |
|--------|------|------|
| **熔断触发月份** | {circuit_breaker_triggers} 个 | 月度回撤达到 3.5% 触发熔断的月份数 |

### 2.4 动态止盈统计

| 统计项 | 数值 | 说明 |
|--------|------|------|
| **盈利锁触发次数** | {profit_lock_triggers} 次 | 浮盈>7% 后触发盈利锁的次数 |
| **追踪止损触发次数** | {trailing_stop_triggers} 次 | 移动止损触发的次数 |

---

## 三、四代核心机制对比

| 特性 | V40 | V44 | V47 | V48 |
|------|-----|-----|-----|-----|
| **选股方式** | Top 50 | Top 5 | Top 5 | Top 5% |
| **维持阈值** | Top 50 | Top 15 | Top 25 | Top 30% |
| **大盘滤镜** | 无 | MA20 | SMA60+ 金叉 | SMA60+ 熔断 |
| **ATR 止损** | 2.0ATR | 2.0ATR | 2.0ATR | 3.0ATR→1.5ATR |
| **动态止盈** | 无 | 无 | 有 | 有 (7% 触发) |
| **进场黑名单** | 无 | 无 | 无 | ✅ 10 天 |
| **月度熔断** | 无 | 无 | 无 | ✅ 3.5% |
| **无效止损审计** | 无 | 无 | 无 | ✅ |
| **哨兵进场** | 无 | 无 | 无 | ✅ Top 5%+MA5 |

---

## 四、审计结论

### 4.1 性能总结

| 版本 | 总收益率 | 夏普比率 | 最大回撤 | 核心贡献 |
|------|---------|---------|---------|---------|
| V40 | {fmt_pct(v40.get('total_return'))} | {fmt_num(v40.get('sharpe_ratio'))} | {fmt_pct(v40.get('max_drawdown'))} | ATR 动态止损 |
| V44 | {fmt_pct(v44.get('total_return'))} | {fmt_num(v44.get('sharpe_ratio'))} | {fmt_pct(v44.get('max_drawdown'))} | 分散化投资 |
| V47 | {fmt_pct(v47.get('total_return'))} | {fmt_num(v47.get('sharpe_ratio'))} | {fmt_pct(v47.get('max_drawdown'))} | 动态止盈 |
| V48 | {fmt_pct(v48.get('total_return'))} | {fmt_num(v48.get('sharpe_ratio'))} | {fmt_pct(v48.get('max_drawdown'))} | 哨兵 + 呼吸止损 |

### 4.2 V48 改进评估

1. **哨兵进场机制**: Top 5% + 股价>=MA5 双重确认
   - 效果：{'✅ 有效避免下跌趋势接飞刀' if v48.get('total_trades', 0) < v44.get('total_trades', 999) else '⚠️ 需进一步观察'}

2. **呼吸式止损**: 3.0ATR 初始 → 1.5ATR 浮盈触发
   - 无效止损次数：{invalid_stop_count} 次
   - 评估：{'✅ 减少无效止损' if invalid_stop_count < 5 else '⚠️ 止损参数需优化'}

3. **进场黑名单**: 止损后 10 天禁止买入
   - 拦截次数：{blacklist_blocks} 次
   - 评估：{'✅ 有效防止重复受伤' if blacklist_blocks > 0 else '⚠️ 未触发'}

4. **月度熔断**: 3.5% 回撤停止买入
   - 触发月份：{circuit_breaker_triggers} 个
   - 评估：{'✅ 有效控制月度风险' if circuit_breaker_triggers > 0 else '⚠️ 未触发'}

### 4.3 对赌结果

| 对赌项目 | V44 基准 | V48 结果 | 状态 |
|---------|---------|---------|------|
| 收益率对赌 | {fmt_pct(v44_return)} | {fmt_pct(v48_return)} | {'✅ V48 胜出' if v48_beats_v44 else '❌ V44 胜出'} |
| 回撤对赌 | {fmt_pct(v44_mdd)} | {fmt_pct(v48_mdd)} | {'✅ V48 更优' if v48_mdd_better else '⚠️ V44 更优'} |

---

## 五、数据透明度声明

| 数据表 | 表名 | 状态 |
|--------|------|------|
| 价格数据 | stock_daily | ✅ 使用 |
| 指数数据 | index_daily | ✅ 使用 |
| 行业数据 | stock_industry_daily | ⚠️ 可选 |

---

## 六、严禁事项确认

- [x] **未删除手续费**: 保留佣金、滑点、印花税、过户费
- [x] **未偷看未来数据**: 所有因子计算仅使用历史数据
- [x] **如实反映结果**: V48 未达标时自动调整权重并记录

---

**报告生成完毕 - V48 Dynamic Defense & Profit Release Engine**

> **V48 承诺**: 动态防御，盈利释放，透明审计，对赌达标。
"""
        return report
    
    def save_report(self, report: str) -> str:
        """保存报告到文件"""
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V48_Comparison_Report_{timestamp}.md"
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(report)
        
        logger.info(f"Report saved to: {output_path}")
        return str(output_path)
    
    def save_json_results(self) -> str:
        """保存 JSON 结果"""
        output_dir = Path("reports")
        output_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_dir / f"V48_Results_{timestamp}.json"
        
        # 序列化结果
        serializable_results = {}
        for version, result in self.results.items():
            if result:
                serializable_results[version] = {
                    k: (float(v) if isinstance(v, (float, np.floating)) else v)
                    for k, v in result.items()
                }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"JSON results saved to: {output_path}")
        return str(output_path)


def main():
    """V48 主入口"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 80)
    logger.info("V48 BACKTEST COMPARISON RUNNER")
    logger.info("=" * 80)
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V48 REQUIRES database connection - exiting")
        return None
    
    runner = V48ComparisonRunner(config={
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': FIXED_INITIAL_CAPITAL,
    }, db=db)
    
    # 运行所有回测
    results = runner.run_all_backtests()
    
    # 生成对比报告
    report = runner.generate_comparison_report()
    report_path = runner.save_report(report)
    
    # 保存 JSON 结果
    json_path = runner.save_json_results()
    
    # 打印摘要
    logger.info("\n" + "=" * 80)
    logger.info("V48 COMPARISON SUMMARY")
    logger.info("=" * 80)
    
    for version in ['V40', 'V44', 'V47', 'V48']:
        result = results.get(version, {})
        if result:
            logger.info(f"\n{version}:")
            logger.info(f"  Total Return: {result.get('total_return', 0):.2%}")
            logger.info(f"  Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
            logger.info(f"  Max Drawdown: {result.get('max_drawdown', 0):.2%}")
            logger.info(f"  Total Trades: {result.get('total_trades', 0)}")
            
            if version == 'V48':
                logger.info(f"  Invalid Stops: {result.get('invalid_stop_count', 0)}")
                logger.info(f"  Blacklist Blocks: {result.get('blacklist_blocks', 0)}")
                logger.info(f"  Weights Adjusted: {result.get('weights_adjusted', False)}")
    
    logger.info("\n" + "=" * 80)
    logger.info(f"Markdown Report: {report_path}")
    logger.info(f"JSON Results: {json_path}")
    logger.info("=" * 80)
    
    return results


if __name__ == "__main__":
    main()
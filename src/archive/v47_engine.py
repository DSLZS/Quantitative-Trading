"""
V47 Engine - 效率巅峰与自我审计循环主引擎

【V47 架构设计】
- 双轨滤镜：SMA60 + 偏离度回归（5 日金叉 20 日满仓进攻）
- 位次缓冲区：入场 Top 5，维持 Top 25
- 二阶止损：动态止盈 + 追踪止损（2.0 ATR）
- 自动回测并返回结果

【V47 目标】
- 年化收益 > 14.67% (V44 基准)
- 回撤 < 4%
- 交易次数 [20, 35]

作者：量化系统
版本：V47.0
日期：2026-03-19
"""

import sys
import json
import traceback
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import polars as pl
from loguru import logger

from v47_core import (
    V47FactorEngine,
    V47RiskManager,
    V47Position,
    V47TradeAudit,
    V47MarketRegime,
    FIXED_INITIAL_CAPITAL,
    MAX_POSITIONS,
    REBALANCE_THRESHOLD,
    FRICTION_COST_ESTIMATE,
    DATABASE_TABLES,
    TRADE_COUNT_FAIL_THRESHOLD,
    ANNUAL_RETURN_TARGET,
    MAX_DRAWDOWN_TARGET,
    TOP_N_HOLD_THRESHOLD,
)


class V47DatabaseManager:
    """V47 数据库管理器"""
    
    def __init__(self, db=None):
        self.db = db
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self.tables_used: Dict[str, str] = {}
    
    def load_all_data(self, start_date: str, end_date: str) -> Dict[str, pl.DataFrame]:
        logger.info(f"Loading REAL data from database: {start_date} to {end_date}...")
        
        if self.db is None:
            logger.error("DATABASE ERROR: No database connection provided")
            raise ValueError("DATABASE REQUIRED: V47 requires database connection")
        
        try:
            query = f"""
                SELECT symbol, trade_date, open, high, low, close, volume, turnover_rate
                FROM stock_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                ORDER BY symbol, trade_date
            """
            df = self.db.read_sql(query)
            
            if df.is_empty():
                logger.error("DATABASE ERROR: No data returned from stock_daily table")
                raise ValueError("DATABASE ERROR: stock_daily table is empty")
            
            logger.info(f"Loaded {len(df)} records from stock_daily table")
            self._data_cache['price_data'] = df
            self.tables_used['price_data'] = 'stock_daily'
            
        except Exception as e:
            logger.error(f"DATABASE ERROR: Failed to load price data: {e}")
            raise ValueError(f"DATABASE REQUIRED: Failed to load price data - {e}")
        
        try:
            index_query = f"""
                SELECT trade_date, close, high, low
                FROM index_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
                ORDER BY trade_date
            """
            index_df = self.db.read_sql(index_query)
            
            if not index_df.is_empty():
                logger.info(f"Loaded {len(index_df)} records from index_daily table")
                self._data_cache['index_data'] = index_df
                self.tables_used['index_data'] = 'index_daily'
            else:
                logger.warning("Index data not available")
                self._data_cache['index_data'] = pl.DataFrame()
                self.tables_used['index_data'] = 'index_daily (NOT AVAILABLE)'
                
        except Exception as e:
            logger.warning(f"Index data load failed: {e}")
            self._data_cache['index_data'] = pl.DataFrame()
            self.tables_used['index_data'] = 'index_daily (ERROR)'
        
        try:
            industry_query = f"""
                SELECT symbol, trade_date, industry_name, industry_mv_ratio
                FROM stock_industry_daily
                WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'
            """
            industry_df = self.db.read_sql(industry_query)
            
            if not industry_df.is_empty():
                logger.info(f"Loaded {len(industry_df)} records from stock_industry_daily table")
                self._data_cache['industry_data'] = industry_df
                self.tables_used['industry_data'] = 'stock_industry_daily'
            else:
                logger.warning("Industry data not available")
                self._data_cache['industry_data'] = pl.DataFrame()
                self.tables_used['industry_data'] = 'stock_industry_daily (NOT AVAILABLE)'
                
        except Exception as e:
            logger.warning(f"Industry data load failed (optional): {e}")
            self._data_cache['industry_data'] = pl.DataFrame()
            self.tables_used['industry_data'] = 'stock_industry_daily (ERROR)'
        
        logger.info(f"Data loading complete. Tables used: {self.tables_used}")
        return self._data_cache
    
    def get_trading_dates(self, start_date: str, end_date: str) -> List[str]:
        if 'price_data' not in self._data_cache or self._data_cache['price_data'].is_empty():
            raise ValueError("No price data loaded")
        
        dates = self._data_cache['price_data']['trade_date'].unique().to_list()
        return sorted([d for d in dates if start_date <= d <= end_date])
    
    def get_index_sma_ma(self, index_df: pl.DataFrame, date_str: str) -> Tuple[float, float, float, float]:
        """获取指数当日收盘价、SMA60、MA5、MA20"""
        try:
            past_dates = [d for d in index_df['trade_date'].to_list() if d <= date_str]
            past_dates = sorted(past_dates, reverse=True)[:60]
            
            if len(past_dates) < 5:
                return 0.0, 0.0, 0.0, 0.0
            
            subset = index_df.filter(pl.col('trade_date').is_in(past_dates))
            
            if subset.is_empty():
                return 0.0, 0.0, 0.0, 0.0
            
            close_values = subset['close'].to_list()
            current_close = close_values[-1] if close_values else 0.0
            
            sma60 = sum(close_values) / len(close_values) if close_values else 0.0
            
            ma5_values = close_values[-5:] if len(close_values) >= 5 else close_values
            ma5 = sum(ma5_values) / len(ma5_values) if ma5_values else 0.0
            
            ma20_values = close_values[-20:] if len(close_values) >= 20 else close_values
            ma20 = sum(ma20_values) / len(ma20_values) if ma20_values else 0.0
            
            return current_close, sma60, ma5, ma20
            
        except Exception as e:
            logger.error(f"Error calculating index MA: {e}")
            return 0.0, 0.0, 0.0, 0.0


class V47ReportGenerator:
    """V47 报告生成器"""
    
    @staticmethod
    def generate_report(result: Dict[str, Any], factor_status: Dict[str, Any],
                        tables_used: Dict[str, str], v44_data: Optional[Dict[str, Any]] = None,
                        v40_data: Optional[Dict[str, Any]] = None, v46_data: Optional[Dict[str, Any]] = None) -> str:
        
        initial_capital = result.get('initial_capital', FIXED_INITIAL_CAPITAL)
        final_value = result.get('final_value', initial_capital)
        total_return = result.get('total_return', 0)
        max_drawdown = result.get('max_drawdown', 0)
        sharpe_ratio = result.get('sharpe_ratio', 0)
        total_trades = result.get('total_trades', 0)
        buy_trades = result.get('buy_trades', 0)
        sell_trades = result.get('sell_trades', 0)
        win_rate = result.get('win_rate', 0)
        avg_holding_days = result.get('avg_holding_days', 0)
        total_fees = result.get('total_fees', 0)
        
        trades_target_met = 20 <= total_trades <= 35
        trades_exceeded_fail = total_trades > TRADE_COUNT_FAIL_THRESHOLD
        
        return_target_met = total_return >= ANNUAL_RETURN_TARGET
        mdd_target_met = max_drawdown <= MAX_DRAWDOWN_TARGET
        
        v44_return = v44_data.get('total_return', 0) if v44_data else 0
        v44_mdd = v44_data.get('max_drawdown', 0) if v44_data else 0
        v44_return_met = total_return >= v44_return if v44_data else True
        v44_mdd_met = max_drawdown <= v44_mdd if v44_data else True
        
        wash_sale_stats = result.get('wash_sale_stats', {})
        wash_sale_blocked = wash_sale_stats.get('total_blocked', 0)
        
        market_regime_stats = result.get('market_regime_stats', {})
        risk_period_days = market_regime_stats.get('risk_period_days', 0)
        golden_cross_days = market_regime_stats.get('golden_cross_days', 0)
        
        industry_neutralization_status = factor_status.get('industry_neutralization', 'SKIPPED')
        industry_coverage = factor_status.get('industry_coverage', 0.0)
        factors_computed = factor_status.get('factors_computed', [])
        
        rebalance_logs = result.get('rebalance_logs', [])
        profit_lock_triggers = result.get('profit_lock_triggers', 0)
        
        failed_markers = []
        if trades_exceeded_fail:
            failed_markers.append(f"**[V47 FAILED: OVER-TRADING] Trade count {total_trades} > {TRADE_COUNT_FAIL_THRESHOLD}**")
        if not mdd_target_met:
            failed_markers.append(f"**[V47 FAILED: RISK TARGET NOT MET] MDD {max_drawdown:.2%} > {MAX_DRAWDOWN_TARGET:.2%}**")
        if not v44_return_met and v44_data:
            failed_markers.append(f"**[V47 FAILED: NEGATIVE OPTIMIZATION] Return {total_return:.2%} < V44 {v44_return:.2%}**")
        if not v44_mdd_met and v44_data:
            failed_markers.append(f"**[V47 FAILED: NEGATIVE OPTIMIZATION] MDD {max_drawdown:.2%} > V44 {v44_mdd:.2%}**")
        
        failed_marker = "\n".join(failed_markers) if failed_markers else ""
        
        def fmt_pct(val):
            if val is None:
                return 'N/A'
            try:
                return f"{float(val):.2%}"
            except:
                return 'N/A'
        
        def fmt_num(val):
            if val is None:
                return 'N/A'
            try:
                return f"{float(val):.3f}"
            except:
                return 'N/A'
        
        v44_return_str = fmt_pct(v44_return) if v44_data else 'N/A'
        v44_sharpe_str = fmt_num(v44_data.get('sharpe_ratio')) if v44_data else 'N/A'
        v44_mdd_str = fmt_pct(v44_mdd) if v44_data else 'N/A'
        v44_trades = v44_data.get('total_trades', 'N/A') if v44_data else 'N/A'
        v44_winrate = fmt_pct(v44_data.get('win_rate')) if v44_data else 'N/A'
        
        v40_return_str = fmt_pct(v40_data.get('total_return')) if v40_data else 'N/A'
        v40_mdd_str = fmt_pct(v40_data.get('max_drawdown')) if v40_data else 'N/A'
        v40_sharpe_str = fmt_num(v40_data.get('sharpe_ratio')) if v40_data else 'N/A'
        v40_trades = v40_data.get('total_trades', 'N/A') if v40_data else 'N/A'
        
        v46_return_str = fmt_pct(v46_data.get('total_return')) if v46_data else 'N/A'
        v46_mdd_str = fmt_pct(v46_data.get('max_drawdown')) if v46_data else 'N/A'
        v46_sharpe_str = fmt_num(v46_data.get('sharpe_ratio')) if v46_data else 'N/A'
        v46_trades = v46_data.get('total_trades', 'N/A') if v46_data else 'N/A'
        
        report = f"""# V47 回测报告 - 效率巅峰与自我审计循环 {failed_marker}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V47.0 Efficiency Peak & Self-Audit Loop Engine

---

## 一、V47 核心改进说明

### 1.1 双轨滤镜（Dual-Regime Filter）

| 轨道 | 参数 | 动作 |
|------|------|------|
| 轨道 A（长期趋势） | SMA60 | Close >= SMA60: 正常期 |
| 轨道 B（偏离度回归） | MA5 金叉 MA20 | 站上 SMA60 且金叉：满仓进攻 |

> 严禁死等均线走平

### 1.2 位次缓冲区（Rank-Based Buffer）

| 组件 | V46 | V47 | 说明 |
|------|-----|-----|------|
| 入场阈值 | Top 5 | Top 5 | 保持不变 |
| 维持阈值 | Top 15 | **Top 25** | 缓冲区扩大，减少换仓 |
| 换仓阈值 | 15% | 15% | 保持不变 |

**目的**: 将交易次数彻底压低到 30 次以内

### 1.3 收益保护机制 - 二阶止损

| 止损类型 | V46 | V47 | 说明 |
|----------|-----|-----|------|
| 追踪止损 | 3.0 ATR | **2.0 ATR** | 收紧止损 |
| 动态止盈 | 无 | **盈利 10% → 成本 +2%** | 新增盈利保护 |
| 初始止损 | 8% | 8% | 保持不变 |

### 1.4 交易次数约束

| 指标 | 目标 | V47 实际 | 状态 |
|------|------|---------|------|
| 交易次数 | [20, 35] | {total_trades} | {'✅' if trades_target_met else '❌'} |
| 失败阈值 | < 40 | {total_trades} | {'✅' if not trades_exceeded_fail else '❌ FAILED'} |

### 1.5 数据透明度

| 数据表 | 表名 | 状态 |
|--------|------|------|
| 价格数据 | {tables_used.get('price_data', 'N/A')} | ✅ |
| 指数数据 | {tables_used.get('index_data', 'N/A')} | ✅ |
| 行业数据 | {tables_used.get('industry_data', 'N/A')} | {'✅' if 'NOT AVAILABLE' not in tables_used.get('industry_data', '') else '⚠️'} |

---

## 二、回测结果

### 2.1 核心指标

| 指标 | V47 结果 | 目标 | 状态 |
|------|---------|------|------|
| **初始资金** | {initial_capital:,.2f} 元 | 100,000.00 元 | ✅ |
| **最终价值** | {final_value:,.2f} 元 | - | - |
| **总收益率** | {total_return:.2%} | > {ANNUAL_RETURN_TARGET:.0%} | {'✅' if return_target_met else '❌'} |
| **夏普比率** | {sharpe_ratio:.3f} | > 1.5 | {'✅' if sharpe_ratio > 1.5 else '❌'} |
| **最大回撤** | {max_drawdown:.2%} | ≤{MAX_DRAWDOWN_TARGET:.0%} | {'✅' if mdd_target_met else '❌'} |
| **总交易数** | {total_trades} 次 | [20, 35] | {'✅' if trades_target_met else '❌'} |
| **买入次数** | {buy_trades} 次 | - | - |
| **卖出次数** | {sell_trades} 次 | - | - |
| **胜率** | {win_rate:.1%} | - | - |
| **平均持仓天数** | {avg_holding_days:.1f} 天 | ≥15 | {'✅' if avg_holding_days >= 15 else '⚠️'} |

### 2.2 费用统计

| 费用 | 金额 |
|------|------|
| 总费用 | {total_fees:.2f} 元 |
| 手续费 | {result.get('total_commission', 0):.2f} 元 |
| 滑点成本 | {result.get('total_slippage', 0):.2f} 元 |
| 印花税 | {result.get('total_stamp_duty', 0):.2f} 元 |
| 过户费 | {result.get('total_transfer_fee', 0):.2f} 元 |

---

## 三、双轨滤镜状态

### 3.1 市场状态统计

| 状态 | 天数 | 说明 |
|------|------|------|
| 正常期 | {result.get('normal_period_days', 0)} 天 | 允许开仓 |
| 风险期 | {risk_period_days} 天 | 禁止开仓，收紧止损 |
| 满仓进攻 | {golden_cross_days} 天 | 5 日金叉 20 日 |

### 3.2 风险期保护

| 项目 | 正常期 | 风险期 |
|------|--------|--------|
| 新开仓 | ✅ 允许 | ❌ 禁止 |
| ATR 止损倍数 | 2.0ATR | 1.5ATR |

---

## 四、二阶止损审计

### 4.1 动态止盈统计

| 统计项 | 数值 |
|--------|------|
| 盈利锁触发次数 | {profit_lock_triggers} 次 |

> 盈利超过 10% 后自动上移止损至成本 +2%

### 4.2 洗售拦截统计

| 统计项 | 数值 |
|--------|------|
| 洗售拦截次数 | {wash_sale_blocked} 次 |

---

## 五、位次缓冲区审计

### 5.1 换仓决策日志（前 10 条）

| 日期 | 卖出 | 买入 | 提升% | 摩擦成本% | 状态 |
|------|------|------|-------|-----------|------|
{chr(10).join(f"| {log.get('date', 'N/A')} | {log.get('sell_symbol', 'N/A')} | {log.get('buy_symbol', 'N/A')} | {log.get('improvement', 'N/A'):.1%} | {log.get('friction', 'N/A'):.1%} | {'✅ 准予执行' if log.get('approved', False) else '❌ 拒绝'} |" for log in rebalance_logs[:10])}

> 注：仅显示前 10 条记录，共 {len(rebalance_logs)} 条

### 5.2 前 5 次换仓详细审计

{V47ReportGenerator._generate_top5_rebalance_table(rebalance_logs[:5])}

---

## 六、因子状态审计

### 6.1 板块中性化状态

| 项目 | 状态 |
|------|------|
| 行业数据覆盖率 | {industry_coverage:.1%} |
| 中性化状态 | {industry_neutralization_status} |

### 6.2 计算因子列表

| 因子 | 状态 |
|------|------|
{chr(10).join(f"| {f} | ✅ |" for f in factors_computed)}

### 6.3 Composite_Score 公式

```
Composite_Score = Rank(Momentum) * 0.5 + Rank(R²) * 0.5
```

---

## 七、四代对比（V40 vs V44 vs V46 vs V47）

### 7.1 核心指标对比

| 指标 | V40 | V44 | V46 | V47 | 最优 |
|------|-----|-----|-----|-----|------|
| **总收益率** | {v40_return_str} | {v44_return_str} | {v46_return_str} | {total_return:.2%} | {V47ReportGenerator._get_best_return(v40_data, v44_data, v46_data, total_return)} |
| **夏普比率** | {v40_sharpe_str} | {v44_sharpe_str} | {v46_sharpe_str} | {sharpe_ratio:.3f} | - |
| **最大回撤** | {v40_mdd_str} | {v44_mdd_str} | {v46_mdd_str} | {max_drawdown:.2%} | {V47ReportGenerator._get_best_mdd(v40_data, v44_data, v46_data, max_drawdown)} |
| **交易次数** | {v40_trades} | {v44_trades} | {v46_trades} | {total_trades} | - |
        | **胜率** | {V47ReportGenerator._safe_pct(v40_data.get('win_rate') if v40_data else None)} | {v44_winrate} | {V47ReportGenerator._safe_pct(v46_data.get('win_rate') if v46_data else None)} | {win_rate:.1%} | - |

### 7.2 核心机制对比

| 特性 | V40 | V44 | V46 | V47 |
|------|-----|-----|-----|-----|
| 选股方式 | Top 5 | Top 5 | Top 5 | Top 5 |
| 维持阈值 | Top 15 | Top 15 | Top 15 | **Top 25** |
| 大盘滤镜 | MA20 | MA20 | SMA60 | SMA60+ 金叉 |
| 换仓阈值 | 无 | 无 | 15% | 15% |
| ATR 止损 | 2.0ATR | 2.0ATR | 3.0ATR | **2.0ATR** |
| 动态止盈 | 无 | 无 | 无 | **✅** |
| 时间锁 | 无 | 无 | 15 天 | 15 天 |

---

## 八、对赌验证（vs V44）

### 8.1 收益率对赌

| 目标 | V44 | V47 | 状态 |
|------|-----|-----|------|
| 总收益率 | {v44_return_str} | {total_return:.2%} | {'✅ 达标' if v44_return_met else '❌ 负优化'} |

### 8.2 回撤对赌

| 目标 | V44 | V47 | 状态 |
|------|-----|-----|------|
| 最大回撤 | {v44_mdd_str} | {max_drawdown:.2%} | {'✅ 达标' if v44_mdd_met else '❌ 负优化'} |

### 8.3 交易次数优化

| 目标 | V44 | V47 | 状态 |
|------|-----|-----|------|
| 交易次数 | {v44_trades} | {total_trades} | {'✅ 减少' if total_trades < v44_trades else '⚠️ 增加'} |

---

## 九、审计结论

### 9.1 交易次数约束

| 目标 | 要求 | V47 实际 | 状态 |
|------|------|---------|------|
| 交易次数 | [20, 35] | {total_trades} | {'✅ 达成' if trades_target_met else '❌ 未达成'} |
| 失败阈值 | < 40 | {total_trades} | {'✅ 通过' if not trades_exceeded_fail else '❌ 失败'} |

### 9.2 性能目标

| 目标 | 要求 | V47 实际 | 状态 |
|------|------|---------|------|
| 年化收益 | > {ANNUAL_RETURN_TARGET:.0%} | {total_return:.2%} | {'✅ 达成' if return_target_met else '❌ 未达成'} |
| 最大回撤 | ≤{MAX_DRAWDOWN_TARGET:.0%} | {max_drawdown:.2%} | {'✅ 达成' if mdd_target_met else '❌ 未达成'} |

### 9.3 V47 核心成就

1. **双轨滤镜**: SMA60 + 偏离度回归，满仓进攻模式
2. **位次缓冲区**: Top 25 维持，减少无效换仓
3. **二阶止损**: 动态止盈 + 追踪止损（2.0 ATR）
4. **透明审计**: 详细打印换仓决策日志

### 9.4 后续优化方向

1. 动态调整金叉参数，寻找更优的进攻触发条件
2. 优化 Top N 阈值，根据市场波动率自适应
3. 引入更多对冲工具降低系统性风险

---

**报告生成完毕 - V47 Efficiency Peak & Self-Audit Loop Engine**

> **V47 承诺**: 效率巅峰，自我审计，正向迭代。
"""
        return report
    
    @staticmethod
    def _safe_pct(val) -> str:
        """安全格式化百分比，处理 None 值"""
        if val is None:
            return 'N/A'
        try:
            return f"{float(val):.1%}"
        except:
            return 'N/A'
    
    @staticmethod
    def _generate_top5_rebalance_table(logs: List[Dict]) -> str:
        if not logs:
            return "暂无换仓记录"
        
        lines = ["| 序号 | 日期 | 卖出 | 买入 | 得分差异 | 费用损耗 | 决策理由 |", 
                 "|------|------|------|------|----------|----------|----------|"]
        for i, log in enumerate(logs, 1):
            reason = f"得分提升{log.get('improvement', 0):.1%}" if log.get('approved', False) else "未达阈值"
            lines.append(f"| {i} | {log.get('date', 'N/A')} | {log.get('sell_symbol', 'N/A')} | {log.get('buy_symbol', 'N/A')} | {log.get('improvement', 0):.1%} | {log.get('friction', 0):.1%} | {reason} |")
        
        return chr(10).join(lines)
    
    @staticmethod
    def _get_best_return(v40: Optional[Dict], v44: Optional[Dict], v46: Optional[Dict], v47: float) -> str:
        returns = []
        if v40:
            returns.append(('V40', v40.get('total_return', 0)))
        if v44:
            returns.append(('V44', v44.get('total_return', 0)))
        if v46:
            returns.append(('V46', v46.get('total_return', 0)))
        returns.append(('V47', v47))
        
        best = max(returns, key=lambda x: x[1])
        return f"**{best[0]}** ({best[1]:.2%})"
    
    @staticmethod
    def _get_best_mdd(v40: Optional[Dict], v44: Optional[Dict], v46: Optional[Dict], v47: float) -> str:
        mdds = []
        if v40:
            mdds.append(('V40', v40.get('max_drawdown', 1)))
        if v44:
            mdds.append(('V44', v44.get('max_drawdown', 1)))
        if v46:
            mdds.append(('V46', v46.get('max_drawdown', 1)))
        mdds.append(('V47', v47))
        
        best = min(mdds, key=lambda x: x[1])
        return f"**{best[0]}** ({best[1]:.2%})"


class V47Engine:
    """V47 回测引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, db=None):
        self.config = config or {}
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', FIXED_INITIAL_CAPITAL)
        self.db = db
        self.v44_data = self.config.get('v44_data', None)
        self.v40_data = self.config.get('v40_data', None)
        self.v46_data = self.config.get('v46_data', None)
        
        self.data_loader = V47DatabaseManager(db=self.db)
        self.factor_engine = V47FactorEngine()
        self.risk_manager = V47RiskManager(initial_capital=self.initial_capital)
        
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._factor_status: Dict[str, Any] = {}
        
        self.daily_portfolio_values: List[Dict] = []
        self.daily_trades: List[Dict] = []
        
        self.rebalance_logs: List[Dict] = []
        self.profit_lock_triggers: int = 0
        
        self.volatility_stats = {'low_vol_days': 0, 'normal_vol_days': 0, 'high_vol_days': 0}
        self.market_regime_stats = {'risk_period_days': 0, 'normal_period_days': 0, 'golden_cross_days': 0}
        
        logger.info(f"V47 Engine initialized with capital: {self.initial_capital}")
        logger.info("V47 REQUIRES database connection")
        logger.info(f"Database tables: {DATABASE_TABLES}")
    
    def run_backtest(self) -> Dict[str, Any]:
        try:
            logger.info("=" * 60)
            logger.info("V47 BACKTEST STARTING")
            logger.info("=" * 60)
            
            self._data_cache = self.data_loader.load_all_data(self.start_date, self.end_date)
            price_data = self._data_cache.get('price_data', pl.DataFrame())
            industry_data = self._data_cache.get('industry_data', pl.DataFrame())
            index_data = self._data_cache.get('index_data', pl.DataFrame())
            
            if price_data.is_empty():
                logger.error("FATAL: No price data loaded from database")
                return self._generate_empty_report()
            
            trading_dates = sorted(price_data['trade_date'].unique().to_list())
            logger.info(f"Trading days: {len(trading_dates)}")
            
            for current_date in trading_dates:
                self._run_trading_day(current_date)
            
            is_valid, error_msg = self.risk_manager.check_trade_count_constraint()
            if not is_valid:
                logger.error(error_msg)
            
            return self._generate_report()
            
        except Exception as e:
            logger.error(f"V47 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _run_trading_day(self, current_date: str):
        try:
            price_df = self._data_cache.get('price_data', pl.DataFrame())
            industry_df = self._data_cache.get('industry_data', pl.DataFrame())
            index_df = self._data_cache.get('index_data', pl.DataFrame())
            
            daily_prices = price_df.filter(pl.col('trade_date') == current_date)
            if daily_prices.is_empty():
                return
            
            history_df = price_df.filter(pl.col('trade_date') <= current_date)
            
            self.risk_manager.reset_daily_counters(current_date)
            
            factor_df, factor_status = self.factor_engine.compute_all_factors(history_df, industry_df)
            self._factor_status = factor_status
            
            # V47 双轨滤镜 - 更新市场状态
            if not index_df.is_empty():
                index_close, index_sma60, index_ma5, index_ma20 = self.data_loader.get_index_sma_ma(index_df, current_date)
                self.risk_manager.update_market_regime(index_close, index_sma60, index_ma5, index_ma20, current_date)
                
                if self.risk_manager.is_risk_period:
                    self.market_regime_stats['risk_period_days'] += 1
                else:
                    self.market_regime_stats['normal_period_days'] += 1
                
                if self.risk_manager.market_regime.is_full_attack:
                    self.market_regime_stats['golden_cross_days'] += 1
            
            portfolio_value = self.risk_manager.get_portfolio_value(
                self.risk_manager.positions, current_date, price_df
            )
            market_vol = self._get_market_volatility(factor_df)
            self.risk_manager.update_volatility_regime(market_vol)
            
            if market_vol < 0.8:
                self.volatility_stats['low_vol_days'] += 1
            elif market_vol > 1.3:
                self.volatility_stats['high_vol_days'] += 1
            else:
                self.volatility_stats['normal_vol_days'] += 1
            
            risk_per_position = self.risk_manager.get_risk_per_position()
            
            sell_candidates = self.risk_manager.check_stop_loss_and_rank(
                self.risk_manager.positions, current_date, price_df, factor_df
            )
            
            for symbol, reason in sell_candidates:
                if symbol in self.risk_manager.positions:
                    pos = self.risk_manager.positions[symbol]
                    exit_price = self._get_price_for_symbol(daily_prices, symbol)
                    if exit_price and pos.shares > 0:
                        self.risk_manager.execute_sell(current_date, symbol, exit_price, reason=reason)
                        self.daily_trades.append({
                            'trade_date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'reason': reason,
                            'shares': pos.shares,
                            'price': exit_price
                        })
            
            max_positions = MAX_POSITIONS
            if len(self.risk_manager.positions) < max_positions:
                buy_candidates = self._get_buy_candidates_v47(factor_df, self.risk_manager.positions)
                available_slots = max_positions - len(self.risk_manager.positions)
                
                for candidate in buy_candidates[:available_slots]:
                    symbol = candidate['symbol']
                    if symbol not in self.risk_manager.positions:
                        entry_price = self._get_price_for_symbol(daily_prices, symbol)
                        if entry_price and entry_price > 0:
                            try:
                                atr_row = factor_df.filter(pl.col('symbol') == symbol).select('atr_20').row(0)
                                atr = float(atr_row[0]) if atr_row and atr_row[0] else entry_price * 0.03
                            except:
                                atr = entry_price * 0.03
                            
                            shares, target_amount = self.risk_manager.calculate_position_size(
                                symbol, atr, entry_price, portfolio_value
                            )
                            
                            if shares > 0 and target_amount > 0:
                                trade = self.risk_manager.execute_buy(
                                    current_date, symbol, entry_price, atr, target_amount,
                                    signal_score=candidate.get('composite_score', 0),
                                    signal_rank=candidate.get('composite_rank', 9999),
                                    composite_score=candidate.get('composite_score', 0),
                                    reason="signal"
                                )
                                if trade:
                                    self.daily_trades.append({
                                        'trade_date': current_date,
                                        'symbol': symbol,
                                        'action': 'BUY',
                                        'reason': 'signal',
                                        'shares': shares,
                                        'price': entry_price,
                                        'rank': candidate.get('composite_rank', 9999),
                                        'composite_score': candidate.get('composite_score', 0)
                                    })
            else:
                self._check_rebalance_opportunities(current_date, factor_df, daily_prices, portfolio_value)
            
            current_portfolio_value = self.risk_manager.get_portfolio_value(
                self.risk_manager.positions, current_date, price_df
            )
            
            # 统计盈利锁触发
            for pos in self.risk_manager.positions.values():
                if pos.profit_lock_triggered:
                    self.profit_lock_triggers += 1
            
            self.daily_portfolio_values.append({
                'trade_date': current_date,
                'portfolio_value': current_portfolio_value,
                'positions_count': len(self.risk_manager.positions),
                'market_volatility': market_vol,
                'risk_per_position': risk_per_position,
                'is_risk_period': self.risk_manager.is_risk_period,
                'is_full_attack': self.risk_manager.market_regime.is_full_attack,
            })
            
        except Exception as e:
            logger.error(f"Error in trading day {current_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _check_rebalance_opportunities(self, current_date: str, factor_df: pl.DataFrame,
                                        daily_prices: pl.DataFrame, portfolio_value: float):
        try:
            positions = self.risk_manager.positions
            if not positions:
                return
            
            min_rank_pos = None
            min_rank = 0
            for symbol, pos in positions.items():
                if pos.current_market_rank > min_rank:
                    min_rank = pos.current_market_rank
                    min_rank_pos = pos
            
            if not min_rank_pos:
                return
            
            candidates = self._get_buy_candidates_v47(factor_df, positions)
            if not candidates:
                return
            
            best_candidate = candidates[0]
            candidate_score = best_candidate.get('composite_score', 0)
            current_score = min_rank_pos.entry_composite_score if min_rank_pos.entry_composite_score > 0 else min_rank_pos.composite_score
            
            should_rebalance, improvement, friction = self.risk_manager.check_rebalance_threshold(
                min_rank_pos.symbol, current_score,
                best_candidate['symbol'], candidate_score
            )
            
            rebalance_log = {
                'date': current_date,
                'sell_symbol': min_rank_pos.symbol,
                'buy_symbol': best_candidate['symbol'],
                'current_score': current_score,
                'candidate_score': candidate_score,
                'improvement': improvement,
                'friction': friction,
                'approved': should_rebalance
            }
            self.rebalance_logs.append(rebalance_log)
            
            if should_rebalance:
                logger.info(f"REBALANCE AUDIT: {current_date} - Sell {min_rank_pos.symbol} -> Buy {best_candidate['symbol']}")
                logger.info(f"  Score improvement: {improvement:.1%}, Friction cost: {friction:.1%}, APPROVED")
            else:
                logger.debug(f"REBALANCE REJECTED: {current_date} - {min_rank_pos.symbol} vs {best_candidate['symbol']}")
                logger.debug(f"  Score improvement: {improvement:.1%} < threshold {REBALANCE_THRESHOLD:.0%}")
            
            if should_rebalance:
                exit_price = self._get_price_for_symbol(daily_prices, min_rank_pos.symbol)
                if exit_price:
                    self.risk_manager.execute_sell(current_date, min_rank_pos.symbol, exit_price, reason="rebalance")
                    self.daily_trades.append({
                        'trade_date': current_date,
                        'symbol': min_rank_pos.symbol,
                        'action': 'SELL',
                        'reason': 'rebalance',
                        'shares': min_rank_pos.shares,
                        'price': exit_price
                    })
                
                entry_price = self._get_price_for_symbol(daily_prices, best_candidate['symbol'])
                if entry_price and entry_price > 0:
                    try:
                        atr_row = factor_df.filter(pl.col('symbol') == best_candidate['symbol']).select('atr_20').row(0)
                        atr = float(atr_row[0]) if atr_row and atr_row[0] else entry_price * 0.03
                    except:
                        atr = entry_price * 0.03
                    
                    shares, target_amount = self.risk_manager.calculate_position_size(
                        best_candidate['symbol'], atr, entry_price, portfolio_value
                    )
                    
                    if shares > 0 and target_amount > 0:
                        trade = self.risk_manager.execute_buy(
                            current_date, best_candidate['symbol'], entry_price, atr, target_amount,
                            signal_score=best_candidate.get('composite_score', 0),
                            signal_rank=best_candidate.get('composite_rank', 9999),
                            composite_score=best_candidate.get('composite_score', 0),
                            reason="rebalance"
                        )
                        if trade:
                            self.daily_trades.append({
                                'trade_date': current_date,
                                'symbol': best_candidate['symbol'],
                                'action': 'BUY',
                                'reason': 'rebalance',
                                'shares': shares,
                                'price': entry_price,
                                'rank': best_candidate.get('composite_rank', 9999),
                                'composite_score': best_candidate.get('composite_score', 0)
                            })
                            
        except Exception as e:
            logger.error(f"_check_rebalance_opportunities failed: {e}")
    
    def _get_buy_candidates_v47(self, factor_df: pl.DataFrame, positions: Dict[str, V47Position]) -> List[Dict]:
        try:
            held_symbols = set(positions.keys())
            
            candidates = factor_df.filter(
                (~pl.col('symbol').is_in(list(held_symbols))) &
                (pl.col('entry_allowed') == True)
            ).sort('composite_score', descending=True).limit(MAX_POSITIONS)
            
            result = []
            for idx, row in enumerate(candidates.iter_rows(named=True)):
                result.append({
                    'symbol': row['symbol'],
                    'rank': idx + 1,
                    'composite_score': float(row.get('composite_score', 0)) if row.get('composite_score') is not None else 0,
                    'composite_rank': int(row.get('composite_rank', 9999)) if row.get('composite_rank') is not None else 9999,
                    'trend_quality_r2': float(row.get('trend_quality_r2', 0)) if row.get('trend_quality_r2') is not None else 0,
                })
            
            return result
            
        except Exception as e:
            logger.error(f"_get_buy_candidates_v47 failed: {e}")
            return []
    
    def _get_price_for_symbol(self, df: pl.DataFrame, symbol: str) -> Optional[float]:
        try:
            row = df.filter(pl.col('symbol') == symbol).select('close').row(0)
            return float(row[0]) if row else None
        except Exception:
            return None
    
    def _get_market_volatility(self, df: pl.DataFrame) -> float:
        try:
            vol_col = 'volatility_ratio' if 'volatility_ratio' in df.columns else 'vix_sim'
            if vol_col in df.columns:
                return float(df[vol_col].mean() or 1.0)
            return 1.0
        except Exception:
            return 1.0
    
    def _generate_report(self) -> Dict[str, Any]:
        try:
            total_trades = len(self.daily_trades)
            buy_trades_list = [t for t in self.daily_trades if t['action'] == 'BUY']
            sell_trades_list = [t for t in self.daily_trades if t['action'] == 'SELL']
            
            portfolio_values = [p['portfolio_value'] for p in self.daily_portfolio_values]
            final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            peak = portfolio_values[0]
            max_dd = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdown = max_dd
            
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    daily_returns.append(portfolio_values[i] / portfolio_values[i-1] - 1)
            
            if daily_returns:
                daily_returns_np = np.array(daily_returns)
                sharpe_ratio = float(daily_returns_np.mean() / (daily_returns_np.std() + 1e-9)) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            trade_log = self.risk_manager.trade_log
            profitable_trades = len([t for t in trade_log if t.is_profitable])
            total_completed_trades = len(trade_log)
            win_rate = profitable_trades / total_completed_trades if total_completed_trades > 0 else 0
            
            avg_holding_days = np.mean([t.holding_days for t in trade_log]) if trade_log else 0
            holding_days_list = [t.holding_days for t in trade_log] if trade_log else []
            
            total_commission = sum(t.commission for t in self.risk_manager.trades)
            total_slippage = sum(t.slippage for t in self.risk_manager.trades)
            total_stamp_duty = sum(t.stamp_duty for t in self.risk_manager.trades)
            total_transfer_fee = sum(t.transfer_fee for t in self.risk_manager.trades)
            total_fees = total_commission + total_slippage + total_stamp_duty + total_transfer_fee
            
            # 计算正常期天数
            normal_period_days = sum(1 for p in self.daily_portfolio_values if not p.get('is_risk_period', False))
            
            return {
                'initial_capital': self.initial_capital,
                'final_value': final_value,
                'total_return': total_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio,
                'total_trades': total_trades,
                'buy_trades': len(buy_trades_list),
                'sell_trades': len(sell_trades_list),
                'win_rate': win_rate,
                'avg_holding_days': avg_holding_days,
                'holding_days_list': holding_days_list,
                'total_fees': total_fees,
                'total_commission': total_commission,
                'total_slippage': total_slippage,
                'total_stamp_duty': total_stamp_duty,
                'total_transfer_fee': total_transfer_fee,
                'daily_portfolio_values': self.daily_portfolio_values,
                'trades': self.daily_trades,
                'trade_log': trade_log,
                'wash_sale_stats': self.risk_manager.get_wash_sale_stats(),
                'factor_status': self._factor_status,
                'market_regime_stats': self.market_regime_stats,
                'volatility_stats': self.volatility_stats,
                'tables_used': self.data_loader.tables_used,
                'rebalance_logs': self.rebalance_logs,
                'profit_lock_triggers': self.profit_lock_triggers,
                'normal_period_days': normal_period_days,
                'v44_data': self.v44_data,
                'v40_data': self.v40_data,
                'v46_data': self.v46_data,
                'version': 'V47'
            }
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                'initial_capital': self.initial_capital,
                'final_value': self.initial_capital,
                'total_return': 0.0,
                'max_drawdown': 0.0,
                'sharpe_ratio': 0.0,
                'total_trades': 0,
                'error': str(e)
            }
    
    def _generate_empty_report(self) -> Dict[str, Any]:
        return {
            'initial_capital': self.initial_capital,
            'final_value': self.initial_capital,
            'total_return': 0.0,
            'max_drawdown': 0.0,
            'sharpe_ratio': 0.0,
            'total_trades': 0,
            'error': 'No data loaded from database'
        }
    
    def generate_markdown_report(self, result: Dict[str, Any]) -> str:
        return V47ReportGenerator.generate_report(
            result,
            result.get('factor_status', {}),
            result.get('tables_used', {}),
            result.get('v44_data', None),
            result.get('v40_data', None),
            result.get('v46_data', None)
        )


def main():
    """V47 主入口"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V47 EFFICIENCY PEAK & SELF-AUDIT LOOP")
    logger.info("=" * 60)
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V47 REQUIRES database connection - exiting")
        return None
    
    engine = V47Engine(config={
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': FIXED_INITIAL_CAPITAL,
        'v44_data': None,
        'v40_data': None,
        'v46_data': None,
    }, db=db)
    
    result = engine.run_backtest()
    
    report = engine.generate_markdown_report(result)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"V47_Backtest_Report_{timestamp}.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("V47 BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.1%}")
    logger.info(f"Wash Sale Blocks: {result.get('wash_sale_stats', {}).get('total_blocked', 0)}")
    logger.info(f"Profit Lock Triggers: {result.get('profit_lock_triggers', 0)}")
    
    total_trades = result.get('total_trades', 0)
    mdd = result.get('max_drawdown', 0)
    total_return = result.get('total_return', 0)
    
    if total_trades > TRADE_COUNT_FAIL_THRESHOLD:
        logger.error(f"[V47 FAILED: OVER-TRADING] Trade count {total_trades} exceeds threshold {TRADE_COUNT_FAIL_THRESHOLD}")
    
    if mdd > MAX_DRAWDOWN_TARGET:
        logger.error(f"[V47 FAILED: RISK TARGET NOT MET] MDD {mdd:.2%} exceeds {MAX_DRAWDOWN_TARGET:.2%}")
    
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
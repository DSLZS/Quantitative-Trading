"""
V50 Engine - 自适应动量与信号纯化回测引擎

【V50 回测引擎架构】
1. 双重动量确认：Score = Rank(Momentum)*0.4 + Rank(R²)*0.6
2. 趋势确认：MA5 > MA20 且 股价 > MA20
3. 分档动态止损 (Tiered-Exit)
4. 位次缓冲带 (Rank Buffer): Top 10 入场，Top 30 维持
5. 个股波动率头寸管理
6. 对赌协议验证：年化收益 > 15%, MDD < 4%, 盈亏比 > 3:1

作者：量化系统
版本：V50.0
日期：2026-03-20
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

from v50_core import (
    V50FactorEngine,
    V50RiskManager,
    V50Position,
    V50TradeAudit,
    V50MarketRegime,
    FIXED_INITIAL_CAPITAL,
    MAX_POSITIONS,
    DATABASE_TABLES,
    TRADE_COUNT_FAIL_THRESHOLD,
    ANNUAL_RETURN_TARGET,
    MAX_DRAWDOWN_TARGET,
    PROFIT_LOSS_RATIO_TARGET,
    MAINTAIN_TOP_N,
    ENTRY_TOP_N,
    MIN_TRADES_TARGET,
    MAX_TRADES_TARGET,
)


class V50DatabaseManager:
    """V50 数据库管理器"""
    
    def __init__(self, db=None):
        self.db = db
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self.tables_used: Dict[str, str] = {}
    
    def load_all_data(self, start_date: str, end_date: str) -> Dict[str, pl.DataFrame]:
        logger.info(f"Loading data from database: {start_date} to {end_date}...")
        
        if self.db is None:
            logger.error("DATABASE ERROR: No database connection provided")
            raise ValueError("DATABASE REQUIRED: V50 requires database connection")
        
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


class V50ReportGenerator:
    """V50 报告生成器"""
    
    @staticmethod
    def generate_report(result: Dict[str, Any], factor_status: Dict[str, Any],
                        tables_used: Dict[str, str],
                        v49_data: Optional[Dict[str, Any]] = None,
                        v44_data: Optional[Dict[str, Any]] = None,
                        v40_data: Optional[Dict[str, Any]] = None) -> str:
        
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
        
        # 对赌协议验证
        return_target_met = total_return >= ANNUAL_RETURN_TARGET
        mdd_target_met = max_drawdown <= MAX_DRAWDOWN_TARGET
        
        # 盈亏比计算
        max_profit = result.get('max_single_profit', 0)
        max_loss = result.get('max_single_loss', 0)
        profit_loss_ratio = abs(max_profit / max_loss) if max_loss != 0 else float('inf')
        plr_target_met = profit_loss_ratio >= PROFIT_LOSS_RATIO_TARGET
        
        # V49 对比
        v49_return = v49_data.get('total_return', 0) if v49_data else 0
        v49_mdd = v49_data.get('max_drawdown', 0) if v49_data else 0
        v49_return_met = total_return >= v49_return if v49_data else True
        v49_mdd_met = max_drawdown <= v49_mdd if v49_data else True
        
        # V44 对比
        v44_return = v44_data.get('total_return', 0) if v44_data else 0
        v44_mdd = v44_data.get('max_drawdown', 0) if v44_data else 0
        v44_return_met = total_return >= v44_return if v44_data else True
        v44_mdd_met = max_drawdown <= v44_mdd if v44_data else True
        
        # V40 对比
        v40_return = v40_data.get('total_return', 0) if v40_data else 0
        v40_mdd = v40_data.get('max_drawdown', 0) if v40_data else 0
        
        # 交易次数检查
        over_trading_warning = ""
        if total_trades > TRADE_COUNT_FAIL_THRESHOLD:
            over_trading_warning = f"\n\n> **⚠️ [V50 OVER-TRADING FAILURE]**: 交易次数 {total_trades} 次，超过 {TRADE_COUNT_FAIL_THRESHOLD} 次限制！"
        
        # 对赌协议失败标记
        failed_markers = []
        if not return_target_met:
            failed_markers.append(f"**[V50 FAILED: RETURN TARGET]** Return {total_return:.2%} < {ANNUAL_RETURN_TARGET:.0%}**")
        if not mdd_target_met:
            failed_markers.append(f"**[V50 FAILED: RISK TARGET]** MDD {max_drawdown:.2%} > {MAX_DRAWDOWN_TARGET:.0%}**")
        if not plr_target_met and max_loss != 0:
            failed_markers.append(f"**[V50 FAILED: PROFIT/LOSS RATIO]** P/L Ratio {profit_loss_ratio:.2f} < {PROFIT_LOSS_RATIO_TARGET:.1f}**")
        if total_trades > TRADE_COUNT_FAIL_THRESHOLD:
            failed_markers.append(f"**[V50 OVER-TRADING]** Trade count {total_trades} > {TRADE_COUNT_FAIL_THRESHOLD}**")
        
        failed_marker = "\n".join(failed_markers) if failed_markers else ""
        
        # 对赌协议结果
        bet_result = "✅ **对赌协议成功**" if (return_target_met and mdd_target_met) else "❌ **对赌协议失败**"
        
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
        
        v49_return_str = fmt_pct(v49_return) if v49_data else 'N/A'
        v49_mdd_str = fmt_pct(v49_mdd) if v49_data else 'N/A'
        v49_sharpe_str = fmt_num(v49_data.get('sharpe_ratio')) if v49_data else 'N/A'
        v49_trades = v49_data.get('total_trades', 0) if v49_data else 0
        
        v44_return_str = fmt_pct(v44_return) if v44_data else 'N/A'
        v44_mdd_str = fmt_pct(v44_mdd) if v44_data else 'N/A'
        v44_sharpe_str = fmt_num(v44_data.get('sharpe_ratio')) if v44_data else 'N/A'
        v44_trades = v44_data.get('total_trades', 0) if v44_data else 0
        
        v40_return_str = fmt_pct(v40_return) if v40_data else 'N/A'
        v40_mdd_str = fmt_pct(v40_mdd) if v40_data else 'N/A'
        v40_sharpe_str = fmt_num(v40_data.get('sharpe_ratio')) if v40_data else 'N/A'
        v40_trades = v40_data.get('total_trades', 0) if v40_data else 0
        
        momentum_weight = factor_status.get('momentum_weight', 0.4)
        r2_weight = factor_status.get('r2_weight', 0.6)
        
        # 分档止损统计
        tier1_triggers = result.get('tier1_triggers', 0)
        tier2_triggers = result.get('tier2_triggers', 0)
        ma20_breaks = result.get('ma20_breaks', 0)
        
        report = f"""# V50 回测报告 - 自适应动量与信号纯化 {failed_marker}{over_trading_warning}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V50.0 Adaptive Momentum & Signal Purification Engine
**回测区间**: {result.get('start_date', 'N/A')} 至 {result.get('end_date', 'N/A')}

---

## 一、V50 核心改进说明

### 1.1 逻辑大拆除（废除 V49 枷锁）

| 废除项 | V49 设置 | V50 改进 |
|--------|---------|---------|
| 时间锁 | 15 天强制持有 | ❌ 废除 - 灵活持有期 |
| 换仓阈值 | 30% 得分提升 | ❌ 废除 - 位次缓冲带 |
| 强制减仓 | 回撤>3% 降至 10% | ❌ 废除 - 个股波动率头寸 |

### 1.2 双重动量确认 (Dual-Momentum Confirm)

```
Composite_Score = Rank(Momentum) * {momentum_weight} + Rank(R²) * {r2_weight}
```

| 条件 | V50 要求 |
|------|---------|
| 选股范围 | Top {ENTRY_TOP_N} |
| 趋势确认 | MA5 > MA20 |
| 价格确认 | 股价 > MA20 |
| 维持缓冲 | Top {MAINTAIN_TOP_N} |

### 1.3 分档动态止损 (Tiered-Exit)

| 阶段 | 浮盈范围 | 止损策略 |
|------|---------|---------|
| 初段保护 | < 5% | 2.5 * ATR |
| 中段护航 | 5% ~ 15% | 1.5 * ATR 追踪 |
| 高段奔跑 | > 15% | 跌破 MA20 清仓 |

### 1.4 对赌协议

| 目标 | 要求 | 状态 |
|------|------|------|
| 年化收益 | > {ANNUAL_RETURN_TARGET:.0%} | {'✅' if return_target_met else '❌'} |
| 最大回撤 | ≤ {MAX_DRAWDOWN_TARGET:.0%} | {'✅' if mdd_target_met else '❌'} |
| 盈亏比 | > {PROFIT_LOSS_RATIO_TARGET:.1f}:1 | {'✅' if plr_target_met else '❌'} |

**对赌结果**: {bet_result}

---

## 二、回测结果

### 2.1 核心指标

| 指标 | V50 结果 | 目标 | 状态 |
|------|---------|------|------|
| **初始资金** | {initial_capital:,.2f} 元 | 100,000.00 元 | ✅ |
| **最终价值** | {final_value:,.2f} 元 | - | - |
| **总收益率** | {total_return:.2%} | > {ANNUAL_RETURN_TARGET:.0%} | {'✅' if return_target_met else '❌'} |
| **夏普比率** | {sharpe_ratio:.3f} | > 1.5 | {'✅' if sharpe_ratio > 1.5 else '❌'} |
| **最大回撤** | {max_drawdown:.2%} | ≤ {MAX_DRAWDOWN_TARGET:.0%} | {'✅' if mdd_target_met else '❌'} |
| **总交易数** | {total_trades} 次 | [{MIN_TRADES_TARGET}, {MAX_TRADES_TARGET}] | {'✅' if MIN_TRADES_TARGET <= total_trades <= MAX_TRADES_TARGET else '⚠️'} |
| **买入次数** | {buy_trades} 次 | - | - |
| **卖出次数** | {sell_trades} 次 | - | - |
| **胜率** | {win_rate:.1%} | - | - |
| **平均持仓天数** | {avg_holding_days:.1f} 天 | - | - |

### 2.2 费用统计

| 费用 | 金额 |
|------|------|
| 总费用 | {total_fees:.2f} 元 |
| 手续费 | {result.get('total_commission', 0):.2f} 元 |
| 滑点成本 (0.1%) | {result.get('total_slippage', 0):.2f} 元 |
| 印花税 | {result.get('total_stamp_duty', 0):.2f} 元 |
| 过户费 | {result.get('total_transfer_fee', 0):.2f} 元 |

---

## 三、V50 特有审计

### 3.1 分档止损统计

| 阶段 | 触发次数 | 说明 |
|------|---------|------|
| 初段 (2.5*ATR) | {tier1_triggers} | 浮盈<5% 触发 |
| 中段 (1.5*ATR) | {tier2_triggers} | 浮盈 5%~15% 触发 |
| 高段 (MA20) | {ma20_breaks} | 浮盈>15% 跌破 MA20 |

### 3.2 位次缓冲带统计

| 统计项 | 数值 |
|--------|------|
| 入场选股范围 | Top {ENTRY_TOP_N} |
| 维持缓冲范围 | Top {MAINTAIN_TOP_N} |
| 因排名下跌卖出 | {result.get('rank_drop_sells', 0)} 次 |

### 3.3 洗售审计

| 统计项 | 数值 |
|--------|------|
| 洗售拦截次数 | {result.get('wash_sale_blocks', 0)} 次 |

### 3.4 进场黑名单

| 统计项 | 数值 |
|--------|------|
| 黑名单拦截次数 | {result.get('blacklist_blocks', 0)} 次 |

### 3.5 透明审计 - 盈亏分析

| 统计项 | 数值 |
|--------|------|
| **单笔交易最大盈利** | {max_profit:.2f} 元 |
| **单笔交易最大亏损** | {max_loss:.2f} 元 |
| **盈亏比** | {profit_loss_ratio:.2f}:1 |
| 盈亏比目标 | > {PROFIT_LOSS_RATIO_TARGET:.1f}:1 |
| 状态 | {'✅ 达标' if plr_target_met else '❌ 未达标'} |

---

## 四、数据透明度

| 数据表 | 表名 | 状态 |
|--------|------|------|
| 价格数据 | {tables_used.get('price_data', 'N/A')} | ✅ |
| 指数数据 | {tables_used.get('index_data', 'N/A')} | ✅ |
| 行业数据 | {tables_used.get('industry_data', 'N/A')} | {'✅' if 'NOT AVAILABLE' not in tables_used.get('industry_data', '') else '⚠️'} |

---

## 五、四代对比（V40 vs V44 vs V49 vs V50）

### 5.1 核心指标对比

| 指标 | V40 | V44 | V49 | V50 | 最优 |
|------|-----|-----|-----|-----|------|
| **总收益率** | {v40_return_str} | {v44_return_str} | {v49_return_str} | {total_return:.2%} | {V50ReportGenerator._get_best_return(v40_data, v44_data, v49_data, total_return)} |
| **夏普比率** | {v40_sharpe_str} | {v44_sharpe_str} | {v49_sharpe_str} | {sharpe_ratio:.3f} | - |
| **最大回撤** | {v40_mdd_str} | {v44_mdd_str} | {v49_mdd_str} | {max_drawdown:.2%} | {V50ReportGenerator._get_best_mdd(v40_data, v44_data, v49_data, max_drawdown)} |
| **交易次数** | {v40_trades} | {v44_trades} | {v49_trades} | {total_trades} | - |
| **胜率** | {V50ReportGenerator._safe_fmt(v40_data.get('win_rate') if v40_data else None, '%')} | {V50ReportGenerator._safe_fmt(v44_data.get('win_rate') if v44_data else None, '%')} | {V50ReportGenerator._safe_fmt(v49_data.get('win_rate') if v49_data else None, '%')} | {win_rate:.1%} | - |

### 5.2 核心机制对比

| 特性 | V40 | V44 | V49 | V50 |
|------|-----|-----|-----|-----|
| 选股方式 | Top 50 | Top 5 | Top 5% | Top {ENTRY_TOP_N} |
| 维持阈值 | Top 50 | Top 15 | Top 30% | Top {MAINTAIN_TOP_N} |
| 大盘滤镜 | 无 | MA20 | SMA60+ 熔断 | SMA60+ 熔断 |
| ATR 止损 | 2.0ATR | 2.0ATR | 3.0ATR→1.5ATR | 分档动态 |
| 动态止盈 | 无 | 无 | 7% 触发 | 三档 |
| 时间锁 | 无 | 无 | 15 天 | ❌ 废除 |
| 换仓阈值 | - | - | 30% | ❌ 废除 |
| 强制减仓 | 无 | 无 | 回撤>3% | ❌ 废除 |
| 动量权重 | - | 0.5 | 0.3 | {momentum_weight} |
| R²权重 | - | 0.5 | 0.7 | {r2_weight} |

---

## 六、对赌验证

### 6.1 收益率对赌

| 对比 | V40 | V44 | V49 | V50 | 状态 |
|------|-----|-----|-----|-----|------|
| 总收益率 | {v40_return_str} | {v44_return_str} | {v49_return_str} | {total_return:.2%} | {'✅' if return_target_met else '❌'} |

### 6.2 回撤对赌

| 对比 | V40 | V44 | V49 | V50 | 状态 |
|------|-----|-----|-----|-----|------|
| 最大回撤 | {v40_mdd_str} | {v44_mdd_str} | {v49_mdd_str} | {max_drawdown:.2%} | {'✅' if mdd_target_met else '❌'} |

### 6.3 交易次数优化

| 目标 | 要求 | V50 实际 | 状态 |
|------|------|---------|------|
| 交易次数 | [{MIN_TRADES_TARGET}, {MAX_TRADES_TARGET}] | {total_trades} | {'✅' if MIN_TRADES_TARGET <= total_trades <= MAX_TRADES_TARGET else '⚠️'} |

### 6.4 盈亏比对赌

| 目标 | 要求 | V50 实际 | 状态 |
|------|------|---------|------|
| 盈亏比 | > {PROFIT_LOSS_RATIO_TARGET:.1f}:1 | {profit_loss_ratio:.2f}:1 | {'✅' if plr_target_met else '❌'} |

---

## 七、审计结论

### 7.1 性能目标

| 目标 | 要求 | V50 实际 | 状态 |
|------|------|---------|------|
| 年化收益 | > {ANNUAL_RETURN_TARGET:.0%} | {total_return:.2%} | {'✅ 达成' if return_target_met else '❌ 未达成'} |
| 最大回撤 | ≤ {MAX_DRAWDOWN_TARGET:.0%} | {max_drawdown:.2%} | {'✅ 达成' if mdd_target_met else '❌ 未达成'} |
| 盈亏比 | > {PROFIT_LOSS_RATIO_TARGET:.1f}:1 | {profit_loss_ratio:.2f}:1 | {'✅ 达成' if plr_target_met else '❌ 未达成'} |

### 7.2 V50 核心成就

1. **废除时间锁**: 灵活持有期，根据趋势状态决定卖出
2. **废除换仓阈值**: 位次缓冲带 (Top {MAINTAIN_TOP_N}) 自然控制交易频率
3. **废除强制减仓**: 个股波动率头寸管理，更精准的风险控制
4. **分档动态止损**: 三档策略让亏损有限，利润奔跑
5. **双重动量确认**: Score = Rank(Momentum)*{momentum_weight} + Rank(R²)*{r2_weight}

### 7.3 反思与改进

{V50ReportGenerator._generate_reflection(return_target_met, mdd_target_met, plr_target_met, max_profit, max_loss)}

---

**报告生成完毕 - V50 Adaptive Momentum & Signal Purification Engine**

> **V50 承诺**: 自适应动量，信号纯化，让牛股奔跑。
"""
        return report
    
    @staticmethod
    def _safe_fmt(val, fmt_type: str = '%') -> str:
        if val is None:
            return 'N/A'
        try:
            if fmt_type == '%':
                return f"{float(val):.1%}"
            else:
                return f"{float(val):.3f}"
        except:
            return 'N/A'
    
    @staticmethod
    def _get_best_return(v40: Optional[Dict], v44: Optional[Dict], v49: Optional[Dict], v50: float) -> str:
        returns = []
        if v40:
            returns.append(('V40', v40.get('total_return', 0)))
        if v44:
            returns.append(('V44', v44.get('total_return', 0)))
        if v49:
            returns.append(('V49', v49.get('total_return', 0)))
        returns.append(('V50', v50))
        
        best = max(returns, key=lambda x: x[1])
        return f"**{best[0]}** ({best[1]:.2%})"
    
    @staticmethod
    def _get_best_mdd(v40: Optional[Dict], v44: Optional[Dict], v49: Optional[Dict], v50: float) -> str:
        mdds = []
        if v40:
            mdds.append(('V40', v40.get('max_drawdown', 1)))
        if v44:
            mdds.append(('V44', v44.get('max_drawdown', 1)))
        if v49:
            mdds.append(('V49', v49.get('max_drawdown', 1)))
        mdds.append(('V50', v50))
        
        best = min(mdds, key=lambda x: x[1])
        return f"**{best[0]}** ({best[1]:.2%})"
    
    @staticmethod
    def _generate_reflection(return_met: bool, mdd_met: bool, plr_met: bool, 
                             max_profit: float, max_loss: float) -> str:
        """生成反思内容"""
        reflections = []
        
        if not return_met:
            reflections.append(f"- **收益率未达标**: 实际 {ANNUAL_RETURN_TARGET:.0%} 目标未达成，需要检查入场信号质量或考虑优化动量/R²权重")
        
        if not mdd_met:
            reflections.append(f"- **回撤超标**: 实际回撤超过 {MAX_DRAWDOWN_TARGET:.0%}，需要考虑收紧初始止损或降低仓位")
        
        if not plr_met and max_loss != 0:
            reflections.append(f"- **盈亏比不足**: 最大盈利 {max_profit:.2f} 元 vs 最大亏损 {max_loss:.2f} 元，需要让盈利单奔跑更久或更快截断亏损")
        
        if return_met and mdd_met and plr_met:
            return "✅ **所有对赌目标达成** - V50 策略表现优异，可以进行实盘部署。"
        
        return "\n".join(reflections) if reflections else "策略表现良好，继续优化。"


class V50Engine:
    """V50 回测引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, db=None):
        self.config = config or {}
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', FIXED_INITIAL_CAPITAL)
        self.db = db
        self.v49_data = self.config.get('v49_data', None)
        self.v44_data = self.config.get('v44_data', None)
        self.v40_data = self.config.get('v40_data', None)
        
        self.data_loader = V50DatabaseManager(db=self.db)
        self.factor_engine = V50FactorEngine()
        self.risk_manager = V50RiskManager(initial_capital=self.initial_capital)
        
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._factor_status: Dict[str, Any] = {}
        
        self.daily_portfolio_values: List[Dict] = []
        self.daily_trades: List[Dict] = []
        
        # V50 分档止损统计
        self.tier1_triggers: int = 0
        self.tier2_triggers: int = 0
        self.ma20_breaks: int = 0
        
        logger.info(f"V50 Engine initialized with capital: {self.initial_capital}")
        logger.info("V50 REQUIRES database connection")
        logger.info(f"Database tables: {DATABASE_TABLES}")
    
    def run_backtest(self) -> Dict[str, Any]:
        try:
            logger.info("=" * 60)
            logger.info("V50 BACKTEST STARTING")
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
            
            result = self._generate_report()
            
            return result
            
        except Exception as e:
            logger.error(f"V50 backtest FAILED: {e}")
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
            
            # 更新市场状态
            if not index_df.is_empty():
                index_close, index_sma60, index_ma5, index_ma20 = self.data_loader.get_index_sma_ma(index_df, current_date)
                self._update_market_regime(index_close, index_sma60, index_ma5, index_ma20, current_date)
            
            portfolio_value = self.risk_manager.get_portfolio_value(
                self.risk_manager.positions, current_date, price_df
            )
            market_vol = self._get_market_volatility(factor_df)
            self.risk_manager.update_volatility_regime(market_vol)
            
            risk_per_position = self.risk_manager.get_risk_per_position()
            
            # V50 分档动态止损检查
            sell_candidates = self.risk_manager.check_tiered_exit(
                self.risk_manager.positions, current_date, price_df, factor_df
            )
            
            for symbol, reason in sell_candidates:
                if symbol in self.risk_manager.positions:
                    pos = self.risk_manager.positions[symbol]
                    exit_price = self._get_price_for_symbol(daily_prices, symbol)
                    if exit_price and pos.shares > 0:
                        if reason == "ma20_break":
                            self.ma20_breaks += 1
                        elif reason == "trailing_stop":
                            if pos.profit_tier2_triggered:
                                self.tier2_triggers += 1
                            else:
                                self.tier1_triggers += 1
                        
                        self.risk_manager.execute_sell(current_date, symbol, exit_price, reason=reason)
                        self.daily_trades.append({
                            'trade_date': current_date,
                            'symbol': symbol,
                            'action': 'SELL',
                            'reason': reason,
                            'shares': pos.shares,
                            'price': exit_price
                        })
            
            # 检查买入
            max_positions = MAX_POSITIONS
            if len(self.risk_manager.positions) < max_positions:
                buy_candidates = self._get_buy_candidates_v50(factor_df, self.risk_manager.positions)
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
                            
                            try:
                                ma5_row = factor_df.filter(pl.col('symbol') == symbol).select('ma5').row(0)
                                ma5 = float(ma5_row[0]) if ma5_row and ma5_row[0] else 0
                            except:
                                ma5 = 0
                            
                            try:
                                ma20_row = factor_df.filter(pl.col('symbol') == symbol).select('ma20').row(0)
                                ma20 = float(ma20_row[0]) if ma20_row and ma20_row[0] else 0
                            except:
                                ma20 = 0
                            
                            shares, target_amount = self.risk_manager.calculate_position_size(
                                symbol, atr, entry_price, portfolio_value
                            )
                            
                            if shares > 0 and target_amount > 0:
                                trade = self.risk_manager.execute_buy(
                                    current_date, symbol, entry_price, atr, target_amount,
                                    signal_score=candidate.get('composite_score', 0),
                                    signal_rank=candidate.get('composite_rank', 9999),
                                    composite_score=candidate.get('composite_score', 0),
                                    composite_percentile=candidate.get('composite_percentile', 0),
                                    ma5=ma5, ma20=ma20,
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
                                        'composite_score': candidate.get('composite_score', 0),
                                        'composite_percentile': candidate.get('composite_percentile', 0)
                                    })
            
            current_portfolio_value = self.risk_manager.get_portfolio_value(
                self.risk_manager.positions, current_date, price_df
            )
            
            self.daily_portfolio_values.append({
                'trade_date': current_date,
                'portfolio_value': current_portfolio_value,
                'positions_count': len(self.risk_manager.positions),
                'market_volatility': market_vol,
                'risk_per_position': risk_per_position,
                'is_risk_period': self.risk_manager.is_risk_period,
            })
            
        except Exception as e:
            logger.error(f"Error in trading day {current_date}: {e}")
            logger.error(traceback.format_exc())
    
    def _update_market_regime(self, index_close: float, index_sma60: float, 
                              index_ma5: float, index_ma20: float, trade_date: str):
        """更新市场状态"""
        is_risk = index_close < index_sma60
        is_golden_cross = index_ma5 > index_ma20
        is_full_attack = (not is_risk) and is_golden_cross
        
        if is_full_attack:
            regime_reason = f"Full Attack: Close({index_close:.2f})>=SMA60({index_sma60:.2f}) & MA5({index_ma5:.2f})>MA20({index_ma20:.2f})"
        elif is_risk:
            regime_reason = f"Risk: Close({index_close:.2f}) < SMA60({index_sma60:.2f})"
        else:
            regime_reason = f"Normal: Close({index_close:.2f}) >= SMA60({index_sma60:.2f})"
        
        self.risk_manager.market_regime = V50MarketRegime(
            trade_date=trade_date,
            index_close=index_close,
            index_sma60=index_sma60,
            index_ma5=index_ma5,
            index_ma20=index_ma20,
            is_risk_period=is_risk,
            is_golden_cross=is_golden_cross,
            is_full_attack=is_full_attack,
            regime_reason=regime_reason
        )
        
        self.risk_manager.is_risk_period = self.risk_manager.market_regime.is_risk_period
    
    def _get_buy_candidates_v50(self, factor_df: pl.DataFrame, positions: Dict[str, V50Position]) -> List[Dict]:
        """获取买入候选 - Top 10"""
        try:
            held_symbols = set(positions.keys())
            
            candidates = factor_df.filter(
                (~pl.col('symbol').is_in(list(held_symbols))) &
                (pl.col('entry_allowed') == True)
            ).sort('composite_score', descending=True).limit(MAX_POSITIONS * 2)
            
            result = []
            for idx, row in enumerate(candidates.iter_rows(named=True)):
                result.append({
                    'symbol': row['symbol'],
                    'rank': idx + 1,
                    'composite_score': float(row.get('composite_score', 0)) if row.get('composite_score') is not None else 0,
                    'composite_rank': int(row.get('composite_rank', 9999)) if row.get('composite_rank') is not None else 9999,
                    'composite_percentile': float(row.get('composite_percentile', 0)) if row.get('composite_percentile') is not None else 0,
                    'trend_quality_r2': float(row.get('trend_quality_r2', 0)) if row.get('trend_quality_r2') is not None else 0,
                })
            
            return result
            
        except Exception as e:
            logger.error(f"_get_buy_candidates_v50 failed: {e}")
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
            
            # 透明审计 - 最大盈利/亏损
            max_profit = max([t.gross_pnl for t in trade_log], default=0)
            max_loss = min([t.gross_pnl for t in trade_log], default=0)
            
            # V50 特有统计
            wash_sale_stats = self.risk_manager.get_wash_sale_stats()
            blacklist_stats = self.risk_manager.get_blacklist_stats()
            position_sizing_stats = self.risk_manager.get_position_sizing_stats()
            
            # 计算排名下跌卖出次数
            rank_drop_sells = len([t for t in trade_log if t.sell_reason == 'rank_drop'])
            
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
                'wash_sale_stats': wash_sale_stats,
                'factor_status': self._factor_status,
                'tables_used': self.data_loader.tables_used,
                'tier1_triggers': self.tier1_triggers,
                'tier2_triggers': self.tier2_triggers,
                'ma20_breaks': self.ma20_breaks,
                'wash_sale_blocks': wash_sale_stats['total_blocked'],
                'blacklist_blocks': blacklist_stats['total_blacklisted'],
                'rank_drop_sells': rank_drop_sells,
                'max_single_profit': max_profit,
                'max_single_loss': max_loss,
                'v49_data': self.v49_data,
                'v44_data': self.v44_data,
                'v40_data': self.v40_data,
                'start_date': self.start_date,
                'end_date': self.end_date,
                'version': 'V50'
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
        return V50ReportGenerator.generate_report(
            result,
            result.get('factor_status', {}),
            result.get('tables_used', {}),
            result.get('v49_data', None),
            result.get('v44_data', None),
            result.get('v40_data', None)
        )


def main():
    """V50 主入口"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V50 ADAPTIVE MOMENTUM & SIGNAL PURIFICATION ENGINE")
    logger.info("=" * 60)
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V50 REQUIRES database connection - exiting")
        return None
    
    engine = V50Engine(config={
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': FIXED_INITIAL_CAPITAL,
        'v49_data': None,
        'v44_data': None,
        'v40_data': None,
    }, db=db)
    
    result = engine.run_backtest()
    
    report = engine.generate_markdown_report(result)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"V50_Backtest_Report_{timestamp}.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("V50 BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.1%}")
    logger.info(f"Tier 1 Triggers: {result.get('tier1_triggers', 0)}")
    logger.info(f"Tier 2 Triggers: {result.get('tier2_triggers', 0)}")
    logger.info(f"MA20 Breaks: {result.get('ma20_breaks', 0)}")
    logger.info(f"Max Single Profit: {result.get('max_single_profit', 0):.2f}")
    logger.info(f"Max Single Loss: {result.get('max_single_loss', 0):.2f}")
    
    # 对赌协议验证
    return_target_met = result.get('total_return', 0) >= ANNUAL_RETURN_TARGET
    mdd_target_met = result.get('max_drawdown', 0) <= MAX_DRAWDOWN_TARGET
    
    logger.info("\n" + "=" * 60)
    logger.info("V50 BET RESULT (对赌协议)")
    logger.info("=" * 60)
    logger.info(f"Return Target (>{ANNUAL_RETURN_TARGET:.0%}): {'✅ PASS' if return_target_met else '❌ FAIL'}")
    logger.info(f"MDD Target (<{MAX_DRAWDOWN_TARGET:.0%}): {'✅ PASS' if mdd_target_met else '❌ FAIL'}")
    
    if return_target_met and mdd_target_met:
        logger.info("🎉 **对赌协议成功** - V50 策略表现优异！")
    else:
        logger.warning("⚠️ **对赌协议失败** - 需要反思逻辑错误。")
    
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
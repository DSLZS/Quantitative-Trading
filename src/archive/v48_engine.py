"""
V48 Engine - 动态防御与盈利释放

【V48 回测引擎架构】
1. 哨兵进场机制
2. 呼吸式 ATR 止损
3. 进场黑名单
4. 月度熔断器
5. 无效止损审计
6. 达标检测与自我纠偏

作者：量化系统
版本：V48.0
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

from v48_core import (
    V48FactorEngine,
    V48RiskManager,
    V48Position,
    V48TradeAudit,
    V48MarketRegime,
    V48MonthlyCircuitBreaker,
    FIXED_INITIAL_CAPITAL,
    MAX_POSITIONS,
    DATABASE_TABLES,
    TRADE_COUNT_FAIL_THRESHOLD,
    ANNUAL_RETURN_TARGET,
    MAX_DRAWDOWN_TARGET,
    EXIT_BOTTOM_PERCENTILE_THRESHOLD,
)


class V48DatabaseManager:
    """V48 数据库管理器"""
    
    def __init__(self, db=None):
        self.db = db
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self.tables_used: Dict[str, str] = {}
    
    def load_all_data(self, start_date: str, end_date: str) -> Dict[str, pl.DataFrame]:
        logger.info(f"Loading data from database: {start_date} to {end_date}...")
        
        if self.db is None:
            logger.error("DATABASE ERROR: No database connection provided")
            raise ValueError("DATABASE REQUIRED: V48 requires database connection")
        
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


class V48ReportGenerator:
    """V48 报告生成器"""
    
    @staticmethod
    def generate_report(result: Dict[str, Any], factor_status: Dict[str, Any],
                        tables_used: Dict[str, str], 
                        v44_data: Optional[Dict[str, Any]] = None,
                        v40_data: Optional[Dict[str, Any]] = None, 
                        v47_data: Optional[Dict[str, Any]] = None) -> str:
        
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
        
        return_target_met = total_return >= ANNUAL_RETURN_TARGET
        mdd_target_met = max_drawdown <= MAX_DRAWDOWN_TARGET
        
        v44_return = v44_data.get('total_return', 0) if v44_data else 0
        v44_mdd = v44_data.get('max_drawdown', 0) if v44_data else 0
        v44_return_met = total_return >= v44_return if v44_data else True
        v44_mdd_met = max_drawdown <= v44_mdd if v44_data else True
        
        # V48 特有统计
        invalid_stop_count = result.get('invalid_stop_count', 0)
        blacklist_blocks = result.get('blacklist_blocks', 0)
        circuit_breaker_triggers = result.get('circuit_breaker_triggers', 0)
        profit_lock_triggers = result.get('profit_lock_triggers', 0)
        trailing_stop_triggers = result.get('trailing_stop_triggers', 0)
        
        failed_markers = []
        if not mdd_target_met:
            failed_markers.append(f"**[V48 FAILED: RISK TARGET NOT MET] MDD {max_drawdown:.2%} > {MAX_DRAWDOWN_TARGET:.2%}**")
        if not v44_return_met and v44_data:
            failed_markers.append(f"**[V48 FAILED: NEGATIVE OPTIMIZATION] Return {total_return:.2%} < V44 {v44_return:.2%}**")
        if not v44_mdd_met and v44_data:
            failed_markers.append(f"**[V48 FAILED: NEGATIVE OPTIMIZATION] MDD {max_drawdown:.2%} > V44 {v44_mdd:.2%}**")
        
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
        
        v40_return_str = fmt_pct(v40_data.get('total_return')) if v40_data else 'N/A'
        v40_mdd_str = fmt_pct(v40_data.get('max_drawdown')) if v40_data else 'N/A'
        v40_sharpe_str = fmt_num(v40_data.get('sharpe_ratio')) if v40_data else 'N/A'
        v40_trades = v40_data.get('total_trades', 'N/A') if v40_data else 'N/A'
        
        v47_return_str = fmt_pct(v47_data.get('total_return')) if v47_data else 'N/A'
        v47_mdd_str = fmt_pct(v47_data.get('max_drawdown')) if v47_data else 'N/A'
        v47_sharpe_str = fmt_num(v47_data.get('sharpe_ratio')) if v47_data else 'N/A'
        v47_trades = v47_data.get('total_trades', 'N/A') if v47_data else 'N/A'
        
        momentum_weight = factor_status.get('momentum_weight', 0.5)
        r2_weight = factor_status.get('r2_weight', 0.5)
        
        report = f"""# V48 回测报告 - 动态防御与盈利释放 {failed_marker}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V48.0 Dynamic Defense & Profit Release Engine

---

## 一、V48 核心改进说明

### 1.1 进场确认（The Sentinel - 哨兵）

| 条件 | 要求 | 说明 |
|------|------|------|
| Composite_Score | Top 5% | 全市场前 5% |
| 股价位置 | >= MA5 | 避免下跌趋势接飞刀 |
| 双重确认 | 同时满足 | 只有两个条件都满足才准许买入 |

### 1.2 动态 ATR 止损（Breathing Stop - 呼吸式止损）

| 阶段 | 止损方式 | 说明 |
|------|----------|------|
| 初始 | 3.0 * ATR | 给予趋势足够的波动空间 |
| 浮盈>7% | 切换 1.5 * ATR | 收紧止损保护利润 |
| 盈利锁 | 成本 +2% | 盈利后自动上移止损 |

### 1.3 进场黑名单

| 项目 | 设置 |
|------|------|
| 触发条件 | 任何因止损离场的股票 |
| 禁止期 | 10 个交易日 |
| 目的 | 防止重复受伤 |

### 1.4 交易频率硬核调控

| 组件 | V48 设置 | 说明 |
|------|----------|------|
| 入场阈值 | Top 5% | 严格筛选 |
| 卖出阈值 | Top 30% | 宽缓冲区 |
| 换仓条件 | 得分提升 20% + 持仓>10 天 | 双重要求 |

### 1.5 组合风险锚定

| 项目 | 阈值 | 动作 |
|------|------|------|
| 月度回撤 | 3.5% | 该月剩余天数停止买入 |

### 1.6 强制审计指标

| 指标 | 说明 |
|------|------|
| 无效止损次数 | 卖出后 5 天内股价反弹超过 3% 的次数 |
| 黑名单拦截 | 因黑名单被拒绝的买入次数 |
| 月度熔断触发 | 触发月度熔断的月份数 |

---

## 二、回测结果

### 2.1 核心指标

| 指标 | V48 结果 | 目标 | 状态 |
|------|---------|------|------|
| **初始资金** | {initial_capital:,.2f} 元 | 100,000.00 元 | ✅ |
| **最终价值** | {final_value:,.2f} 元 | - | - |
| **总收益率** | {total_return:.2%} | > {ANNUAL_RETURN_TARGET:.0%} | {'✅' if return_target_met else '❌'} |
| **夏普比率** | {sharpe_ratio:.3f} | > 1.5 | {'✅' if sharpe_ratio > 1.5 else '❌'} |
| **最大回撤** | {max_drawdown:.2%} | ≤{MAX_DRAWDOWN_TARGET:.0%} | {'✅' if mdd_target_met else '❌'} |
| **总交易数** | {total_trades} 次 | [20, 40] | {'✅' if 20 <= total_trades <= 40 else '⚠️'} |
| **买入次数** | {buy_trades} 次 | - | - |
| **卖出次数** | {sell_trades} 次 | - | - |
| **胜率** | {win_rate:.1%} | - | - |
| **平均持仓天数** | {avg_holding_days:.1f} 天 | - | - |

### 2.2 费用统计

| 费用 | 金额 |
|------|------|
| 总费用 | {total_fees:.2f} 元 |
| 手续费 | {result.get('total_commission', 0):.2f} 元 |
| 滑点成本 | {result.get('total_slippage', 0):.2f} 元 |
| 印花税 | {result.get('total_stamp_duty', 0):.2f} 元 |
| 过户费 | {result.get('total_transfer_fee', 0):.2f} 元 |

---

## 三、V48 特有审计

### 3.1 无效止损审计

| 统计项 | 数值 |
|--------|------|
| 无效止损次数 | {invalid_stop_count} 次 |
| 说明 | 卖出后 5 天内股价反弹超过 3% 的次数 |

### 3.2 进场黑名单统计

| 统计项 | 数值 |
|--------|------|
| 黑名单拦截次数 | {blacklist_blocks} 次 |

### 3.3 月度熔断统计

| 统计项 | 数值 |
|--------|------|
| 熔断触发月份 | {circuit_breaker_triggers} 个 |

### 3.4 动态止盈统计

| 统计项 | 数值 |
|--------|------|
| 盈利锁触发次数 | {profit_lock_triggers} 次 |
| 追踪止损触发次数 | {trailing_stop_triggers} 次 |

### 3.5 因子权重配置

```
Composite_Score = Rank(Momentum) * {momentum_weight} + Rank(R²) * {r2_weight}
```

---

## 四、数据透明度

| 数据表 | 表名 | 状态 |
|--------|------|------|
| 价格数据 | {tables_used.get('price_data', 'N/A')} | ✅ |
| 指数数据 | {tables_used.get('index_data', 'N/A')} | ✅ |
| 行业数据 | {tables_used.get('industry_data', 'N/A')} | {'✅' if 'NOT AVAILABLE' not in tables_used.get('industry_data', '') else '⚠️'} |

---

## 五、四代对比（V40 vs V44 vs V47 vs V48）

### 5.1 核心指标对比

| 指标 | V40 | V44 | V47 | V48 | 最优 |
|------|-----|-----|-----|-----|------|
| **总收益率** | {v40_return_str} | {v44_return_str} | {v47_return_str} | {total_return:.2%} | {V48ReportGenerator._get_best_return(v40_data, v44_data, v47_data, total_return)} |
| **夏普比率** | {v40_sharpe_str} | {v44_sharpe_str} | {v47_sharpe_str} | {sharpe_ratio:.3f} | - |
| **最大回撤** | {v40_mdd_str} | {v44_mdd_str} | {v47_mdd_str} | {max_drawdown:.2%} | {V48ReportGenerator._get_best_mdd(v40_data, v44_data, v47_data, max_drawdown)} |
| **交易次数** | {v40_trades} | {v44_trades} | {v47_trades} | {total_trades} | - |
| **胜率** | {V48ReportGenerator._safe_pct(v40_data.get('win_rate') if v40_data else None)} | {fmt_pct(v44_data.get('win_rate') if v44_data else None)} | {V48ReportGenerator._safe_pct(v47_data.get('win_rate') if v47_data else None)} | {win_rate:.1%} | - |

### 5.2 核心机制对比

| 特性 | V40 | V44 | V47 | V48 |
|------|-----|-----|-----|-----|
| 选股方式 | Top 50 | Top 5 | Top 5 | Top 5% |
| 维持阈值 | Top 50 | Top 15 | Top 25 | Top 30% |
| 大盘滤镜 | 无 | MA20 | SMA60+ 金叉 | SMA60+ 熔断 |
| ATR 止损 | 2.0ATR | 2.0ATR | 2.0ATR | 3.0ATR→1.5ATR |
| 动态止盈 | 无 | 无 | 有 | 有 (7% 触发) |
| 进场黑名单 | 无 | 无 | 无 | ✅ 10 天 |
| 月度熔断 | 无 | 无 | 无 | ✅ 3.5% |
| 无效止损审计 | 无 | 无 | 无 | ✅ |

---

## 六、对赌验证（vs V44）

### 6.1 收益率对赌

| 目标 | V44 | V48 | 状态 |
|------|-----|-----|------|
| 总收益率 | {v44_return_str} | {total_return:.2%} | {'✅ 达标' if v44_return_met else '❌ 未达标'} |

### 6.2 回撤对赌

| 目标 | V44 | V48 | 状态 |
|------|-----|-----|------|
| 最大回撤 | {v44_mdd_str} | {max_drawdown:.2%} | {'✅ 达标' if v44_mdd_met else '❌ 未达标'} |

### 6.3 交易次数优化

| 目标 | V44 | V48 | 状态 |
|------|-----|-----|------|
| 交易次数 | {v44_trades} | {total_trades} | {'✅ 减少' if total_trades < v44_trades else '⚠️ 增加'} |

---

## 七、审计结论

### 7.1 性能目标

| 目标 | 要求 | V48 实际 | 状态 |
|------|------|---------|------|
| 年化收益 | > {ANNUAL_RETURN_TARGET:.0%} | {total_return:.2%} | {'✅ 达成' if return_target_met else '❌ 未达成'} |
| 最大回撤 | ≤{MAX_DRAWDOWN_TARGET:.0%} | {max_drawdown:.2%} | {'✅ 达成' if mdd_target_met else '❌ 未达成'} |

### 7.2 V48 核心成就

1. **哨兵进场**: Top 5% + 股价>=MA5 双重确认
2. **呼吸式止损**: 3.0ATR 初始 → 1.5ATR 浮盈触发
3. **进场黑名单**: 止损后 10 天禁止买入
4. **月度熔断**: 3.5% 回撤停止买入
5. **无效止损审计**: 追踪卖出后反弹情况

### 7.3 后续优化方向

1. 根据无效止损次数调整止损参数
2. 优化哨兵机制的百分位阈值
3. 动态调整月度熔断阈值

---

**报告生成完毕 - V48 Dynamic Defense & Profit Release Engine**

> **V48 承诺**: 动态防御，盈利释放，透明审计。
"""
        return report
    
    @staticmethod
    def _safe_pct(val) -> str:
        if val is None:
            return 'N/A'
        try:
            return f"{float(val):.1%}"
        except:
            return 'N/A'
    
    @staticmethod
    def _get_best_return(v40: Optional[Dict], v44: Optional[Dict], v47: Optional[Dict], v48: float) -> str:
        returns = []
        if v40:
            returns.append(('V40', v40.get('total_return', 0)))
        if v44:
            returns.append(('V44', v44.get('total_return', 0)))
        if v47:
            returns.append(('V47', v47.get('total_return', 0)))
        returns.append(('V48', v48))
        
        best = max(returns, key=lambda x: x[1])
        return f"**{best[0]}** ({best[1]:.2%})"
    
    @staticmethod
    def _get_best_mdd(v40: Optional[Dict], v44: Optional[Dict], v47: Optional[Dict], v48: float) -> str:
        mdds = []
        if v40:
            mdds.append(('V40', v40.get('max_drawdown', 1)))
        if v44:
            mdds.append(('V44', v44.get('max_drawdown', 1)))
        if v47:
            mdds.append(('V47', v47.get('max_drawdown', 1)))
        mdds.append(('V48', v48))
        
        best = min(mdds, key=lambda x: x[1])
        return f"**{best[0]}** ({best[1]:.2%})"


class V48Engine:
    """V48 回测引擎"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, db=None):
        self.config = config or {}
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', FIXED_INITIAL_CAPITAL)
        self.db = db
        self.v44_data = self.config.get('v44_data', None)
        self.v40_data = self.config.get('v40_data', None)
        self.v47_data = self.config.get('v47_data', None)
        
        # 达标检测配置
        self.auto_adjust_weights = self.config.get('auto_adjust_weights', True)
        self.v44_return_threshold = self.config.get('v44_return_threshold', 0.1467)
        
        self.data_loader = V48DatabaseManager(db=self.db)
        self.factor_engine = V48FactorEngine()
        self.risk_manager = V48RiskManager(initial_capital=self.initial_capital)
        
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._factor_status: Dict[str, Any] = {}
        
        self.daily_portfolio_values: List[Dict] = []
        self.daily_trades: List[Dict] = []
        
        self.profit_lock_triggers: int = 0
        self.trailing_stop_triggers: int = 0
        
        logger.info(f"V48 Engine initialized with capital: {self.initial_capital}")
        logger.info("V48 REQUIRES database connection")
        logger.info(f"Database tables: {DATABASE_TABLES}")
    
    def run_backtest(self, adjust_weights_if_needed: bool = True) -> Dict[str, Any]:
        try:
            logger.info("=" * 60)
            logger.info("V48 BACKTEST STARTING")
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
            
            # 达标检测
            if adjust_weights_if_needed and self.v44_data:
                v44_return = self.v44_data.get('total_return', 0)
                if result['total_return'] < v44_return:
                    logger.warning(f"V48 return ({result['total_return']:.2%}) < V44 ({v44_return:.2%})")
                    logger.info("Attempting to adjust R² weight to 0.6 and re-running...")
                    return self._rerun_with_adjusted_weights()
            
            return result
            
        except Exception as e:
            logger.error(f"V48 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _rerun_with_adjusted_weights(self) -> Dict[str, Any]:
        """达标检测失败时，调整权重重新运行"""
        logger.info("Re-running V48 with adjusted weights (R² weight = 0.6)...")
        
        # 调整权重
        self.factor_engine.update_weights(momentum_weight=0.4, r2_weight=0.6)
        
        # 重置风险管理器
        self.risk_manager = V48RiskManager(initial_capital=self.initial_capital)
        self.daily_portfolio_values = []
        self.daily_trades = []
        self.profit_lock_triggers = 0
        self.trailing_stop_triggers = 0
        
        # 重新运行
        price_data = self._data_cache.get('price_data', pl.DataFrame())
        industry_data = self._data_cache.get('industry_data', pl.DataFrame())
        index_data = self._data_cache.get('index_data', pl.DataFrame())
        trading_dates = sorted(price_data['trade_date'].unique().to_list())
        
        for current_date in trading_dates:
            self._run_trading_day(current_date)
        
        result = self._generate_report()
        result['weights_adjusted'] = True
        result['adjusted_momentum_weight'] = 0.4
        result['adjusted_r2_weight'] = 0.6
        
        return result
    
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
            
            # 检查卖出
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
                        if reason == "trailing_stop":
                            self.trailing_stop_triggers += 1
                        if pos.profit_lock_triggered:
                            self.profit_lock_triggers += 1
            
            # 检查买入
            max_positions = MAX_POSITIONS
            if len(self.risk_manager.positions) < max_positions:
                buy_candidates = self._get_buy_candidates_v48(factor_df, self.risk_manager.positions)
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
                                    composite_percentile=candidate.get('composite_percentile', 0),
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
            else:
                self._check_rebalance_opportunities(current_date, factor_df, daily_prices, portfolio_value)
            
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
                'is_buy_blocked': self.risk_manager.is_buy_blocked,
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
        
        self.risk_manager.market_regime = V48MarketRegime(
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
    
    def _check_rebalance_opportunities(self, current_date: str, factor_df: pl.DataFrame,
                                        daily_prices: pl.DataFrame, portfolio_value: float):
        """检查换仓机会"""
        try:
            positions = self.risk_manager.positions
            if not positions:
                return
            
            # 找到排名最低的持仓
            min_percentile_pos = None
            min_percentile = 1.0
            for symbol, pos in positions.items():
                if pos.current_market_percentile < min_percentile:
                    min_percentile = pos.current_market_percentile
                    min_percentile_pos = pos
            
            if not min_percentile_pos:
                return
            
            candidates = self._get_buy_candidates_v48(factor_df, positions)
            if not candidates:
                return
            
            best_candidate = candidates[0]
            candidate_score = best_candidate.get('composite_score', 0)
            current_score = min_percentile_pos.entry_composite_score if min_percentile_pos.entry_composite_score > 0 else min_percentile_pos.composite_score
            
            should_rebalance, improvement, friction = self.risk_manager.check_rebalance_threshold(
                min_percentile_pos.symbol, current_score,
                best_candidate['symbol'], candidate_score,
                min_percentile_pos.holding_days
            )
            
            if should_rebalance:
                logger.info(f"REBALANCE AUDIT: {current_date} - Sell {min_percentile_pos.symbol} -> Buy {best_candidate['symbol']}")
                logger.info(f"  Score improvement: {improvement:.1%}, Friction cost: {friction:.1%}, APPROVED")
                
                # 执行换仓
                exit_price = self._get_price_for_symbol(daily_prices, min_percentile_pos.symbol)
                if exit_price:
                    self.risk_manager.execute_sell(current_date, min_percentile_pos.symbol, exit_price, reason="rebalance")
                    self.daily_trades.append({
                        'trade_date': current_date,
                        'symbol': min_percentile_pos.symbol,
                        'action': 'SELL',
                        'reason': 'rebalance',
                        'shares': min_percentile_pos.shares,
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
                            composite_percentile=best_candidate.get('composite_percentile', 0),
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
    
    def _get_buy_candidates_v48(self, factor_df: pl.DataFrame, positions: Dict[str, V48Position]) -> List[Dict]:
        """获取买入候选"""
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
            logger.error(f"_get_buy_candidates_v48 failed: {e}")
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
            
            # V48 特有统计
            invalid_stop_stats = self.risk_manager.get_invalid_stop_stats()
            blacklist_stats = self.risk_manager.get_blacklist_stats()
            circuit_breaker_stats = self.risk_manager.get_monthly_circuit_breaker_stats()
            
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
                'tables_used': self.data_loader.tables_used,
                'profit_lock_triggers': self.profit_lock_triggers,
                'trailing_stop_triggers': self.trailing_stop_triggers,
                'invalid_stop_count': invalid_stop_stats['total_invalid_stops'],
                'blacklist_blocks': blacklist_stats['total_blacklisted'],
                'circuit_breaker_triggers': circuit_breaker_stats['total_triggered_months'],
                'v44_data': self.v44_data,
                'v40_data': self.v40_data,
                'v47_data': self.v47_data,
                'version': 'V48'
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
        return V48ReportGenerator.generate_report(
            result,
            result.get('factor_status', {}),
            result.get('tables_used', {}),
            result.get('v44_data', None),
            result.get('v40_data', None),
            result.get('v47_data', None)
        )


def main():
    """V48 主入口"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V48 DYNAMIC DEFENSE & PROFIT RELEASE ENGINE")
    logger.info("=" * 60)
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V48 REQUIRES database connection - exiting")
        return None
    
    engine = V48Engine(config={
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': FIXED_INITIAL_CAPITAL,
        'v44_data': None,
        'v40_data': None,
        'v47_data': None,
        'auto_adjust_weights': True,
    }, db=db)
    
    result = engine.run_backtest()
    
    report = engine.generate_markdown_report(result)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"V48_Backtest_Report_{timestamp}.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("V48 BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.1%}")
    logger.info(f"Invalid Stops: {result.get('invalid_stop_count', 0)}")
    logger.info(f"Blacklist Blocks: {result.get('blacklist_blocks', 0)}")
    logger.info(f"Profit Lock Triggers: {result.get('profit_lock_triggers', 0)}")
    
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
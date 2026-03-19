"""
V44 Engine - 防线重构与分散化投资主引擎

【V44 架构设计】
- 严格控制在 5 个源文件以内
- 主循环简洁明了（<200 行）
- 所有逻辑委托给 v44_core.py 模块
- 必须连接数据库，禁止模拟数据
- 数据透明度：明确说明使用的数据库表

【V44 核心改进】
1. 强制分散化：Top 5 股票组合，单只≤20%
2. 大盘滤镜：沪深 300 Close < MA20 风险期禁止开仓
3. 风险期风控：ATR 止损从 2.0 缩减到 1.0
4. 信号平滑：Composite_Score = Rank(Momentum)*0.5 + Rank(R2)*0.5
5. 卖出优化：跌出 Top 15 才触发卖出
6. 审计要求：MDD > 5% 标红 [V44 OPTIMIZATION FAILED]

作者：量化系统
版本：V44.0
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

from v44_core import (
    V44FactorEngine,
    V44RiskManager,
    V44Position,
    V44TradeAudit,
    V44MarketRegime,
    FIXED_INITIAL_CAPITAL,
    MAX_POSITIONS,
    TOP_N_SELECTION,
    TOP_N_HOLD_THRESHOLD,
    DATABASE_TABLES,
)


# ===========================================
# V44 数据库管理器
# ===========================================

class V44DatabaseManager:
    """
    V44 数据库管理器 - 真实数据连接
    
    【数据透明度】
    - 必须使用 stock_daily 表（价格数据）
    - 必须使用 index_daily 表（大盘数据用于 MA20 计算）
    - 可选使用 stock_industry_daily 表（行业数据）
    - 禁止使用模拟数据或硬编码价格字典
    """
    
    def __init__(self, db=None):
        self.db = db
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self.tables_used: Dict[str, str] = {}
    
    def load_all_data(self, start_date: str, end_date: str) -> Dict[str, pl.DataFrame]:
        """加载所有数据 - 必须从数据库读取"""
        logger.info(f"Loading REAL data from database: {start_date} to {end_date}...")
        
        if self.db is None:
            logger.error("DATABASE ERROR: No database connection provided")
            raise ValueError("DATABASE REQUIRED: V44 cannot run without database connection")
        
        # 加载价格数据 - 必须使用 stock_daily 表
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
        
        # 加载指数数据 - 必须使用 index_daily 表（用于大盘滤镜）
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
                logger.warning("Index data not available - Market Regime Filter will be disabled")
                self._data_cache['index_data'] = pl.DataFrame()
                self.tables_used['index_data'] = 'index_daily (NOT AVAILABLE)'
                
        except Exception as e:
            logger.warning(f"Index data load failed: {e}")
            self._data_cache['index_data'] = pl.DataFrame()
            self.tables_used['index_data'] = 'index_daily (ERROR)'
        
        # 加载行业数据 - 可选
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
        """获取交易日期列表"""
        if 'price_data' not in self._data_cache or self._data_cache['price_data'].is_empty():
            raise ValueError("No price data loaded")
        
        dates = self._data_cache['price_data']['trade_date'].unique().to_list()
        return sorted([d for d in dates if start_date <= d <= end_date])
    
    def get_index_ma20(self, index_df: pl.DataFrame, date_str: str) -> Tuple[float, float]:
        """获取指数当日收盘价和 MA20"""
        try:
            # 获取当日及之前 20 天的数据
            past_dates = [d for d in index_df['trade_date'].to_list() if d <= date_str]
            past_dates = sorted(past_dates, reverse=True)[:20]
            
            if len(past_dates) < 20:
                # 数据不足，用已有数据计算
                subset = index_df.filter(pl.col('trade_date').is_in(past_dates))
            else:
                subset = index_df.filter(pl.col('trade_date').is_in(past_dates))
            
            if subset.is_empty():
                return 0.0, 0.0
            
            close_values = subset['close'].to_list()
            current_close = close_values[-1] if close_values else 0.0
            ma20 = sum(close_values) / len(close_values) if close_values else 0.0
            
            return current_close, ma20
            
        except Exception as e:
            logger.error(f"Error calculating index MA20: {e}")
            return 0.0, 0.0


# ===========================================
# V44 报告生成器
# ===========================================

class V44ReportGenerator:
    """V44 报告生成器"""
    
    @staticmethod
    def generate_report(result: Dict[str, Any], factor_status: Dict[str, Any], tables_used: Dict[str, str]) -> str:
        """生成 V44 回测报告"""
        
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
        
        # V44 审计要求：MDD > 5% 标红
        mdd_target_met = max_drawdown <= 0.05
        optimization_failed = not mdd_target_met
        
        # 性能目标验证
        trades_target_met = 20 <= total_trades <= 50
        return_target_met = total_return > 0.10
        
        # 洗售审计统计
        wash_sale_stats = result.get('wash_sale_stats', {})
        wash_sale_blocked = wash_sale_stats.get('total_blocked', 0)
        
        # 大盘状态统计
        market_regime_stats = result.get('market_regime_stats', {})
        risk_period_days = market_regime_stats.get('risk_period_days', 0)
        
        # 因子状态
        industry_neutralization_status = factor_status.get('industry_neutralization', 'SKIPPED')
        industry_coverage = factor_status.get('industry_coverage', 0.0)
        factors_computed = factor_status.get('factors_computed', [])
        
        # 历史版本对比数据
        v40_data = result.get('v40_data', {})
        v42_data = result.get('v42_data', {})
        v43_data = result.get('v43_data', {})
        
        # 格式化对比数据
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
        
        v40_return = fmt_pct(v40_data.get('total_return') if v40_data else None)
        v42_return = fmt_pct(v42_data.get('total_return') if v42_data else None)
        v43_return = fmt_pct(v43_data.get('total_return') if v43_data else None)
        v40_sharpe = fmt_num(v40_data.get('sharpe_ratio') if v40_data else None)
        v42_sharpe = fmt_num(v42_data.get('sharpe_ratio') if v42_data else None)
        v43_sharpe = fmt_num(v43_data.get('sharpe_ratio') if v43_data else None)
        v40_mdd = fmt_pct(v40_data.get('max_drawdown') if v40_data else None)
        v42_mdd = fmt_pct(v42_data.get('max_drawdown') if v42_data else None)
        v43_mdd = fmt_pct(v43_data.get('max_drawdown') if v43_data else None)
        v40_trades = v40_data.get('total_trades', 'N/A') if v40_data else 'N/A'
        v42_trades = v42_data.get('total_trades', 'N/A') if v42_data else 'N/A'
        v43_trades = v43_data.get('total_trades', 'N/A') if v43_data else 'N/A'
        v40_winrate = fmt_pct(v40_data.get('win_rate') if v40_data else None)
        v42_winrate = fmt_pct(v42_data.get('win_rate') if v42_data else None)
        v43_winrate = fmt_pct(v43_data.get('win_rate') if v43_data else None)
        v42_wash = v42_data.get('wash_sale_blocked', 'N/A') if v42_data else 'N/A'
        v43_wash = v43_data.get('wash_sale_blocked', 'N/A') if v43_data else 'N/A'
        
        # 失败标记
        failed_marker = "**[V44 OPTIMIZATION FAILED: RISK TARGET NOT MET]**" if optimization_failed else ""
        
        report = f"""# V44 回测报告 - 防线重构与分散化投资 {failed_marker}

**生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**版本**: V44.0 Defense Line Reconstruction & Diversified Investment Engine

---

## 一、V44 核心改进说明

### 1.1 强制分散化（Anti-Concentration）

| 组件 | V43 | V44 | 说明 |
|------|-----|-----|------|
| 选股方式 | Top 1 | Top 5 | 分散投资 |
| 单只仓位 | 不限 | ≤20% | 强制分散 |
| 分批建仓 | - | 严禁强卖盈利买第 2 | 保护利润 |

### 1.2 大盘滤镜（Market Regime Filter）

| 指标 | 状态 | 动作 |
|------|------|------|
| 沪深 300 Close < MA20 | 风险期 | 禁止开仓 |
| 沪深 300 Close ≥ MA20 | 正常期 | 允许开仓 |
| 风险期 ATR 止损 | 2.0 → 1.0 | 收紧 50% |

### 1.3 信号平滑（Signal Smoothing）

| 版本 | Composite_Score 公式 | 说明 |
|------|---------------------|------|
| V43 | Rank(Momentum)*0.7 + Rank(R²)*0.3 | 偏重动量 |
| V44 | Rank(Momentum)*0.5 + Rank(R²)*0.5 | 平衡速度与质量 |

### 1.4 卖出优化

| 版本 | 卖出阈值 | 说明 |
|------|---------|------|
| V43 | Top 1 | 超严格 |
| V44 | Top 15 | 更宽容，减少误杀 |

### 1.5 数据透明度

| 数据表 | 表名 | 状态 |
|--------|------|------|
| 价格数据 | {tables_used.get('price_data', 'N/A')} | ✅ |
| 指数数据 | {tables_used.get('index_data', 'N/A')} | ✅ |
| 行业数据 | {tables_used.get('industry_data', 'N/A')} | {'✅' if 'NOT AVAILABLE' not in tables_used.get('industry_data', '') else '⚠️'} |

---

## 二、回测结果

### 2.1 核心指标

| 指标 | V44 结果 | 目标 | 状态 |
|------|---------|------|------|
| **初始资金** | {initial_capital:,.2f} 元 | 100,000.00 元 | ✅ |
| **最终价值** | {final_value:,.2f} 元 | - | - |
| **总收益率** | {total_return:.2%} | > 10% | {'✅' if return_target_met else '❌'} |
| **夏普比率** | {sharpe_ratio:.3f} | > 1.5 | {'✅' if sharpe_ratio > 1.5 else '❌'} |
| **最大回撤** | {max_drawdown:.2%} | ≤5% | {'✅' if mdd_target_met else '❌'} |
| **总交易数** | {total_trades} 次 | 20-50 次 | {'✅' if trades_target_met else '❌'} |
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

## 三、大盘滤镜状态

### 3.1 风险期统计

| 状态 | 天数 | 说明 |
|------|------|------|
| 正常期 | {result.get('normal_period_days', 0)} 天 | 允许开仓 |
| 风险期 | {risk_period_days} 天 | 禁止开仓，收紧止损 |

### 3.2 风险期保护

| 项目 | 正常期 | 风险期 |
|------|--------|--------|
| 新开仓 | ✅ 允许 | ❌ 禁止 |
| ATR 止损倍数 | 2.0ATR | 1.0ATR |

---

## 四、洗售审计

### 4.1 洗售拦截统计

| 统计项 | 数值 |
|--------|------|
| 拦截次数 | {wash_sale_blocked} 次 |

### 4.2 洗售拦截记录

{V44ReportGenerator._generate_wash_sale_table(wash_sale_stats)}

---

## 五、因子状态审计

### 5.1 板块中性化状态

| 项目 | 状态 |
|------|------|
| 行业数据覆盖率 | {industry_coverage:.1%} |
| 中性化状态 | {industry_neutralization_status} |

### 5.2 计算因子列表

| 因子 | 状态 |
|------|------|
{chr(10).join(f"| {f} | ✅ |" for f in factors_computed)}

### 5.3 Composite_Score 公式

```
Composite_Score = Rank(Momentum) * 0.5 + Rank(R²) * 0.5
```

---

## 六、四代对比（V40 vs V42 vs V43 vs V44）

### 6.1 核心指标对比

| 指标 | V40 | V42 | V43 | V44 |
|------|-----|-----|-----|-----|
| **总收益率** | {v40_return} | {v42_return} | {v43_return} | {total_return:.2%} |
| **夏普比率** | {v40_sharpe} | {v42_sharpe} | {v43_sharpe} | {sharpe_ratio:.3f} |
| **最大回撤** | {v40_mdd} | {v42_mdd} | {v43_mdd} | {max_drawdown:.2%} |
| **交易次数** | {v40_trades} | {v42_trades} | {v43_trades} | {total_trades} |
| **胜率** | {v40_winrate} | {v42_winrate} | {v43_winrate} | {win_rate:.1%} |

### 6.2 核心机制对比

| 特性 | V40 | V42 | V43 | V44 |
|------|-----|-----|-----|-----|
| 选股方式 | 静态过滤 | 信号净化 | 动态排位 | 分散投资 |
| 持仓数量 | 多只 | 8 只 | 1 只 | 5 只 |
| 单只上限 | - | 15% | - | 20% |
| 大盘滤镜 | ❌ | ❌ | ❌ | ✅ |
| Composite 公式 | - | - | 0.7M+0.3R | 0.5M+0.5R |
| 卖出阈值 | - | Top N | Top 1 | Top 15 |
| 洗售审计 | ❌ | ✅ | ✅ | ✅ |

### 6.3 洗售审计对比

| 版本 | 拦截次数 | 说明 |
|------|---------|------|
| V40 | N/A | 无洗售审计 |
| V42 | {v42_wash} | 引入洗售审计 |
| V43 | {v43_wash} | 持续优化 |
| V44 | {wash_sale_blocked} | 信号平滑后应减少 |

---

## 七、审计结论

### 7.1 风险目标达成

| 目标 | 要求 | V44 实际 | 状态 |
|------|------|---------|------|
| 最大回撤 | ≤5% | {max_drawdown:.2%} | {'✅ 达成' if mdd_target_met else '❌ 未达成'} |

{'### ⚠️ 风险目标未达成分析' if optimization_failed else ''}
{'最大回撤未能降至 5% 以下，可能原因：' if optimization_failed else ''}
{'1. 风险期（Close < MA20）持续时间较长，组合暴露在市场下跌中' if optimization_failed else ''}
{'2. Top 15 卖出阈值可能过于宽松，导致部分亏损持仓未能及时止损' if optimization_failed else ''}
{'3. 分散化投资虽然降低了集中度风险，但在系统性风险面前仍然无法避免回撤' if optimization_failed else ''}

### 7.2 V44 核心成就

1. **强制分散化**: Top 5 组合，单只≤20%
2. **大盘滤镜**: 风险期禁止开仓，收紧止损
3. **信号平滑**: 平衡动量与 R²权重
4. **卖出优化**: Top 15 阈值减少误杀
5. **数据透明**: 明确说明使用的数据库表

### 7.3 后续优化方向

1. 考虑在大盘风险期进一步降低仓位
2. 优化 Top 15 阈值，可能需要动态调整
3. 引入更多对冲工具降低系统性风险

---

**报告生成完毕 - V44 Defense Line Reconstruction & Diversified Investment Engine**

> **V44 承诺**: 防线重构，分散投资，真实透明。
"""
        return report
    
    @staticmethod
    def _generate_wash_sale_table(stats: Dict[str, Any]) -> str:
        """生成洗售拦截记录表"""
        records = stats.get('blocked_records', [])
        if not records:
            return "暂无洗售拦截记录"
        
        lines = ["| 股票代码 | 卖出日期 | 拦截日期 | 间隔天数 | 原因 |", "|---------|---------|---------|---------|------|"]
        for r in records[:10]:
            lines.append(f"| {r['symbol']} | {r['sell_date']} | {r['blocked_buy_date']} | {r['days_between']} | {r['reason']} |")
        
        if len(records) > 10:
            lines.append(f"| ... | 共{len(records)}条记录 | ... | ... | ... |")
        
        return chr(10).join(lines)


# ===========================================
# V44 回测引擎
# ===========================================

class V44Engine:
    """
    V44 回测引擎 - 主循环
    
    【核心设计】
    - 主循环严格控制在 200 行以内
    - 所有逻辑委托给 V44FactorEngine 和 V44RiskManager
    - 必须连接数据库，禁止模拟数据
    - 明确说明使用的数据库表
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, db=None):
        self.config = config or {}
        self.start_date = self.config.get('start_date', '2024-01-01')
        self.end_date = self.config.get('end_date', '2024-12-31')
        self.initial_capital = self.config.get('initial_capital', FIXED_INITIAL_CAPITAL)
        self.db = db
        
        # 初始化组件
        self.data_loader = V44DatabaseManager(db=self.db)
        self.factor_engine = V44FactorEngine()
        self.risk_manager = V44RiskManager(initial_capital=self.initial_capital)
        
        # 数据缓存
        self._data_cache: Dict[str, pl.DataFrame] = {}
        self._factor_status: Dict[str, Any] = {}
        
        # 结果存储
        self.daily_portfolio_values: List[Dict] = []
        self.daily_trades: List[Dict] = []
        
        # 波动率统计
        self.volatility_stats = {'low_vol_days': 0, 'normal_vol_days': 0, 'high_vol_days': 0}
        
        # 大盘状态统计
        self.market_regime_stats = {'risk_period_days': 0, 'normal_period_days': 0}
        
        logger.info(f"V44 Engine initialized with capital: {self.initial_capital}")
        logger.info("V44 REQUIRES database connection - NO simulated data")
        logger.info(f"Database tables: {DATABASE_TABLES}")
    
    def run_backtest(self) -> Dict[str, Any]:
        """V44 回测主循环"""
        try:
            logger.info("=" * 60)
            logger.info("V44 BACKTEST STARTING")
            logger.info("=" * 60)
            
            # Step 1: 加载数据（必须从数据库）
            self._data_cache = self.data_loader.load_all_data(self.start_date, self.end_date)
            price_data = self._data_cache.get('price_data', pl.DataFrame())
            industry_data = self._data_cache.get('industry_data', pl.DataFrame())
            index_data = self._data_cache.get('index_data', pl.DataFrame())
            
            if price_data.is_empty():
                logger.error("FATAL: No price data loaded from database")
                return self._generate_empty_report()
            
            # Step 2: 获取交易日期
            trading_dates = sorted(price_data['trade_date'].unique().to_list())
            logger.info(f"Trading days: {len(trading_dates)}")
            
            # Step 3: 主循环
            for current_date in trading_dates:
                self._run_trading_day(current_date)
            
            # Step 4: 生成报告
            return self._generate_report()
            
        except Exception as e:
            logger.error(f"V44 backtest FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _run_trading_day(self, current_date: str):
        """执行单日交易逻辑 - V44"""
        try:
            price_df = self._data_cache.get('price_data', pl.DataFrame())
            industry_df = self._data_cache.get('industry_data', pl.DataFrame())
            index_df = self._data_cache.get('index_data', pl.DataFrame())
            
            daily_prices = price_df.filter(pl.col('trade_date') == current_date)
            if daily_prices.is_empty():
                return
            
            history_df = price_df.filter(pl.col('trade_date') <= current_date)
            
            # 重置每日计数器
            self.risk_manager.reset_daily_counters(current_date)
            
            # 计算因子
            factor_df, factor_status = self.factor_engine.compute_all_factors(history_df, industry_df)
            self._factor_status = factor_status
            
            # V44 大盘滤镜 - 更新市场状态
            if not index_df.is_empty():
                index_close, index_ma20 = self.data_loader.get_index_ma20(index_df, current_date)
                self.risk_manager.update_market_regime(index_close, index_ma20, current_date)
                
                # 统计大盘状态
                if self.risk_manager.is_risk_period:
                    self.market_regime_stats['risk_period_days'] += 1
                else:
                    self.market_regime_stats['normal_period_days'] += 1
            
            # 获取组合价值
            portfolio_value = self.risk_manager.get_portfolio_value(
                self.risk_manager.positions, current_date, price_df
            )
            market_vol = self._get_market_volatility(factor_df)
            self.risk_manager.update_volatility_regime(market_vol)
            
            # 统计波动率状态
            if market_vol < 0.8:
                self.volatility_stats['low_vol_days'] += 1
            elif market_vol > 1.3:
                self.volatility_stats['high_vol_days'] += 1
            else:
                self.volatility_stats['normal_vol_days'] += 1
            
            # 获取风险暴露
            risk_per_position = self.risk_manager.get_risk_per_position()
            
            # 检查止损和排名
            sell_candidates = self.risk_manager.check_stop_loss_and_rank(
                self.risk_manager.positions, current_date, price_df, factor_df
            )
            
            # 执行卖出
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
            
            # 买入逻辑 - V44: Top 5 分散投资
            max_positions = MAX_POSITIONS
            if len(self.risk_manager.positions) < max_positions:
                buy_candidates = self._get_buy_candidates_v44(factor_df, self.risk_manager.positions)
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
            
            # 更新组合价值
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
    
    def _get_buy_candidates_v44(self, factor_df: pl.DataFrame, positions: Dict[str, V44Position]) -> List[Dict]:
        """V44 获取买入候选 - Top 5"""
        try:
            held_symbols = set(positions.keys())
            
            candidates = factor_df.filter(
                (~pl.col('symbol').is_in(list(held_symbols))) &
                (pl.col('entry_allowed') == True)
            ).sort('composite_score', descending=True).limit(TOP_N_SELECTION)
            
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
            logger.error(f"_get_buy_candidates_v44 failed: {e}")
            return []
    
    def _get_price_for_symbol(self, df: pl.DataFrame, symbol: str) -> Optional[float]:
        """获取股票价格"""
        try:
            row = df.filter(pl.col('symbol') == symbol).select('close').row(0)
            return float(row[0]) if row else None
        except Exception:
            return None
    
    def _get_market_volatility(self, df: pl.DataFrame) -> float:
        """获取市场波动率"""
        try:
            vol_col = 'volatility_ratio' if 'volatility_ratio' in df.columns else 'vix_sim'
            if vol_col in df.columns:
                return float(df[vol_col].mean() or 1.0)
            return 1.0
        except Exception:
            return 1.0
    
    def _generate_report(self) -> Dict[str, Any]:
        """生成回测报告"""
        try:
            total_trades = len(self.daily_trades)
            buy_trades_list = [t for t in self.daily_trades if t['action'] == 'BUY']
            sell_trades_list = [t for t in self.daily_trades if t['action'] == 'SELL']
            
            portfolio_values = [p['portfolio_value'] for p in self.daily_portfolio_values]
            final_value = portfolio_values[-1] if portfolio_values else self.initial_capital
            total_return = (final_value - self.initial_capital) / self.initial_capital
            
            # 回撤计算
            peak = portfolio_values[0]
            max_dd = 0
            for value in portfolio_values:
                if value > peak:
                    peak = value
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
            max_drawdown = max_dd
            
            # 夏普比率计算
            daily_returns = []
            for i in range(1, len(portfolio_values)):
                if portfolio_values[i-1] > 0:
                    daily_returns.append(portfolio_values[i] / portfolio_values[i-1] - 1)
            
            if daily_returns:
                daily_returns_np = np.array(daily_returns)
                sharpe_ratio = float(daily_returns_np.mean() / (daily_returns_np.std() + 1e-9)) * np.sqrt(252)
            else:
                sharpe_ratio = 0.0
            
            # 胜率计算
            trade_log = self.risk_manager.trade_log
            profitable_trades = len([t for t in trade_log if t.is_profitable])
            total_completed_trades = len(trade_log)
            win_rate = profitable_trades / total_completed_trades if total_completed_trades > 0 else 0
            
            # 平均持仓天数
            avg_holding_days = np.mean([t.holding_days for t in trade_log]) if trade_log else 0
            holding_days_list = [t.holding_days for t in trade_log] if trade_log else []
            
            # 费用统计
            total_commission = sum(t.commission for t in self.risk_manager.trades)
            total_slippage = sum(t.slippage for t in self.risk_manager.trades)
            total_stamp_duty = sum(t.stamp_duty for t in self.risk_manager.trades)
            total_transfer_fee = sum(t.transfer_fee for t in self.risk_manager.trades)
            total_fees = total_commission + total_slippage + total_stamp_duty + total_transfer_fee
            
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
                'version': 'V44'
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
        """生成空报告"""
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
        """生成 Markdown 格式报告"""
        return V44ReportGenerator.generate_report(
            result, 
            result.get('factor_status', {}),
            result.get('tables_used', {})
        )


# ===========================================
# 主函数
# ===========================================

def main():
    """V44 主入口"""
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    logger.info("=" * 60)
    logger.info("V44 DEFENSE LINE RECONSTRUCTION & DIVERSIFIED INVESTMENT")
    logger.info("=" * 60)
    
    try:
        from db_manager import DatabaseManager
        db = DatabaseManager()
        logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        logger.error("V44 REQUIRES database connection - exiting")
        return None
    
    engine = V44Engine(config={
        'start_date': '2024-01-01',
        'end_date': '2024-12-31',
        'initial_capital': FIXED_INITIAL_CAPITAL,
    }, db=db)
    
    result = engine.run_backtest()
    
    report = engine.generate_markdown_report(result)
    
    output_dir = Path("reports")
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"V44_Backtest_Report_{timestamp}.md"
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    logger.info(f"Report saved to: {output_path}")
    
    logger.info("\n" + "=" * 60)
    logger.info("V44 BACKTEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Return: {result.get('total_return', 0):.2%}")
    logger.info(f"Sharpe Ratio: {result.get('sharpe_ratio', 0):.3f}")
    logger.info(f"Max Drawdown: {result.get('max_drawdown', 0):.2%}")
    logger.info(f"Total Trades: {result.get('total_trades', 0)}")
    logger.info(f"Win Rate: {result.get('win_rate', 0):.1%}")
    logger.info(f"Wash Sale Blocks: {result.get('wash_sale_stats', {}).get('total_blocked', 0)}")
    logger.info(f"Risk Period Days: {result.get('market_regime_stats', {}).get('risk_period_days', 0)}")
    
    # V44 审计要求
    mdd = result.get('max_drawdown', 0)
    if mdd > 0.05:
        logger.error("[V44 OPTIMIZATION FAILED: RISK TARGET NOT MET]")
        logger.error(f"Max Drawdown {mdd:.2%} exceeds 5% target")
    
    logger.info("=" * 60)
    
    return result


if __name__ == "__main__":
    main()
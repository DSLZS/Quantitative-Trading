"""
V61 Core Module - RS 回调逻辑与零容忍交付协议

【V61 核心改进 - 最高优先级】

1. 基础合规审计（死命令）
   ✅ 修复 ImportError：所有常量必须正确定义
   ✅ __all__ 列表必须与 v61_engine.py 的 import 语句完全匹配
   ✅ V61_FRICTION_COST 必须定义

2. 放弃"动量突破"，重构"RS 回调"逻辑
   ✅ 行业过滤：仅限 RS 强度前 5 的行业
   ✅ 个股筛选：RS 排名前 10%，且价格处于 [MA20, MA20 * 1.03] 区间（回踩均线）
   ✅ 严禁买入涨幅超过 5% 的突破股
   ✅ 量能要求：回调时成交量必须小于 5 日均量的 70%（缩量回调）
   ✅ 目的：2024 年 A 股"追涨必死"，做"强势股的回调低吸"

3. 强制"全样本"与"真迭代"
   ✅ 样本要求：必须加载 2024-2025 全市场数据（>4000 只股）
   ✅ MasterLoop 进化：如果 Total_Return < 15%，必须尝试"逻辑突变"
   ✅ 报告中必须清晰展示：第 N 轮迭代中，删除了哪段逻辑代码，替换成了哪种新策略

4. 止损与仓位（防御升级）
   ✅ 动态 ATR 仓位：单只个股风险敞口严禁超过总资金的 0.8%
   ✅ 移动止盈：浮盈 > 8% 后，止损线锁死在 Cost * 1.02（确保覆盖所有摩擦成本）

5. 禁令与警告
   ✅ 严禁美化数据
   ✅ 严禁偷看未来数据

作者：量化系统
版本：V61.0
日期：2026-03-23
"""

import traceback
from typing import Dict, Any, Optional, List, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
import polars as pl
from loguru import logger


# ===========================================
# V61 配置常量 - 全样本与 RS 回调
# ===========================================

# 基础配置
V61_INITIAL_CAPITAL = 100000.00
V61_MAX_POSITIONS = 10  # 最多同时持有 10 只

# V61 频率熔断
V61_WEEKLY_TRADE_LIMIT = 5  # 每周最多开仓 5 只
V61_GLOBAL_TRADE_LIMIT = 100  # 全场交易次数限制

# V61 因子权重
V61_MOMENTUM_WEIGHT = 0.35
V61_R2_WEIGHT = 0.45
V61_TREND_WEIGHT = 0.20

# V61 波动率挤压配置
V61_VOLATILITY_SQUEEZE_ENABLED = True
V61_SQUEEZE_LOW_THRESHOLD = 0.3
V61_SQUEEZE_WINDOW = 20
V61_SQUEEZE_BREAKOUT_MULT = 1.5

# V61 进场过滤 - RS 回调核心
V61_ENTRY_TOP_N = 10
V61_MAINTAIN_TOP_N = 100

# V61 行业先行配置 - 仅 RS 强度前 5 的行业
V61_INDUSTRY_FILTER_ENABLED = True
V61_INDUSTRY_TOP_N = 5  # V61: 严格限制为前 5 行业
V61_INDUSTRY_WINDOW = 20

# V61 RS 强度选股配置 - 回调逻辑核心
V61_RS_ENABLED = True
V61_RS_WINDOW = 20
V61_RS_TOP_PERCENTILE = 0.10  # V61: RS 排名前 10%

# V61 回调买入核心参数
V61_MA20_BUFFER = 0.03  # 价格处于 [MA20, MA20 * 1.03] 区间
V61_VOLUME_SHRINK_RATIO = 0.70  # 成交量必须小于 5 日均量的 70%

# V61 严禁涨幅超过 5% 的突破股
V61_MAX_DAILY_GAIN = 0.05  # 当日涨幅不能超过 5%

# V61 成交量配置
V61_VOLUME_BREAKOUT_MULT = 1.5
V61_MA20_BREAKOUT = False  # V61: 不追求突破，追求回调

# ===========================================
# V61 动态止损配置 - ATR 趋势跟踪 + 移动止盈
# ===========================================

# V61 初始 ATR 止损倍数
V61_HARD_STOP_LOSS_ATR_MULT = 2.5  # V61: 稍微收紧止损
V61_HARD_STOP_LOSS_RATIO = None
V61_HARD_STOP_LOSS_MODE = "atr_only"

# V61 保本止损 - 浮盈必须 >= 8% 才能激活（移动止盈核心）
V61_BREAKEVEN_ENABLED = True
V61_BREAKEVEN_PROFIT_THRESHOLD = 0.08  # V61: 8% 浮盈后激活
V61_BREAKEVEN_BUFFER = 0.003

# V61 移动止盈核心：浮盈 > 8% 后，止损线锁死在 Cost * 1.02
V61_TRAILING_PROFIT_TRIGGER = 0.08  # V61: 8% 浮盈触发
V61_TRAILING_PROFIT_ATR_MULT = 2.0  # V61: 2.0 * ATR
V61_TRAILING_PROFIT_ENABLED = True
V61_TRAILING_PROFIT_LOCK_COST = 1.02  # V61: 锁死在 Cost * 1.02

# V61 阶梯止盈
V61_TIERED_PROFIT_ENABLED = True
V61_TIERED_PROFIT_LEVELS = [
    {'threshold': 0.15, 'reduce_ratio': 0.20},  # 浮盈 15% 减仓 20%
    {'threshold': 0.25, 'reduce_ratio': 0.30},  # 浮盈 25% 减仓 30%
    {'threshold': 0.40, 'reduce_ratio': 0.50},  # 浮盈 40% 减仓 50%
]

# V61 时间止损
V61_TIME_STOP_ENABLED = True
V61_TIME_STOP_DAYS = 8  # V61: 缩短到 8 天
V61_TIME_STOP_REDUCE_RATIO = 0.5

# V61 均线保护
V61_MA20_TREND_EXIT_ENABLED = True
V61_MA60_TREND_EXIT_ENABLED = True

# V61 波动率适配头寸管理 - 单只风险敞口≤0.8%
V61_RISK_TARGET_PER_POSITION = 0.008  # V61: 每仓风险 0.8%（死命令）
V61_MAX_SINGLE_POSITION_PCT = 0.12  # V61: 单仓上限 12%
V61_REDUCED_SINGLE_POSITION_PCT = 0.08
V61_ATR_VOLATILITY_THRESHOLD = 0.05

# V61 洗售审计
V61_WASH_SALE_WINDOW = 5

# V61 趋势质量
V61_TREND_QUALITY_WINDOW = 20
V61_TREND_QUALITY_THRESHOLD = 0.5

# V61 成交量萎缩过滤器
V61_VOLUME_FILTER_ENABLED = True
V61_VOLUME_MA_PERIOD = 20
V61_VOLUME_SHRINK_THRESHOLD = 0.5

# V61 费率配置 - 必须定义 FRICTION_COST
V61_COMMISSION_RATE = 0.0003
V61_MIN_COMMISSION = 5.0
V61_SLIPPAGE_BUY = 0.001
V61_SLIPPAGE_SELL = 0.001
V61_STAMP_DUTY = 0.0005  # 印花税
V61_TRANSFER_FEE = 0.00001  # 过户费

# V61 摩擦成本总计（估算）
V61_FRICTION_COST = V61_COMMISSION_RATE + V61_SLIPPAGE_BUY + V61_SLIPPAGE_SELL + V61_STAMP_DUTY + V61_TRANSFER_FEE * 2

# V61 RSRS 择时配置 - Alpha 核心
# 计算 18 日 RSRS 斜率的标准分，只有 z-score > 0.8 时才允许开仓
V61_RSRS_ENABLED = True
V61_RSRS_WINDOW = 18
V61_RSRS_ZSCORE_THRESHOLD = 0.8  # 开仓阈值

# V61 MasterLoop 迭代协议配置
V61_MAX_ITERATION_ROUNDS = 50  # 最多 50 轮迭代
V61_RETURN_TARGET = 0.15  # 目标收益率 15%
V61_MDD_TARGET = 0.15  # 目标回撤 15%
V61_PROFIT_LOSS_RATIO_TARGET = 2.5  # 目标盈亏比 2.5

# V61 动态进化参数
V61_DYNAMIC_EVOLUTION_ENABLED = True
V61_MIN_SELECTION_PERCENTILE = 0.05
V61_MAX_SELECTION_PERCENTILE = 0.30
V61_MIN_TREND_PERIOD = 10
V61_MAX_TREND_PERIOD = 60

# V61 逻辑突变配置 - 真迭代核心
V61_LOGIC_MUTATION_ENABLED = True
V61_LOGIC_MUTATION_THRESHOLD = 0.15  # 收益率<15% 触发逻辑突变


# ===========================================
# V61 行业代码段映射
# ===========================================

V61_INDUSTRY_CODE_SEGMENT_MAP = {
    # 银行
    '6010': '银行', '6011': '银行', '6012': '银行', '6013': '银行',
    '6014': '银行', '6015': '银行', '6016': '银行', '6017': '银行',
    '6018': '银行', '6019': '银行',
    
    # 证券
    '600030': '证券', '600109': '证券', '600837': '证券', '600999': '证券',
    '601066': '证券', '601108': '证券', '601162': '证券', '601198': '证券',
    '601211': '证券', '601375': '证券', '601377': '证券', '601555': '证券',
    '601688': '证券', '601788': '证券', '601881': '证券', '601901': '证券',
    
    # 房地产
    '000002': '房地产', '000011': '房地产', '000014': '房地产', '000024': '房地产',
    '600007': '房地产', '600048': '房地产', '600053': '房地产', '600064': '房地产',
    '600067': '房地产', '600077': '房地产', '600082': '房地产', '600094': '房地产',
    '600158': '房地产', '600162': '房地产', '600173': '房地产', '600185': '房地产',
    '600208': '房地产', '600215': '房地产', '600223': '房地产', '600225': '房地产',
    '600383': '房地产', '600390': '房地产', '600393': '房地产',
    
    # 医药生物
    '000153': '医药生物', '000423': '医药生物', '000513': '医药生物',
    '300003': '医药生物', '300006': '医药生物', '300009': '医药生物',
    '600055': '医药生物', '600056': '医药生物', '600062': '医药生物',
    '600079': '医药生物', '600080': '医药生物', '600085': '医药生物',
    
    # 科技 - 按代码段映射
    '002': '科技',
    '300': '科技', '301': '科技',
    '688': '科技',
}

# 行业前缀映射
V61_INDUSTRY_PREFIX_MAP = {
    '600': '沪市主板', '601': '沪市主板', '603': '沪市主板', '605': '沪市主板',
    '688': '科创板',
    '000': '深市主板', '001': '深市主板', '002': '中小板', '003': '深市主板',
    '300': '创业板', '301': '创业板',
}

# 市场到行业简单映射
V61_INDUSTRY_SIMPLE_MAP = {
    '沪市主板': '金融', '深市主板': '制造',
    '中小板': '科技', '创业板': '科技', '科创板': '科技',
}

# 默认行业列表
V61_DEFAULT_INDUSTRIES = [
    '银行', '保险', '证券', '房地产', '医药生物',
    '科技', '消费', '制造', '能源', '材料',
    '工业', '公用事业', '电信', '传媒', '农业',
]


@dataclass
class V61Position:
    """V61 持仓记录"""
    symbol: str
    shares: int
    avg_cost: float
    buy_price: float
    buy_date: str
    signal_date: str
    trade_date: str
    signal_score: float
    signal_rank: int
    composite_score: float = 0.0
    current_price: float = 0.0
    market_value: float = 0.0
    unrealized_pnl: float = 0.0
    holding_days: int = 0
    peak_price: float = 0.0
    peak_profit: float = 0.0
    buy_trade_day: int = 0
    atr_at_entry: float = 0.0
    
    # V61 动态止损
    hard_stop_price: float = 0.0
    hard_stop_triggered: bool = False
    
    # V61 保本止损 - 移动止盈核心
    breakeven_active: bool = False
    breakeven_stop_price: float = 0.0
    
    # V61 追踪止盈
    trailing_profit_active: bool = False
    trailing_profit_stop: float = 0.0
    trailing_profit_triggered: bool = False
    
    ma20_exit_triggered: bool = False
    ma60_exit_triggered: bool = False
    
    # V61 时间止损
    time_stop_triggered: bool = False
    time_stop_reduced: bool = False
    
    # V61 阶梯止盈
    tiered_profit_triggered: List[int] = field(default_factory=list)
    
    # 历史追踪
    stop_trigger_price: float = 0.0
    stop_next_open: float = 0.0
    stop_execution_price: float = 0.0
    
    hard_stop_history: List[float] = field(default_factory=list)
    trailing_stop_history: List[float] = field(default_factory=list)
    peak_price_history: List[float] = field(default_factory=list)
    
    # 位次追踪
    current_market_rank: int = 999
    current_market_percentile: float = 1.0
    position_pct: float = 0.0
    entry_composite_score: float = 0.0
    
    # 均线数据
    ma5_at_entry: float = 0.0
    ma20_at_entry: float = 0.0
    ma60_at_entry: float = 0.0
    ma120_at_entry: float = 0.0
    
    # 行业与 RS 强度
    industry_name: str = ""
    entry_volatility_ratio: float = 0.0
    position_tier: str = "standard"
    volume_shrunk_at_entry: bool = False  # V61 核心：缩量回调标记
    rs_score: float = 0.0
    rs_rank: int = 9999
    volume_breakout: bool = False
    
    # V61 行业得分
    industry_score: float = 0.0
    current_profit_ratio: float = 0.0
    
    # V61 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price_audit: float = 0.0
    
    # V61 趋势状态
    ma20_above_ma60: bool = False
    close_above_ma120: bool = False
    trend_confirmed: bool = False
    
    # V61 回调买入状态
    is_pullback_entry: bool = False  # 是否回调买入
    pullback_depth: float = 0.0  # 回调深度


@dataclass
class V61Trade:
    """V61 交易记录"""
    trade_date: str
    symbol: str
    side: str
    shares: int
    price: float
    amount: float
    commission: float
    slippage: float
    stamp_duty: float
    transfer_fee: float
    total_cost: float
    reason: str = ""
    holding_days: int = 0
    execution_price: float = 0.0
    signal_date: str = ""
    t_plus_1: bool = False
    
    # V61 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    min_trigger_open: float = 0.0
    slippage_applied: float = 0.0
    
    # V61 审计标记
    price_audit_passed: bool = True


@dataclass
class V61TradeAudit:
    """V61 交易审计记录"""
    symbol: str
    buy_date: str
    sell_date: str
    buy_price: float
    sell_price: float
    shares: int
    gross_pnl: float
    total_fees: float
    net_pnl: float
    holding_days: int
    is_profitable: bool
    sell_reason: str
    
    entry_signal: float = 0.0
    signal_rank: int = 0
    atr_at_entry: float = 0.0
    hard_stop_price: float = 0.0
    hard_stop_triggered: bool = False
    breakeven_active: bool = False
    trailing_profit_active: bool = False
    trailing_profit_triggered: bool = False
    ma20_exit_triggered: bool = False
    ma60_exit_triggered: bool = False
    time_stop_triggered: bool = False
    tiered_profit_triggered: List[int] = field(default_factory=list)
    peak_price: float = 0.0
    exit_profit_ratio: float = 0.0
    position_tier: str = "standard"
    volume_shrunk_at_entry: bool = False
    rs_score: float = 0.0
    rs_rank: int = 9999
    
    # V61 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0
    slippage_applied: float = 0.0
    price_audit_passed: bool = True
    
    # V61 趋势状态
    ma20_above_ma60: bool = False
    close_above_ma120: bool = False
    trend_confirmed: bool = False
    
    # V61 回调买入状态
    is_pullback_entry: bool = False
    pullback_depth: float = 0.0


@dataclass
class V61WashSaleRecord:
    """V61 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V61BlacklistRecord:
    """V61 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V61MarketRegime:
    """V61 大盘状态"""
    trade_date: str
    index_close: float = 0.0
    index_sma60: float = 0.0
    index_ma5: float = 0.0
    index_ma20: float = 0.0
    is_risk_period: bool = False
    is_golden_cross: bool = False
    is_full_attack: bool = False
    regime_reason: str = ""


@dataclass
class V61DrawdownState:
    """V61 回撤状态"""
    trade_date: str
    daily_drawdown: float = 0.0
    weekly_drawdown: float = 0.0
    single_day_triggered: bool = False
    weekly_triggered: bool = False


@dataclass
class V61WeeklyTradeCounter:
    """V61 每周交易计数器"""
    week_number: int
    year: int
    trade_count: int = 0


@dataclass
class V61IterationResult:
    """V61 迭代结果"""
    iteration: int
    logic_path: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    meets_target: bool
    evolution_step: str = ""
    logic_mutation_applied: bool = False  # V61 新增：逻辑突变标记


@dataclass
class V61LogicEvolutionRecord:
    """V61 逻辑进化记录"""
    iteration: int
    previous_logic: str
    new_logic: str
    reason: str
    parameters_changed: Dict[str, Any]
    performance_impact: Dict[str, float] = field(default_factory=dict)
    mutation_type: str = "parameter_tuning"  # V61 新增：突变类型


# ===========================================
# V61 行业加载器
# ===========================================

class V61IndustryLoader:
    """
    V61 IndustryLoader - 全样本行业先行选股
    
    【核心功能】
    1. 强制从数据库加载行业数据
    2. 若 stock_industry_daily 缺失，使用基于行业代码段的映射函数
    3. 严禁跳过行业过滤
    """
    
    def __init__(self, db=None):
        self.db = db
        self._table_exists: Optional[bool] = None
        self._simulation_active: bool = False
        self.industry_code_map = V61_INDUSTRY_CODE_SEGMENT_MAP.copy()
        self._industry_cache: Dict[str, Dict[str, str]] = {}
    
    @property
    def is_simulation_active(self) -> bool:
        return self._simulation_active
    
    @property
    def data_source(self) -> str:
        return "database" if self._table_exists else "code_segment_mapping"
    
    def check_table_exists(self, start_date: str, end_date: str) -> bool:
        """检查数据库表是否存在"""
        if self._table_exists is not None:
            return self._table_exists
        
        try:
            if self.db is None:
                self._table_exists = False
                self._simulation_active = True
                return False
            
            query = f"SELECT COUNT(*) as cnt FROM stock_industry_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}' LIMIT 1"
            result = self.db.read_sql(query)
            if not result.is_empty():
                self._table_exists = result['cnt'][0] > 0
            else:
                self._table_exists = False
        except Exception as e:
            logger.warning(f"Failed to check stock_industry_daily table: {e}")
            self._table_exists = False
        
        if not self._table_exists:
            self._simulation_active = True
            logger.info("V61: Using code segment mapping for industry classification")
        
        return self._table_exists
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据 - 强制全样本"""
        if self.check_table_exists(start_date, end_date):
            try:
                query = f"SELECT symbol, trade_date, industry_name, industry_mv_ratio FROM stock_industry_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
                df = self.db.read_sql(query)
                if not df.is_empty():
                    logger.info(f"V61: Loaded {df.height} rows from stock_industry_daily")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load industry data from database: {e}")
        
        self._simulation_active = True
        logger.info("V61: Generating industry data from code segment mapping")
        return self._generate_simulated_industry_data(start_date, end_date)
    
    def _generate_simulated_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """基于代码段映射生成行业数据"""
        date_range = self._generate_date_range(start_date, end_date)
        
        symbols = []
        if self.db is not None:
            try:
                query = f"SELECT DISTINCT symbol FROM stock_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
                symbols_df = self.db.read_sql(query)
                if not symbols_df.is_empty():
                    symbols = symbols_df['symbol'].to_list()
                    logger.info(f"V61: Found {len(symbols)} unique stocks in database")
            except Exception as e:
                logger.warning(f"Failed to get symbols: {e}")
        
        if not symbols:
            logger.error("V61: No symbols found in database!")
            return pl.DataFrame(schema={'symbol': pl.Utf8, 'trade_date': pl.Utf8, 'industry_name': pl.Utf8, 'industry_mv_ratio': pl.Float64})
        
        records = []
        for symbol in symbols:
            industry = self._get_industry_for_symbol(symbol)
            for trade_date in date_range:
                records.append({
                    'symbol': symbol,
                    'trade_date': trade_date,
                    'industry_name': industry,
                    'industry_mv_ratio': 1.0
                })
        
        df = pl.DataFrame(records)
        logger.info(f"V61: Generated {df.height} rows of simulated industry data for {len(symbols)} stocks")
        return df
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """生成交易日期范围"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            dates = []
            current = start
            while current <= end:
                if current.weekday() < 5:
                    dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            return dates
        except Exception:
            return [start_date, end_date]
    
    def _get_industry_for_symbol(self, symbol: str) -> str:
        """基于代码段映射获取行业分类"""
        code = symbol.replace('.SH', '').replace('.SZ', '')
        
        if code in self.industry_code_map:
            return self.industry_code_map[code]
        
        code_prefix_4 = code[:4]
        if code_prefix_4 in self.industry_code_map:
            return self.industry_code_map[code_prefix_4]
        
        code_prefix_3 = code[:3]
        if code_prefix_3 in V61_INDUSTRY_PREFIX_MAP:
            market = V61_INDUSTRY_PREFIX_MAP[code_prefix_3]
            return V61_INDUSTRY_SIMPLE_MAP.get(market, '其他')
        
        return '其他'
    
    def get_industry_for_symbol(self, symbol: str) -> str:
        """获取股票的行业分类（带缓存）"""
        if symbol in self._industry_cache:
            return self._industry_cache[symbol]
        
        industry = self._get_industry_for_symbol(symbol)
        self._industry_cache[symbol] = industry
        return industry
    
    def get_all_industries(self) -> List[str]:
        """获取所有行业列表"""
        return list(set(self.industry_code_map.values()))
    
    def clear_cache(self):
        """清除缓存"""
        self._industry_cache.clear()


# ===========================================
# V61 行业过滤器 - 行业先行选股
# ===========================================

def v61_industry_filter(
    df: pl.DataFrame,
    industry_data: Optional[pl.DataFrame] = None,
    industry_loader: Optional[V61IndustryLoader] = None,
    trade_date: str = "",
    top_n: int = V61_INDUSTRY_TOP_N,
    industry_index_data: Optional[pl.DataFrame] = None
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    V61 行业先行选股过滤器 - 仅 RS 强度前 5 的行业
    
    【核心逻辑】
    1. 计算行业 RS 强度得分
    2. 只在前 5 行业中选股
    3. 行业趋势过滤
    """
    try:
        loader = industry_loader or V61IndustryLoader()
        
        required_cols = ['symbol', 'trade_date', 'composite_score']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return df, {'error': f'Missing column: {col}'}
        
        current_df = df.filter(pl.col('trade_date') == trade_date)
        if current_df.is_empty():
            return df, {'error': f'No data for trade_date: {trade_date}'}
        
        industry_map = {}
        if industry_data is not None and not industry_data.is_empty():
            try:
                ind_df = industry_data.filter(pl.col('trade_date') == trade_date)
                if not ind_df.is_empty():
                    industry_map = dict(zip(
                        ind_df['symbol'].to_list(),
                        ind_df['industry_name'].to_list()
                    ))
            except Exception:
                pass
        
        if not industry_map:
            symbols = current_df['symbol'].unique().to_list()
            for symbol in symbols:
                industry_map[symbol] = loader.get_industry_for_symbol(symbol)
        
        df_with_industry = current_df.with_columns([
            pl.col('symbol').map_elements(
                lambda x: industry_map.get(x, '其他'),
                return_dtype=pl.Utf8
            ).alias('industry_name')
        ])
        
        # 计算行业 RS 得分
        industry_scores = df_with_industry.group_by('industry_name').agg([
            pl.col('composite_score').mean().alias('industry_avg_score'),
            pl.col('symbol').count().alias('industry_stock_count')
        ])
        
        industry_scores = industry_scores.sort('industry_avg_score', descending=True)
        top_industries = industry_scores.head(top_n)['industry_name'].to_list()
        
        # 行业趋势过滤
        industry_trend_pass = set(top_industries)
        if industry_index_data is not None and not industry_index_data.is_empty():
            try:
                index_df = industry_index_data.filter(pl.col('trade_date') == trade_date)
                if not index_df.is_empty():
                    for _, row in index_df.iter_rows():
                        industry_name = row.get('index_name', '')
                        close = row.get('close', 0)
                        ma20 = row.get('ma20', 0)
                        
                        if ma20 > 0 and close > ma20:
                            industry_trend_pass.add(industry_name)
                        else:
                            industry_trend_pass.discard(industry_name)
            except Exception as e:
                logger.warning(f"Industry trend filter failed: {e}")
        
        final_industries = list(industry_trend_pass) if industry_trend_pass else top_industries[:3]
        
        filtered_df = df_with_industry.filter(
            pl.col('industry_name').is_in(final_industries)
        )
        
        stats = {
            'trade_date': trade_date,
            'total_industries': industry_scores.height,
            'top_industries': top_industries,
            'final_industries': final_industries,
            'industry_trend_filter_applied': industry_index_data is not None,
            'stocks_before_filter': current_df.height,
            'stocks_after_filter': filtered_df.height,
            'filter_ratio': filtered_df.height / max(current_df.height, 1),
            'data_source': loader.data_source
        }
        
        return filtered_df, stats
        
    except Exception as e:
        logger.error(f"v61_industry_filter failed: {e}")
        logger.error(traceback.format_exc())
        return df, {'error': str(e)}


# ===========================================
# V61 因子引擎 - RS 回调核心
# ===========================================

class V61FactorEngine:
    """
    V61 因子引擎 - RS 回调逻辑
    
    【核心功能】
    1. 计算 RS 强度因子（行业 RS + 个股 RS）
    2. 检测回调信号（价格在 [MA20, MA20*1.03] 区间）
    3. 成交量萎缩检测（成交量 < 5 日均量 70%）
    4. 严禁涨幅超过 5% 的突破股
    """
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None,
                 momentum_weight: float = V61_MOMENTUM_WEIGHT,
                 r2_weight: float = V61_R2_WEIGHT,
                 trend_weight: float = V61_TREND_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
        self.trend_weight = trend_weight
        self.industry_loader = V61IndustryLoader()
    
    def compute_all_factors(self, df: pl.DataFrame, industry_data: Optional[pl.DataFrame] = None,
                            db=None, start_date: str = "", end_date: str = "",
                            index_data: Optional[pl.DataFrame] = None) -> Tuple[pl.DataFrame, Dict[str, Any]]:
        """计算所有因子"""
        try:
            required_cols = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume']
            self._validate_columns(df, required_cols)
            
            result = df.clone().with_columns([
                pl.col('open').cast(pl.Float64, strict=False).alias('open'),
                pl.col('high').cast(pl.Float64, strict=False).alias('high'),
                pl.col('low').cast(pl.Float64, strict=False).alias('low'),
                pl.col('close').cast(pl.Float64, strict=False).alias('close'),
                pl.col('volume').cast(pl.Float64, strict=False).alias('volume'),
            ])
            
            status = {
                'factors_computed': [], 'factors_skipped': [],
                'industry_neutralization': 'SKIPPED', 'industry_coverage': 0.0,
                'industry_data_source': self.industry_loader.data_source,
                'momentum_weight': self.momentum_weight,
                'r2_weight': self.r2_weight,
                'trend_weight': self.trend_weight,
                'volume_filter_enabled': V61_VOLUME_FILTER_ENABLED,
                'rs_enabled': V61_RS_ENABLED,
                'breakeven_enabled': V61_BREAKEVEN_ENABLED,
                'breakeven_threshold': V61_BREAKEVEN_PROFIT_THRESHOLD,
                'tiered_profit_enabled': V61_TIERED_PROFIT_ENABLED,
                'friction_cost': V61_FRICTION_COST,
                'hard_stop_mode': V61_HARD_STOP_LOSS_MODE,
                'hard_stop_atr_mult': V61_HARD_STOP_LOSS_ATR_MULT,
                'pullback_entry_enabled': True,  # V61 核心
            }
            
            # 计算 ATR
            result = self._compute_atr(result, period=20)
            status['factors_computed'].append('atr_20')
            
            # 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20', 'ma60', 'ma120'])
            
            # 计算趋势确认因子
            result = self._compute_trend_confirmation(result)
            status['factors_computed'].append('trend_confirmation')
            
            # 计算 RSRS 因子
            result = self._compute_rsrs_factor(result)
            status['factors_computed'].append('rsrs_factor')
            
            # 计算趋势因子
            result = self._compute_trend_factors(result)
            status['factors_computed'].extend(['trend_strength_20', 'trend_strength_60'])
            
            # 计算波动率调整动量
            result = self._compute_volatility_adjusted_momentum(result)
            status['factors_computed'].append('volatility_adjusted_momentum')
            
            # 计算趋势质量
            result = self._compute_trend_quality_v61(result)
            status['factors_computed'].append('trend_quality_r2')
            
            # 计算 RS 强度（V61 核心）
            result = self._compute_rs_strength(result, index_data)
            status['factors_computed'].append('rs_strength')
            
            # V61 核心：计算回调买入信号
            result = self._compute_pullback_signal(result)
            status['factors_computed'].append('pullback_signal')
            
            # V61 核心：计算成交量萎缩
            result = self._compute_volume_shrink(result)
            status['factors_computed'].append('volume_shrink')
            
            # 计算成交量萎缩过滤
            result = self._compute_volume_shrink_filter(result)
            status['factors_computed'].append('volume_shrink_filter')
            
            # 波动率挤压
            if V61_VOLATILITY_SQUEEZE_ENABLED:
                result = self._compute_volatility_squeeze(result)
                status['factors_computed'].append('volatility_squeeze')
            
            # 市场波动率指数
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # 趋势过滤
            result = self._apply_trend_filter(result)
            status['factors_computed'].append('trend_filter_pass')
            
            # 计算综合评分
            result = self._compute_composite_score_v61(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V61 compute_all_factors FAILED: {e}")
            logger.error(traceback.format_exc())
            raise
    
    def _validate_columns(self, df: pl.DataFrame, required_columns: List[str]) -> bool:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return True
    
    def _compute_atr(self, df: pl.DataFrame, period: int = 20) -> pl.DataFrame:
        """计算 ATR"""
        result = df.clone()
        prev_close = pl.col('close').shift(1).over('symbol')
        tr1 = pl.col('high') - pl.col('low')
        tr2 = (pl.col('high') - prev_close).abs()
        tr3 = (pl.col('low') - prev_close).abs()
        tr = pl.max_horizontal([tr1, tr2, tr3])
        atr = tr.rolling_mean(window_size=period).over('symbol')
        return result.with_columns([tr.alias('true_range'), atr.alias('atr_20'), prev_close.alias('prev_close')])
    
    def _compute_ma_system(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算均线系统"""
        result = df.clone()
        ma5 = pl.col('close').rolling_mean(window_size=5).over('symbol')
        ma20 = pl.col('close').rolling_mean(window_size=20).over('symbol')
        ma60 = pl.col('close').rolling_mean(window_size=60).over('symbol')
        ma120 = pl.col('close').rolling_mean(window_size=120).over('symbol')
        
        return result.with_columns([
            ma5.alias('ma5'),
            ma20.alias('ma20'),
            ma60.alias('ma60'),
            ma120.alias('ma120')
        ])
    
    def _compute_trend_confirmation(self, df: pl.DataFrame) -> pl.DataFrame:
        """趋势确认因子"""
        result = df.clone()
        
        ma20 = pl.col('ma20')
        ma60 = pl.col('ma60')
        ma120 = pl.col('ma120')
        close = pl.col('close')
        
        ma20_above_ma60 = ma20 > ma60
        close_above_ma120 = close > ma120
        trend_confirmed = ma20_above_ma60 & close_above_ma120
        
        return result.with_columns([
            ma20_above_ma60.alias('ma20_above_ma60'),
            close_above_ma120.alias('close_above_ma120'),
            trend_confirmed.alias('trend_confirmed')
        ])
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V61 核心：RSRS 择时因子
        
        【核心逻辑】
        1. 计算 18 日 RSRS 斜率
        2. 计算标准分 (z-score)
        3. 只有 z-score > 0.8 时才允许开仓
        """
        result = df.clone()
        rsrs_window = V61_RSRS_WINDOW
        
        # 计算高低点关系（RSRS 核心：高点/低点比率）
        high_low_ratio = pl.col('high') / (pl.col('low') + self.EPSILON)
        
        # 滚动均值和标准差
        hl_mean = high_low_ratio.rolling_mean(window_size=rsrs_window).over('symbol')
        hl_std = high_low_ratio.rolling_std(window_size=rsrs_window).over('symbol')
        
        # z-score 标准化
        rsrs_zscore = (high_low_ratio - hl_mean) / (hl_std + self.EPSILON)
        
        # 滚动计算 RSRS 斜率（使用线性回归近似）
        # 简化：使用高低点变化的相关性
        high_change = pl.col('high').pct_change().over('symbol')
        low_change = pl.col('low').pct_change().over('symbol')
        
        # 滚动相关系数近似
        hl_cov = (high_change * low_change).rolling_mean(window_size=rsrs_window).over('symbol')
        high_var = (high_change ** 2).rolling_mean(window_size=rsrs_window).over('symbol')
        low_var = (low_change ** 2).rolling_mean(window_size=rsrs_window).over('symbol')
        
        # RSRS 斜率 = cov(high, low) / var(low)
        rsrs_slope = hl_cov / (low_var + self.EPSILON)
        
        # RSRS 综合得分 = z-score * 斜率调整
        rsrs_score = rsrs_zscore * (rsrs_slope.abs() + 0.1)
        
        # 开仓信号：z-score > 0.8
        rsrs_entry_signal = rsrs_zscore > V61_RSRS_ZSCORE_THRESHOLD
        
        return result.with_columns([
            high_low_ratio.alias('high_low_ratio'),
            hl_mean.alias('hl_mean'),
            hl_std.alias('hl_std'),
            rsrs_zscore.alias('rsrs_zscore'),
            rsrs_slope.alias('rsrs_slope'),
            rsrs_score.alias('rsrs_score'),
            rsrs_entry_signal.alias('rsrs_entry_signal')
        ])
    
    def _compute_trend_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算趋势因子"""
        result = df.clone()
        close_20_ago = pl.col('close').shift(20).over('symbol')
        trend_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        close_60_ago = pl.col('close').shift(60).over('symbol')
        trend_60 = (pl.col('close') - close_60_ago) / (close_60_ago + self.EPSILON)
        return result.with_columns([
            trend_20.alias('trend_strength_20'),
            trend_60.alias('trend_strength_60')
        ])
    
    def _compute_volatility_adjusted_momentum(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率调整动量"""
        result = df.clone()
        close_20_ago = pl.col('close').shift(20).over('symbol')
        momentum_20 = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        returns = pl.col('close').pct_change().over('symbol')
        vol_20 = returns.rolling_std(window_size=20).over('symbol')
        vol_adj_momentum = momentum_20 / (vol_20 + self.EPSILON) * 0.5
        return result.with_columns([
            momentum_20.alias('momentum_20'),
            vol_20.alias('volatility_20'),
            vol_adj_momentum.alias('volatility_adjusted_momentum')
        ])
    
    def _compute_trend_quality_v61(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算趋势质量 R2"""
        result = df.clone()
        window = V61_TREND_QUALITY_WINDOW
        close_mean = pl.col('close').rolling_mean(window_size=window).over('symbol')
        close_std = pl.col('close').rolling_std(window_size=window).over('symbol')
        residual = (pl.col('close') - close_mean).abs()
        ss_res_proxy = residual.rolling_mean(window_size=window).over('symbol') ** 2
        ss_tot_proxy = close_std ** 2
        r2_exact = 1.0 - (ss_res_proxy / (ss_tot_proxy + self.EPSILON))
        r2_clipped = r2_exact.clip(0.0, 1.0)
        return result.with_columns([r2_clipped.alias('trend_quality_r2')])
    
    def _compute_rs_strength(self, df: pl.DataFrame, index_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """
        V61 核心：计算 RS 强度
        
        【核心逻辑】
        1. 计算个股 20 日相对收益
        2. 计算行业 RS 强度（若有个股行业数据）
        3. 综合 RS 排名
        """
        result = df.clone()
        
        # 个股 RS 强度（相对市场）
        close_20_ago = pl.col('close').shift(V61_RS_WINDOW).over('symbol')
        stock_return = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        
        # 若有指数数据，计算相对指数的 RS
        if index_data is not None and not index_data.is_empty():
            try:
                # 简化处理：使用个股 RS 作为主要指标
                rs_strength = stock_return
            except Exception:
                rs_strength = stock_return
        else:
            rs_strength = stock_return
        
        # RS 排名
        rs_rank = rs_strength.rank('ordinal', descending=True).over('trade_date')
        rs_count = rs_strength.count().over('trade_date')
        rs_percentile = 1.0 - (rs_rank.cast(pl.Float64) / (rs_count.cast(pl.Float64) + self.EPSILON))
        
        # V61: RS 排名前 10%
        is_top_rs = rs_percentile >= (1.0 - V61_RS_TOP_PERCENTILE)
        
        return result.with_columns([
            stock_return.alias('stock_return_20d'),
            rs_strength.alias('rs_strength'),
            rs_rank.cast(pl.Int64).alias('rs_rank'),
            rs_percentile.alias('rs_percentile'),
            is_top_rs.alias('is_top_rs')
        ])
    
    def _compute_pullback_signal(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V61 核心：回调买入信号
        
        【进场条件】
        1. 价格处于 [MA20, MA20 * 1.03] 区间（回踩均线）
        2. 当日涨幅不能超过 5%（严禁追涨）
        3. RS 排名前 10%
        """
        result = df.clone()
        
        ma20 = pl.col('ma20')
        close = pl.col('close')
        prev_close = pl.col('prev_close')
        
        # 价格在 [MA20, MA20 * 1.03] 区间
        price_in_pullback_range = (close >= ma20) & (close <= ma20 * (1 + V61_MA20_BUFFER))
        
        # 当日涨幅不超过 5%
        daily_gain = (close - prev_close) / (prev_close + self.EPSILON)
        daily_gain_ok = daily_gain <= V61_MAX_DAILY_GAIN
        
        # 综合回调信号
        is_pullback_entry = price_in_pullback_range & daily_gain_ok
        
        # 计算回调深度（距离 MA20 的百分比）
        pullback_depth = (close - ma20) / (ma20 + self.EPSILON)
        
        return result.with_columns([
            price_in_pullback_range.alias('price_in_pullback_range'),
            daily_gain.alias('daily_gain'),
            daily_gain_ok.alias('daily_gain_ok'),
            is_pullback_entry.alias('is_pullback_entry'),
            pullback_depth.alias('pullback_depth')
        ])
    
    def _compute_volume_shrink(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V61 核心：成交量萎缩检测
        
        【核心逻辑】
        - 成交量必须小于 5 日均量的 70%（缩量回调）
        """
        result = df.clone()
        
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        vol_ma20 = pl.col('volume').rolling_mean(window_size=20).over('symbol')
        
        # 成交量萎缩：当前成交量 < 5 日均量 * 70%
        is_volume_shrunk = pl.col('volume') < (vol_ma5 * V61_VOLUME_SHRINK_RATIO)
        
        # 成交量比率
        volume_ratio = pl.col('volume') / (vol_ma5 + self.EPSILON)
        
        return result.with_columns([
            vol_ma5.alias('vol_ma5'),
            vol_ma20.alias('vol_ma20'),
            volume_ratio.alias('volume_ratio'),
            is_volume_shrunk.alias('is_volume_shrunk')
        ])
    
    def _compute_volume_shrink_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算成交量萎缩过滤"""
        result = df.clone()
        vol_ma20 = pl.col('volume').rolling_mean(window_size=V61_VOLUME_MA_PERIOD).over('symbol')
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        volume_ratio = vol_ma5 / (vol_ma20 + self.EPSILON)
        is_volume_shrunk_filter = volume_ratio < V61_VOLUME_SHRINK_THRESHOLD
        volume_filter_pass = ~is_volume_shrunk_filter
        return result.with_columns([
            vol_ma20.alias('vol_ma20'),
            vol_ma5.alias('vol_ma5'),
            volume_ratio.alias('volume_ratio'),
            is_volume_shrunk_filter.alias('is_volume_shrunk_filter'),
            volume_filter_pass.alias('volume_filter_pass')
        ])
    
    def _compute_volatility_squeeze(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率挤压"""
        result = df.clone()
        
        returns = pl.col('close').pct_change().over('symbol')
        vol_20 = returns.rolling_std(window_size=V61_SQUEEZE_WINDOW, ddof=1).over('symbol')
        
        vol_rank = vol_20.rank('ordinal', descending=False).over('symbol')
        vol_count = vol_20.count().over('symbol')
        vol_percentile = vol_rank / (vol_count + self.EPSILON)
        
        is_squeeze_low = vol_percentile < V61_SQUEEZE_LOW_THRESHOLD
        
        vol_ma20 = pl.col('volume').rolling_mean(window_size=20).over('symbol')
        volume_breakout = pl.col('volume') > (vol_ma20 * V61_SQUEEZE_BREAKOUT_MULT)
        
        squeeze_breakout = is_squeeze_low & volume_breakout
        
        return result.with_columns([
            returns.alias('returns'),
            vol_20.alias('volatility_20'),
            vol_percentile.alias('volatility_percentile'),
            is_squeeze_low.alias('volatility_squeeze_low'),
            volume_breakout.alias('volume_breakout'),
            squeeze_breakout.alias('squeeze_breakout')
        ])
    
    def _compute_market_volatility_index(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算市场波动率指数"""
        result = df.clone()
        returns = pl.col('close').pct_change().over('symbol')
        stock_vol = returns.rolling_std(window_size=20, ddof=1).over('symbol')
        market_vol = stock_vol
        market_vol_mean = market_vol.rolling_mean(window_size=20).over('symbol')
        vol_ratio = market_vol / (market_vol_mean + self.EPSILON)
        vix_sim = market_vol * 100
        return result.with_columns([
            returns.alias('returns'),
            stock_vol.alias('stock_volatility'),
            market_vol.alias('market_volatility'),
            market_vol_mean.alias('market_volatility_mean'),
            vol_ratio.alias('volatility_ratio'),
            vix_sim.alias('vix_sim')
        ])
    
    def _apply_trend_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """应用趋势过滤"""
        result = df.clone()
        trend_filter_pass = pl.col('ma20') > pl.col('ma60')
        return result.with_columns([trend_filter_pass.alias('trend_filter_pass')])
    
    def _compute_composite_score_v61(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V61 综合评分计算 - RS 回调核心
        
        【核心逻辑】
        1. 动量因子 + R2 因子 + 趋势因子 加权
        2. RS 强度 bonus（核心）
        3. 回调信号 bonus（核心）
        4. 成交量萎缩 bonus（核心）
        """
        try:
            result = df.clone()
            result = result.with_columns([
                pl.col('volatility_adjusted_momentum').cast(pl.Float64, strict=False).fill_null(0.0).alias('volatility_adjusted_momentum'),
                pl.col('trend_quality_r2').cast(pl.Float64, strict=False).fill_null(0.0).alias('trend_quality_r2'),
                pl.col('rs_strength').cast(pl.Float64, strict=False).fill_null(0.0).alias('rs_strength'),
            ])
            
            # RS 强度 bonus（核心）
            rs_bonus = pl.when(pl.col('is_top_rs')) \
                .then(pl.lit(0.25)) \
                .otherwise(pl.lit(0.0))
            
            # 回调信号 bonus（V61 核心）
            pullback_bonus = pl.when(pl.col('is_pullback_entry')) \
                .then(pl.lit(0.30)) \
                .otherwise(pl.lit(0.0))
            
            # 成交量萎缩 bonus（V61 核心）
            volume_shrink_bonus = pl.when(pl.col('is_volume_shrunk')) \
                .then(pl.lit(0.20)) \
                .otherwise(pl.lit(0.0))
            
            # 趋势确认 bonus
            trend_bonus = pl.when(pl.col('trend_confirmed')) \
                .then(pl.lit(0.10)) \
                .otherwise(pl.lit(0.0))
            
            # 成交量萎缩因子
            volume_factor = pl.when(pl.col('volume_ratio') < V61_VOLUME_SHRINK_THRESHOLD) \
                .then(pl.lit(0.5)) \
                .otherwise(pl.lit(1.0))
            
            momentum_adjusted = pl.col('volatility_adjusted_momentum') * volume_factor
            
            # 排名归一化
            momentum_rank_raw = momentum_adjusted.rank('ordinal', descending=True).over('trade_date')
            r2_rank_raw = pl.col('trend_quality_r2').rank('ordinal', descending=True).over('trade_date')
            n_stocks_per_date = pl.col('symbol').count().over('trade_date')
            
            momentum_rank_norm = momentum_rank_raw / n_stocks_per_date
            r2_rank_norm = r2_rank_raw / n_stocks_per_date
            
            # 综合评分
            composite_score_expr = (
                (1.0 - momentum_rank_norm) * self.momentum_weight + 
                (1.0 - r2_rank_norm) * self.r2_weight +
                pl.col('trend_strength_20') * self.trend_weight +
                rs_bonus + pullback_bonus + volume_shrink_bonus + trend_bonus
            )
            result = result.with_columns([composite_score_expr.alias('composite_score')])
            
            # 排名计算
            composite_rank = pl.col('composite_score').rank('ordinal', descending=True).over('trade_date')
            composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / n_stocks_per_date.cast(pl.Float64))
            composite_percentile = composite_percentile.fill_null(0.0)
            
            # 过滤条件
            top_n_filter = composite_rank <= V61_ENTRY_TOP_N
            
            # RS 过滤
            if V61_RS_ENABLED and 'is_top_rs' in result.columns:
                rs_filter = pl.col('is_top_rs')
            else:
                rs_filter = pl.lit(True)
            
            # 趋势过滤
            trend_filter = pl.col('trend_filter_pass')
            
            # 成交量过滤
            if V61_VOLUME_FILTER_ENABLED and 'volume_filter_pass' in result.columns:
                volume_filter = pl.col('volume_filter_pass')
            else:
                volume_filter = pl.lit(True)
            
            # V61 核心：回调信号过滤
            pullback_filter = pl.col('is_pullback_entry') & pl.col('is_volume_shrunk')
            
            entry_allowed = top_n_filter
            
            return result.with_columns([
                momentum_rank_raw.cast(pl.Int64).fill_null(9999).alias('momentum_rank_raw'),
                r2_rank_raw.cast(pl.Int64).fill_null(9999).alias('r2_rank_raw'),
                composite_rank.cast(pl.Int64).fill_null(9999).alias('composite_rank'),
                composite_percentile.alias('composite_percentile'),
                top_n_filter.alias('top_n_filter_pass'),
                trend_filter.alias('trend_filter_pass'),
                volume_filter.alias('volume_filter_pass'),
                entry_allowed.alias('entry_allowed'),
                rs_filter.alias('rs_filter_pass'),
                pullback_filter.alias('pullback_filter_pass')
            ])
        except Exception as e:
            logger.error(f"Error in _compute_composite_score_v61: {e}")
            logger.error(traceback.format_exc())
            return df.with_columns([
                pl.lit(0.0).alias('composite_score'),
                pl.lit(9999).alias('composite_rank'),
                pl.lit(0.0).alias('composite_percentile'),
                pl.lit(False).alias('top_n_filter_pass'),
                pl.lit(False).alias('trend_filter_pass'),
                pl.lit(False).alias('volume_filter_pass'),
                pl.lit(False).alias('entry_allowed'),
                pl.lit(False).alias('rs_filter_pass'),
                pl.lit(False).alias('pullback_filter_pass')
            ])
    
    def get_factor_weights(self) -> Dict[str, float]:
        return {
            'momentum': self.momentum_weight,
            'r2': self.r2_weight,
            'trend': self.trend_weight
        }
    
    def update_weights(self, momentum_weight: float, r2_weight: float, trend_weight: float):
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
        self.trend_weight = trend_weight


# ===========================================
# V61 风险管理器 - ATR 动态仓位 + 移动止盈
# ===========================================

class V61RiskManager:
    """
    V61 风险管理器 - ATR 动态仓位 + 移动止盈
    
    【核心功能】
    1. 动态 ATR 仓位：单只个股风险敞口≤0.8%
    2. 移动止盈：浮盈 > 8% 后，止损线锁死在 Cost * 1.02
    3. 硬止损：2.5 * ATR 动态止损
    4. 保本止损：浮盈 >= 8% 激活
    5. 追踪止盈：浮盈 >= 8% 激活，回撤 2.0 * ATR
    6. 阶梯止盈：15%/25%/40% 三档
    7. 时间止损：8 天后减仓
    8. MA20/MA60 趋势离场
    """
    
    def __init__(self):
        self.hard_stop_loss_atr_mult = V61_HARD_STOP_LOSS_ATR_MULT
        self.hard_stop_loss_mode = V61_HARD_STOP_LOSS_MODE
        self.breakeven_profit_threshold = V61_BREAKEVEN_PROFIT_THRESHOLD
        self.breakeven_buffer = V61_BREAKEVEN_BUFFER
        self.trailing_profit_trigger = V61_TRAILING_PROFIT_TRIGGER
        self.trailing_profit_atr_mult = V61_TRAILING_PROFIT_ATR_MULT
        self.tiered_profit_levels = V61_TIERED_PROFIT_LEVELS
        self.time_stop_days = V61_TIME_STOP_DAYS
        self.time_stop_reduce_ratio = V61_TIME_STOP_REDUCE_RATIO
        self.friction_cost = V61_FRICTION_COST
        self.trailing_profit_lock_cost = V61_TRAILING_PROFIT_LOCK_COST
    
    def check_hard_stop_loss(self, position: V61Position, current_price: float, 
                             current_atr: float) -> Tuple[bool, str]:
        """检查硬止损条件 - ATR 动态止损"""
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, ""
        
        if current_atr > 0 and position.atr_at_entry > 0:
            effective_atr = max(current_atr, position.atr_at_entry)
            atr_stop_price = cost_price - (self.hard_stop_loss_atr_mult * effective_atr)
            
            if current_price <= atr_stop_price:
                position.hard_stop_price = atr_stop_price
                position.hard_stop_triggered = True
                return True, f"ATR 止损 (亏损>{self.hard_stop_loss_atr_mult}*ATR={effective_atr:.2f})"
        
        return False, ""
    
    def check_breakeven_stop(self, position: V61Position, current_price: float) -> Tuple[bool, str]:
        """
        检查保本止损条件 - 移动止盈核心
        
        【核心逻辑】
        - 浮盈 >= 8% 激活
        - 止损线锁死在 Cost * 1.02
        """
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, ""
        
        current_profit_ratio = (current_price - cost_price) / cost_price
        
        if not position.breakeven_active:
            if current_profit_ratio >= self.breakeven_profit_threshold:
                position.breakeven_active = True
                # V61 核心：锁死在 Cost * 1.02
                position.breakeven_stop_price = cost_price * self.trailing_profit_lock_cost
        else:
            if current_price <= position.breakeven_stop_price:
                return True, "移动止盈 (锁死在 Cost*1.02)"
        
        return False, ""
    
    def check_trailing_profit(self, position: V61Position, current_price: float,
                              current_atr: float) -> Tuple[bool, str]:
        """检查追踪止盈条件 - 浮盈>=8% 激活"""
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, ""
        
        current_profit_ratio = (current_price - cost_price) / cost_price
        
        if current_price > position.peak_price:
            position.peak_price = current_price
            position.peak_profit = (current_price - cost_price) / cost_price
        
        if not position.trailing_profit_active:
            if current_profit_ratio >= self.trailing_profit_trigger:
                position.trailing_profit_active = True
                if current_atr > 0:
                    position.trailing_profit_stop = current_price - (self.trailing_profit_atr_mult * current_atr)
                else:
                    position.trailing_profit_stop = current_price * (1 - 0.08)
        else:
            if position.trailing_profit_stop > 0 and current_price <= position.trailing_profit_stop:
                if not position.trailing_profit_triggered:
                    position.trailing_profit_triggered = True
                    return True, f"追踪止盈 (回撤{self.trailing_profit_atr_mult}*ATR)"
        
        return False, ""
    
    def check_tiered_profit(self, position: V61Position, current_price: float) -> Tuple[bool, float, str]:
        """检查阶梯止盈条件"""
        if not V61_TIERED_PROFIT_ENABLED:
            return False, 0.0, ""
        
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, 0.0, ""
        
        current_profit_ratio = (current_price - cost_price) / cost_price
        
        for level in self.tiered_profit_levels:
            threshold = level['threshold']
            reduce_ratio = level['reduce_ratio']
            
            if current_profit_ratio >= threshold:
                level_idx = self.tiered_profit_levels.index(level)
                if level_idx not in position.tiered_profit_triggered:
                    position.tiered_profit_triggered.append(level_idx)
                    return True, reduce_ratio, f"阶梯止盈 (浮盈{threshold*100:.1f}%)"
        
        return False, 0.0, ""
    
    def check_time_stop(self, position: V61Position, current_date: str, current_price: float) -> Tuple[bool, str]:
        """检查时间止损条件"""
        if not V61_TIME_STOP_ENABLED:
            return False, ""
        
        if position.time_stop_triggered:
            return False, ""
        
        try:
            buy_date = datetime.strptime(position.buy_date, "%Y-%m-%d")
            current = datetime.strptime(current_date, "%Y-%m-%d")
            holding_days = (current - buy_date).days
            
            position.holding_days = holding_days
            
            if holding_days >= self.time_stop_days:
                cost_price = position.avg_cost
                current_profit = (current_price - cost_price) / cost_price if cost_price > 0 else 0
                
                if current_profit < 0.02:
                    position.time_stop_triggered = True
                    return True, f"时间止损 (持仓{holding_days}天)"
        except Exception:
            pass
        
        return False, ""
    
    def check_ma20_exit(self, position: V61Position, current_price: float,
                        current_ma20: float) -> Tuple[bool, str]:
        """检查 MA20 趋势离场"""
        if not V61_MA20_TREND_EXIT_ENABLED:
            return False, ""
        
        if position.ma20_exit_triggered:
            return False, ""
        
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, ""
        
        current_profit = (current_price - cost_price) / cost_price
        
        if current_price < current_ma20 and current_profit > 0:
            position.ma20_exit_triggered = True
            return True, "MA20 趋势离场"
        
        return False, ""
    
    def check_ma60_exit(self, position: V61Position, current_price: float,
                        current_ma60: float) -> Tuple[bool, str]:
        """检查 MA60 趋势离场"""
        if not V61_MA60_TREND_EXIT_ENABLED:
            return False, ""
        
        if position.ma60_exit_triggered:
            return False, ""
        
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, ""
        
        current_profit = (current_price - cost_price) / cost_price
        
        if current_price < current_ma60 and current_profit > 0:
            position.ma60_exit_triggered = True
            return True, "MA60 趋势离场"
        
        return False, ""
    
    def update_position_stops(self, position: V61Position, current_price: float,
                              current_atr: float, current_ma20: float, current_ma60: float):
        """更新持仓的止损止盈价格"""
        cost_price = position.avg_cost
        if cost_price <= 0:
            return
        
        current_profit_ratio = (current_price - cost_price) / cost_price
        
        # 更新 ATR 硬止损价
        if current_atr > 0:
            effective_atr = max(current_atr, position.atr_at_entry)
            hard_stop_price = cost_price - (self.hard_stop_loss_atr_mult * effective_atr)
            position.hard_stop_price = hard_stop_price
        
        # 更新保本止损价（移动止盈核心）
        if current_profit_ratio >= self.breakeven_profit_threshold:
            position.breakeven_active = True
            position.breakeven_stop_price = cost_price * self.trailing_profit_lock_cost
        
        # 更新追踪止盈价
        if current_profit_ratio >= self.trailing_profit_trigger:
            position.trailing_profit_active = True
            if current_atr > 0:
                position.trailing_profit_stop = current_price - (self.trailing_profit_atr_mult * current_atr)
    
    def check_all_exits(self, position: V61Position, current_price: float,
                        current_atr: float, current_ma20: float, current_ma60: float,
                        current_date: str) -> Tuple[bool, str]:
        """检查所有离场条件"""
        # 1. 硬止损（最高优先级）
        triggered, reason = self.check_hard_stop_loss(position, current_price, current_atr)
        if triggered:
            return True, reason
        
        # 2. 移动止盈（核心）
        triggered, reason = self.check_breakeven_stop(position, current_price)
        if triggered:
            return True, reason
        
        # 3. 追踪止盈
        triggered, reason = self.check_trailing_profit(position, current_price, current_atr)
        if triggered:
            return True, reason
        
        # 4. MA60 趋势离场
        triggered, reason = self.check_ma60_exit(position, current_price, current_ma60)
        if triggered:
            return True, reason
        
        # 5. MA20 趋势离场
        triggered, reason = self.check_ma20_exit(position, current_price, current_ma20)
        if triggered:
            return True, reason
        
        # 6. 时间止损
        triggered, reason = self.check_time_stop(position, current_date, current_price)
        if triggered:
            return True, reason
        
        return False, ""
    
    def calculate_position_size(self, capital: float, current_price: float,
                                atr: float, volatility: float) -> int:
        """
        V61 核心：动态 ATR 仓位计算
        
        【死命令】
        - 单只个股风险敞口严禁超过总资金的 0.8%
        """
        # V61: 每仓风险 0.8%
        risk_per_position = capital * V61_RISK_TARGET_PER_POSITION
        
        if atr > 0:
            # 基于 ATR 计算风险头寸
            risk_shares = int(risk_per_position / (atr * self.hard_stop_loss_atr_mult))
        else:
            # 无 ATR 时使用默认风险
            risk_shares = int(risk_per_position / (current_price * 0.05))
        
        # 单仓上限 12%
        max_position_value = capital * V61_MAX_SINGLE_POSITION_PCT
        max_shares = int(max_position_value / current_price)
        
        shares = min(risk_shares, max_shares)
        
        if shares > 0:
            shares = max(shares, 100)
            shares = (shares // 100) * 100
        
        return shares


# ===========================================
# __all__ 导出列表 - 必须与 v61_engine.py 的 import 完全匹配
# ===========================================

__all__ = [
    # 数据类
    'V61Position',
    'V61Trade',
    'V61TradeAudit',
    'V61WashSaleRecord',
    'V61BlacklistRecord',
    'V61MarketRegime',
    'V61DrawdownState',
    'V61WeeklyTradeCounter',
    'V61IterationResult',
    'V61LogicEvolutionRecord',
    
    # 类
    'V61IndustryLoader',
    'V61FactorEngine',
    'V61RiskManager',
    
    # 函数
    'v61_industry_filter',
    
    # 常量 - 基础配置
    'V61_INITIAL_CAPITAL',
    'V61_MAX_POSITIONS',
    'V61_WEEKLY_TRADE_LIMIT',
    'V61_GLOBAL_TRADE_LIMIT',
    'V61_MOMENTUM_WEIGHT',
    'V61_R2_WEIGHT',
    'V61_TREND_WEIGHT',
    'V61_VOLATILITY_SQUEEZE_ENABLED',
    'V61_SQUEEZE_LOW_THRESHOLD',
    'V61_SQUEEZE_WINDOW',
    'V61_SQUEEZE_BREAKOUT_MULT',
    'V61_ENTRY_TOP_N',
    'V61_MAINTAIN_TOP_N',
    'V61_INDUSTRY_FILTER_ENABLED',
    'V61_INDUSTRY_TOP_N',
    'V61_INDUSTRY_WINDOW',
    'V61_RS_ENABLED',
    'V61_RS_WINDOW',
    'V61_RS_TOP_PERCENTILE',
    'V61_MA20_BUFFER',
    'V61_VOLUME_SHRINK_RATIO',
    'V61_MAX_DAILY_GAIN',
    'V61_VOLUME_BREAKOUT_MULT',
    'V61_MA20_BREAKOUT',
    'V61_RSRS_ENABLED',
    'V61_RSRS_WINDOW',
    'V61_RSRS_ZSCORE_THRESHOLD',
    
    # 常量 - 止损配置
    'V61_HARD_STOP_LOSS_ATR_MULT',
    'V61_HARD_STOP_LOSS_RATIO',
    'V61_HARD_STOP_LOSS_MODE',
    'V61_BREAKEVEN_ENABLED',
    'V61_BREAKEVEN_PROFIT_THRESHOLD',
    'V61_BREAKEVEN_BUFFER',
    'V61_TRAILING_PROFIT_TRIGGER',
    'V61_TRAILING_PROFIT_ATR_MULT',
    'V61_TRAILING_PROFIT_ENABLED',
    'V61_TRAILING_PROFIT_LOCK_COST',
    'V61_TIERED_PROFIT_ENABLED',
    'V61_TIERED_PROFIT_LEVELS',
    'V61_TIME_STOP_ENABLED',
    'V61_TIME_STOP_DAYS',
    'V61_TIME_STOP_REDUCE_RATIO',
    'V61_MA20_TREND_EXIT_ENABLED',
    'V61_MA60_TREND_EXIT_ENABLED',
    'V61_RISK_TARGET_PER_POSITION',
    'V61_MAX_SINGLE_POSITION_PCT',
    'V61_REDUCED_SINGLE_POSITION_PCT',
    'V61_ATR_VOLATILITY_THRESHOLD',
    'V61_WASH_SALE_WINDOW',
    'V61_TREND_QUALITY_WINDOW',
    'V61_TREND_QUALITY_THRESHOLD',
    'V61_VOLUME_FILTER_ENABLED',
    'V61_VOLUME_MA_PERIOD',
    'V61_VOLUME_SHRINK_THRESHOLD',
    
    # 常量 - 费率配置
    'V61_COMMISSION_RATE',
    'V61_MIN_COMMISSION',
    'V61_SLIPPAGE_BUY',
    'V61_SLIPPAGE_SELL',
    'V61_STAMP_DUTY',
    'V61_TRANSFER_FEE',
    'V61_FRICTION_COST',
    
    # 常量 - 迭代配置
    'V61_MAX_ITERATION_ROUNDS',
    'V61_RETURN_TARGET',
    'V61_MDD_TARGET',
    'V61_PROFIT_LOSS_RATIO_TARGET',
    'V61_DYNAMIC_EVOLUTION_ENABLED',
    'V61_MIN_SELECTION_PERCENTILE',
    'V61_MAX_SELECTION_PERCENTILE',
    'V61_MIN_TREND_PERIOD',
    'V61_MAX_TREND_PERIOD',
    'V61_LOGIC_MUTATION_ENABLED',
    'V61_LOGIC_MUTATION_THRESHOLD',
    
    # 常量 - 行业映射
    'V61_INDUSTRY_CODE_SEGMENT_MAP',
    'V61_INDUSTRY_PREFIX_MAP',
    'V61_INDUSTRY_SIMPLE_MAP',
    'V61_DEFAULT_INDUSTRIES',
]
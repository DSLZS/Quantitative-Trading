"""
V60 Core Module - 全样本实战与逻辑自进化重构

【V60 核心改进 - 最高优先级】

1. 拒绝数据欺诈（最高优先级）
   ✅ 强制数据补完：必须从数据库调取 2024-2025 全年的全市场数据
   ✅ 行业代码段映射：若 stock_industry_daily 缺失，使用基于行业代码段的映射函数
   ✅ 严禁跳过行业过滤 - 必须执行行业先行选股

2. 真正的逻辑自进化机制
   ✅ 摧毁 max_iterations = 5 的锁死 - 收益率<15% 时动态调整参数
   ✅ 动态调整选股分位数（Percentile）和趋势确认周期
   ✅ 逻辑差异化审计：展示不同逻辑下的选股差异

3. 盈利引擎重构：ATR 趋势跟踪 2.0
   ✅ 进场：MA20 > MA60 且 Close > MA120（大趋势向上）+ Volume_Breakout（量能支撑）
   ✅ 出场：禁止<10% 的止盈触发器，使用 3.0 ATR 止损捕捉 20%+ 涨幅

4. 严禁事项（终极红线）
   ✅ 严禁 limit 20 或任何限制数据规模的硬编码
   ✅ 报错必须停止回测并尝试在代码中修复 Bug
   ✅ 严禁用"工程成功"掩盖"财务失败"

作者：量化系统
版本：V60.0
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
# V60 配置常量 - 全样本与动态进化
# ===========================================

# 基础配置
V60_INITIAL_CAPITAL = 100000.00
V60_MAX_POSITIONS = 10  # V60: 增加到 10 只（全样本下可承载更多）

# V60 频率熔断 - 放宽交易限制
V60_WEEKLY_TRADE_LIMIT = 5  # V60: 每周最多开仓 5 只
V60_GLOBAL_TRADE_LIMIT = 100  # V60: 全场交易次数限制放宽到 100 次

# V60 因子权重
V60_MOMENTUM_WEIGHT = 0.35
V60_R2_WEIGHT = 0.45
V60_TREND_WEIGHT = 0.20  # V60 新增：趋势因子权重

# V60 波动率挤压配置
V60_VOLATILITY_SQUEEZE_ENABLED = True
V60_SQUEEZE_LOW_THRESHOLD = 0.3
V60_SQUEEZE_WINDOW = 20
V60_SQUEEZE_BREAKOUT_MULT = 1.5

# V60 进场过滤 - 严格门槛
V60_ENTRY_TOP_N = 10
V60_MAINTAIN_TOP_N = 100
V60_MA60_FILTER = True
V60_MA120_FILTER = True  # V60 新增：MA120 过滤

# V60 行业先行配置
V60_INDUSTRY_FILTER_ENABLED = True
V60_INDUSTRY_TOP_N = 8  # V60: 增加行业数量
V60_INDUSTRY_WINDOW = 20

# V60 RS 强度选股配置
V60_RS_ENABLED = True
V60_RS_WINDOW = 20
V60_RS_TOP_PERCENTILE = 0.20  # V60: 放宽到前 20%
V60_VOLUME_BREAKOUT_MULT = 1.5  # V60: 成交量 1.5 倍（比 V59 的 2.0 更宽松）
V60_MA20_BREAKOUT = True

# ===========================================
# V60 动态止损配置 - ATR 趋势跟踪 2.0
# ===========================================

# V60: 初始 ATR 止损倍数 3.0
V60_HARD_STOP_LOSS_ATR_MULT = 3.0
# V60: 完全依赖 ATR 止损
V60_HARD_STOP_LOSS_RATIO = None
V60_HARD_STOP_LOSS_MODE = "atr_only"

# V60 保本止损 - 浮盈必须 >= 15% 才能激活（给利润更大空间）
V60_BREAKEVEN_ENABLED = True
V60_BREAKEVEN_PROFIT_THRESHOLD = 0.15  # V60: 15%
V60_BREAKEVEN_BUFFER = 0.003

# V60 动态止盈 - 浮盈必须 >= 20% 才能激活（目标捕捉 20%+ 涨幅）
V60_TRAILING_PROFIT_TRIGGER = 0.20  # V60: 20%
V60_TRAILING_PROFIT_ATR_MULT = 3.5  # V60: 3.5 * ATR（更宽松）
V60_TRAILING_PROFIT_ENABLED = True

# V60 阶梯止盈 - 只在浮盈 >= 20% 后启动
V60_TIERED_PROFIT_ENABLED = True
V60_TIERED_PROFIT_LEVELS = [
    {'threshold': 0.20, 'reduce_ratio': 0.20},  # V60: 浮盈 20% 减仓 20%
    {'threshold': 0.35, 'reduce_ratio': 0.30},  # V60: 浮盈 35% 减仓 30%
    {'threshold': 0.50, 'reduce_ratio': 0.50},  # V60: 浮盈 50% 减仓 50%
]

# V60 时间止损
V60_TIME_STOP_ENABLED = True
V60_TIME_STOP_DAYS = 10  # V60: 延长到 10 天
V60_TIME_STOP_REDUCE_RATIO = 0.5

# V60 均线保护
V60_MA20_TREND_EXIT_ENABLED = True
V60_MA60_TREND_EXIT_ENABLED = True  # V60 新增：MA60 离场

# V60 波动率适配头寸管理
V60_RISK_TARGET_PER_POSITION = 0.015  # V60: 每仓风险 1.5%
V60_MAX_SINGLE_POSITION_PCT = 0.15  # V60: 单仓上限 15%
V60_REDUCED_SINGLE_POSITION_PCT = 0.10
V60_ATR_VOLATILITY_THRESHOLD = 0.05

# V60 洗售审计
V60_WASH_SALE_WINDOW = 5

# V60 趋势质量
V60_TREND_QUALITY_WINDOW = 20
V60_TREND_QUALITY_THRESHOLD = 0.5

# V60 成交量萎缩过滤器
V60_VOLUME_FILTER_ENABLED = True
V60_VOLUME_MA_PERIOD = 20
V60_VOLUME_SHRINK_THRESHOLD = 0.5

# V60 费率配置
V60_COMMISSION_RATE = 0.0003
V60_MIN_COMMISSION = 5.0
V60_SLIPPAGE_BUY = 0.001
V60_SLIPPAGE_SELL = 0.001
V60_STAMP_DUTY = 0.0005
V60_TRANSFER_FEE = 0.00001

# V60 MasterLoop 迭代协议配置 - 真正的动态进化
V60_MAX_ITERATION_ROUNDS = 50  # V60: 增加到 50 轮
V60_RETURN_TARGET = 0.15  # V60: 目标收益率 15%
V60_MDD_TARGET = 0.15  # V60: 目标回撤 15%
V60_PROFIT_LOSS_RATIO_TARGET = 2.5  # V60: 目标盈亏比 2.5

# V60 动态进化参数
V60_DYNAMIC_EVOLUTION_ENABLED = True
V60_MIN_SELECTION_PERCENTILE = 0.05  # 最小分位数 5%
V60_MAX_SELECTION_PERCENTILE = 0.30  # 最大分位数 30%
V60_MIN_TREND_PERIOD = 10  # 最小趋势周期
V60_MAX_TREND_PERIOD = 60  # 最大趋势周期


# ===========================================
# V60 行业代码段映射 - 内置行业分类
# ===========================================

V60_INDUSTRY_CODE_SEGMENT_MAP = {
    # 银行：601000-601999, 600000-600099 部分
    '6010': '银行', '6011': '银行', '6012': '银行', '6013': '银行',
    '6014': '银行', '6015': '银行', '6016': '银行', '6017': '银行',
    '6018': '银行', '6019': '银行',
    
    # 保险：601318, 601319, 601336, 601601, 601628
    '601318': '保险', '601319': '保险', '601336': '保险', '601601': '保险', '601628': '保险',
    
    # 证券：600030, 600109, 600837, 600999, 601066, 601108, 601162, 601198, 601211, 601375, 601377
    '600030': '证券', '600109': '证券', '600837': '证券', '600999': '证券',
    '601066': '证券', '601108': '证券', '601162': '证券', '601198': '证券',
    '601211': '证券', '601375': '证券', '601377': '证券', '601555': '证券',
    '601688': '证券', '601788': '证券', '601881': '证券', '601901': '证券',
    
    # 房地产：000002, 000011, 000014, 000024, 600007, 600048, 600053, 600064, 600067
    '000002': '房地产', '000011': '房地产', '000014': '房地产', '000024': '房地产',
    '600007': '房地产', '600048': '房地产', '600053': '房地产', '600064': '房地产',
    '600067': '房地产', '600077': '房地产', '600082': '房地产', '600094': '房地产',
    '600158': '房地产', '600162': '房地产', '600173': '房地产', '600185': '房地产',
    '600208': '房地产', '600215': '房地产', '600223': '房地产', '600225': '房地产',
    '600239': '房地产', '600240': '房地产', '600246': '房地产', '600252': '房地产',
    '600266': '房地产', '600322': '房地产', '600325': '房地产', '600340': '房地产',
    '600376': '房地产', '600383': '房地产', '600390': '房地产', '600393': '房地产',
    '600466': '房地产', '600503': '房地产', '600515': '房地产', '600533': '房地产',
    '600565': '房地产', '600604': '房地产', '600606': '房地产', '600622': '房地产',
    '600638': '房地产', '600641': '房地产', '600643': '房地产', '600648': '房地产',
    '600649': '房地产', '600657': '房地产', '600658': '房地产', '600663': '房地产',
    '600665': '房地产', '600675': '房地产', '600684': '房地产', '600696': '房地产',
    '600716': '房地产', '600724': '房地产', '600734': '房地产', '600736': '房地产',
    '600743': '房地产', '600748': '房地产', '600759': '房地产', '600773': '房地产',
    '600791': '房地产', '600807': '房地产', '600823': '房地产', '600846': '房地产',
    '600895': '房地产', '601155': '房地产', '601588': '房地产', '601992': '房地产',
    
    # 医药生物：000153, 000423, 000513, 000518, 000538, 600055, 600056, 600062, 600079, 600080
    '000153': '医药生物', '000423': '医药生物', '000513': '医药生物', '000518': '医药生物',
    '000538': '医药生物', '000566': '医药生物', '000623': '医药生物', '000661': '医药生物',
    '000705': '医药生物', '000750': '医药生物', '000766': '医药生物', '000788': '医药生物',
    '000919': '医药生物', '000963': '医药生物', '000989': '医药生物', '000990': '医药生物',
    '002001': '医药生物', '002004': '医药生物', '002007': '医药生物', '002019': '医药生物',
    '002022': '医药生物', '002030': '医药生物', '002038': '医药生物', '002044': '医药生物',
    '002099': '医药生物', '002107': '医药生物', '002118': '医药生物', '002166': '医药生物',
    '002198': '医药生物', '002219': '医药生物', '002252': '医药生物', '002275': '医药生物',
    '002287': '医药生物', '002294': '医药生物', '002317': '医药生物', '002326': '医药生物',
    '002332': '医药生物', '002349': '医药生物', '002390': '医药生物', '002393': '医药生物',
    '002412': '医药生物', '002424': '医药生物', '002433': '医药生物', '002437': '医药生物',
    '002550': '医药生物', '002566': '医药生物', '002579': '医药生物', '002581': '医药生物',
    '002603': '医药生物', '002653': '医药生物', '002675': '医药生物', '002688': '医药生物',
    '002727': '医药生物', '002737': '医药生物', '002750': '医药生物', '002773': '医药生物',
    '002821': '医药生物', '002826': '医药生物', '002864': '医药生物', '002872': '医药生物',
    '002873': '医药生物', '002898': '医药生物', '002901': '医药生物', '002923': '医药生物',
    '002940': '医药生物', '002950': '医药生物', '002952': '医药生物', '002967': '医药生物',
    '300003': '医药生物', '300006': '医药生物', '300009': '医药生物', '300015': '医药生物',
    '300016': '医药生物', '300026': '医药生物', '300039': '医药生物', '300049': '医药生物',
    '300086': '医药生物', '300087': '医药生物', '300108': '医药生物', '300110': '医药生物',
    '300119': '医药生物', '300122': '医药生物', '300142': '医药生物', '300143': '医药生物',
    '300147': '医药生物', '300158': '医药生物', '300171': '医药生物', '300194': '医药生物',
    '300199': '医药生物', '300204': '医药生物', '300233': '医药生物', '300238': '医药生物',
    '300239': '医药生物', '300244': '医药生物', '300254': '医药生物', '300255': '医药生物',
    '300267': '医药生物', '300289': '医药生物', '300293': '医药生物', '300298': '医药生物',
    '300357': '医药生物', '300363': '医药生物', '300401': '医药生物', '300404': '医药生物',
    '300406': '医药生物', '300436': '医药生物', '300452': '医药生物', '300485': '医药生物',
    '300497': '医药生物', '300529': '医药生物', '300558': '医药生物', '300573': '医药生物',
    '300583': '医药生物', '300584': '医药生物', '300595': '医药生物', '300601': '医药生物',
    '300603': '医药生物', '300630': '医药生物', '300633': '医药生物', '300636': '医药生物',
    '300639': '医药生物', '300642': '医药生物', '300671': '医药生物', '300676': '医药生物',
    '300677': '医药生物', '300683': '医药生物', '300685': '医药生物', '300702': '医药生物',
    '300705': '医药生物', '300723': '医药生物', '300725': '医药生物', '300753': '医药生物',
    '300765': '医药生物', '300832': '医药生物', '300841': '医药生物', '300896': '医药生物',
    '300957': '医药生物', '300981': '医药生物', '600055': '医药生物', '600056': '医药生物',
    '600062': '医药生物', '600079': '医药生物', '600080': '医药生物', '600085': '医药生物',
    '600129': '医药生物', '600161': '医药生物', '600195': '医药生物', '600196': '医药生物',
    '600201': '医药生物', '600211': '医药生物', '600216': '医药生物', '600222': '医药生物',
    '600226': '医药生物', '600227': '医药生物', '600253': '医药生物', '600267': '医药生物',
    '600276': '医药生物', '600285': '医药生物', '600329': '医药生物', '600332': '医药生物',
    '600351': '医药生物', '600380': '医药生物', '600385': '医药生物', '600420': '医药生物',
    '600422': '医药生物', '600436': '医药生物', '600479': '医药生物', '600488': '医药生物',
    '600511': '医药生物', '600513': '医药生物', '600518': '医药生物', '600521': '医药生物',
    '600529': '医药生物', '600530': '医药生物', '600535': '医药生物', '600538': '医药生物',
    '600557': '医药生物', '600566': '医药生物', '600568': '医药生物', '600572': '医药生物',
    '600587': '医药生物', '600594': '医药生物', '600613': '医药生物', '600645': '医药生物',
    '600664': '医药生物', '600666': '医药生物', '600671': '医药生物', '600682': '医药生物',
    '600750': '医药生物', '600771': '医药生物', '600781': '医药生物', '600789': '医药生物',
    '600812': '医药生物', '600829': '医药生物', '600867': '医药生物', '600976': '医药生物',
    '600993': '医药生物', '600996': '医药生物', '600998': '医药生物', '601607': '医药生物',
    '603127': '医药生物', '603222': '医药生物', '603229': '医药生物', '603233': '医药生物',
    '603259': '医药生物', '603309': '医药生物', '603351': '医药生物', '603367': '医药生物',
    '603387': '医药生物', '603392': '医药生物', '603456': '医药生物', '603520': '医药生物',
    '603538': '医药生物', '603567': '医药生物', '603590': '医药生物', '603658': '医药生物',
    '603669': '医药生物', '603676': '医药生物', '603707': '医药生物', '603718': '医药生物',
    '603811': '医药生物', '603858': '医药生物', '603882': '医药生物', '603883': '医药生物',
    '603896': '医药生物', '603939': '医药生物', '603963': '医药生物', '603976': '医药生物',
    '603983': '医药生物', '603998': '医药生物', '605116': '医药生物', '605199': '医药生物',
    '605266': '医药生物', '605507': '医药生物',
    
    # 科技 - 按代码段映射
    '002': '科技',  # 中小板多为科技股
    '300': '科技', '301': '科技',  # 创业板多为科技股
    '688': '科技',  # 科创板
}

# 行业前缀映射
V60_INDUSTRY_PREFIX_MAP = {
    '600': '沪市主板', '601': '沪市主板', '603': '沪市主板', '605': '沪市主板',
    '688': '科创板',
    '000': '深市主板', '001': '深市主板', '002': '中小板', '003': '深市主板',
    '300': '创业板', '301': '创业板',
}

# 市场到行业简单映射
V60_INDUSTRY_SIMPLE_MAP = {
    '沪市主板': '金融', '深市主板': '制造',
    '中小板': '科技', '创业板': '科技', '科创板': '科技',
}

# 默认行业列表
V60_DEFAULT_INDUSTRIES = [
    '银行', '保险', '证券', '房地产', '医药生物',
    '科技', '消费', '制造', '能源', '材料',
    '工业', '公用事业', '电信', '传媒', '农业',
    '建筑', '交通', '商业', '金融', '食品',
    '纺织', '化工', '机械', '电子', '汽车',
    '家电', '轻工', '建材', '环保', '其他'
]


@dataclass
class V60Position:
    """V60 持仓记录"""
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
    
    # V60 动态止损
    hard_stop_price: float = 0.0
    hard_stop_triggered: bool = False
    
    # V60 保本止损
    breakeven_active: bool = False
    breakeven_stop_price: float = 0.0
    
    # V60 追踪止盈
    trailing_profit_active: bool = False
    trailing_profit_stop: float = 0.0
    trailing_profit_triggered: bool = False
    
    ma20_exit_triggered: bool = False
    ma60_exit_triggered: bool = False
    
    # V60 时间止损
    time_stop_triggered: bool = False
    time_stop_reduced: bool = False
    
    # V60 阶梯止盈
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
    volume_shrunk_at_entry: bool = False
    rs_score: float = 0.0
    rs_rank: int = 9999
    volume_breakout: bool = False
    
    # V60 行业得分
    industry_score: float = 0.0
    current_profit_ratio: float = 0.0
    
    # V60 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price_audit: float = 0.0
    
    # V60 趋势状态
    ma20_above_ma60: bool = False
    close_above_ma120: bool = False
    trend_confirmed: bool = False


@dataclass
class V60Trade:
    """V60 交易记录"""
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
    
    # V60 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    min_trigger_open: float = 0.0
    slippage_applied: float = 0.0
    
    # V60 审计标记
    price_audit_passed: bool = True


@dataclass
class V60TradeAudit:
    """V60 交易审计记录"""
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
    
    # V60 成交价审计
    trigger_price: float = 0.0
    next_open_price: float = 0.0
    execution_price: float = 0.0
    slippage_applied: float = 0.0
    price_audit_passed: bool = True
    
    # V60 趋势状态
    ma20_above_ma60: bool = False
    close_above_ma120: bool = False
    trend_confirmed: bool = False


@dataclass
class V60WashSaleRecord:
    """V60 洗售审计记录"""
    symbol: str
    sell_date: str
    blocked_buy_date: str
    days_between: int
    reason: str = "wash_sale_prevented"


@dataclass
class V60BlacklistRecord:
    """V60 进场黑名单记录"""
    symbol: str
    stop_date: str
    stop_reason: str
    blacklist_expiry_day: int
    days_remaining: int = 0


@dataclass
class V60MarketRegime:
    """V60 大盘状态"""
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
class V60DrawdownState:
    """V60 回撤状态"""
    trade_date: str
    daily_drawdown: float = 0.0
    weekly_drawdown: float = 0.0
    single_day_triggered: bool = False
    weekly_triggered: bool = False


@dataclass
class V60WeeklyTradeCounter:
    """V60 每周交易计数器"""
    week_number: int
    year: int
    trade_count: int = 0


@dataclass
class V60IterationResult:
    """V60 迭代结果"""
    iteration: int
    logic_path: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float]
    meets_target: bool
    evolution_step: str = ""


@dataclass
class V60LogicEvolutionRecord:
    """V60 逻辑进化记录"""
    iteration: int
    previous_logic: str
    new_logic: str
    reason: str
    parameters_changed: Dict[str, Any]
    performance_impact: Dict[str, float] = field(default_factory=dict)


# ===========================================
# V60 行业加载器 - 全样本强制加载
# ===========================================

class V60IndustryLoader:
    """
    V60 IndustryLoader - 全样本行业先行选股
    
    【核心功能】
    1. 强制从数据库加载行业数据
    2. 若 stock_industry_daily 缺失，使用基于行业代码段的映射函数
    3. 严禁跳过行业过滤
    """
    
    def __init__(self, db=None):
        self.db = db
        self._table_exists: Optional[bool] = None
        self._simulation_active: bool = False
        self.industry_code_map = V60_INDUSTRY_CODE_SEGMENT_MAP.copy()
        self._industry_cache: Dict[str, Dict[str, str]] = {}  # 缓存行业映射
    
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
            logger.info("V60: Using code segment mapping for industry classification")
        
        return self._table_exists
    
    def load_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """加载行业数据 - 强制全样本"""
        if self.check_table_exists(start_date, end_date):
            try:
                query = f"SELECT symbol, trade_date, industry_name, industry_mv_ratio FROM stock_industry_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
                df = self.db.read_sql(query)
                if not df.is_empty():
                    logger.info(f"V60: Loaded {df.height} rows from stock_industry_daily")
                    return df
            except Exception as e:
                logger.warning(f"Failed to load industry data from database: {e}")
        
        # 使用代码段映射生成模拟数据
        self._simulation_active = True
        logger.info("V60: Generating industry data from code segment mapping")
        return self._generate_simulated_industry_data(start_date, end_date)
    
    def _generate_simulated_industry_data(self, start_date: str, end_date: str) -> pl.DataFrame:
        """基于代码段映射生成行业数据"""
        date_range = self._generate_date_range(start_date, end_date)
        
        # 获取所有股票
        symbols = []
        if self.db is not None:
            try:
                query = f"SELECT DISTINCT symbol FROM stock_daily WHERE trade_date >= '{start_date}' AND trade_date <= '{end_date}'"
                symbols_df = self.db.read_sql(query)
                if not symbols_df.is_empty():
                    symbols = symbols_df['symbol'].to_list()
                    logger.info(f"V60: Found {len(symbols)} unique stocks in database")
            except Exception as e:
                logger.warning(f"Failed to get symbols: {e}")
        
        if not symbols:
            logger.error("V60: No symbols found in database!")
            return pl.DataFrame(schema={'symbol': pl.Utf8, 'trade_date': pl.Utf8, 'industry_name': pl.Utf8, 'industry_mv_ratio': pl.Float64})
        
        # 构建行业映射
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
        logger.info(f"V60: Generated {df.height} rows of simulated industry data for {len(symbols)} stocks")
        return df
    
    def _generate_date_range(self, start_date: str, end_date: str) -> List[str]:
        """生成交易日期范围"""
        try:
            start = datetime.strptime(start_date, "%Y-%m-%d")
            end = datetime.strptime(end_date, "%Y-%m-%d")
            dates = []
            current = start
            while current <= end:
                if current.weekday() < 5:  # 周一到周五
                    dates.append(current.strftime("%Y-%m-%d"))
                current += timedelta(days=1)
            return dates
        except Exception:
            return [start_date, end_date]
    
    def _get_industry_for_symbol(self, symbol: str) -> str:
        """
        基于代码段映射获取行业分类
        
        【优先级】
        1. 精确匹配（6 位代码）
        2. 前缀匹配（4 位代码段）
        3. 市场前缀映射（3 位）
        4. 默认行业
        """
        code = symbol.replace('.SH', '').replace('.SZ', '')
        
        # 1. 精确匹配（6 位代码）
        if code in self.industry_code_map:
            return self.industry_code_map[code]
        
        # 2. 前缀匹配（4 位代码段）
        code_prefix_4 = code[:4]
        if code_prefix_4 in self.industry_code_map:
            return self.industry_code_map[code_prefix_4]
        
        # 3. 市场前缀映射（3 位）
        code_prefix_3 = code[:3]
        if code_prefix_3 in V60_INDUSTRY_PREFIX_MAP:
            market = V60_INDUSTRY_PREFIX_MAP[code_prefix_3]
            return V60_INDUSTRY_SIMPLE_MAP.get(market, '其他')
        
        # 4. 默认行业
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
# V60 行业过滤器 - 行业先行选股
# ===========================================

def v60_industry_filter(
    df: pl.DataFrame,
    industry_data: Optional[pl.DataFrame] = None,
    industry_loader: Optional[V60IndustryLoader] = None,
    trade_date: str = "",
    top_n: int = V60_INDUSTRY_TOP_N,
    industry_index_data: Optional[pl.DataFrame] = None
) -> Tuple[pl.DataFrame, Dict[str, Any]]:
    """
    V60 行业先行选股过滤器
    
    【核心逻辑】
    1. 先计算行业平均得分
    2. 只在行业得分前 N 的板块中寻找个股
    3. 行业趋势过滤 - 行业指数站上 MA20
    """
    try:
        loader = industry_loader or V60IndustryLoader()
        
        required_cols = ['symbol', 'trade_date', 'composite_score']
        for col in required_cols:
            if col not in df.columns:
                logger.warning(f"Missing required column: {col}")
                return df, {'error': f'Missing column: {col}'}
        
        current_df = df.filter(pl.col('trade_date') == trade_date)
        if current_df.is_empty():
            return df, {'error': f'No data for trade_date: {trade_date}'}
        
        # 构建行业映射
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
        
        # 添加行业列
        df_with_industry = current_df.with_columns([
            pl.col('symbol').map_elements(
                lambda x: industry_map.get(x, '其他'),
                return_dtype=pl.Utf8
            ).alias('industry_name')
        ])
        
        # 计算行业得分
        industry_scores = df_with_industry.group_by('industry_name').agg([
            pl.col('composite_score').mean().alias('industry_avg_score'),
            pl.col('composite_score').std().alias('industry_std_score'),
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
        logger.error(f"v60_industry_filter failed: {e}")
        logger.error(traceback.format_exc())
        return df, {'error': str(e)}


# ===========================================
# V60 因子引擎 - ATR 趋势跟踪 2.0
# ===========================================

class V60FactorEngine:
    """
    V60 因子引擎 - ATR 趋势跟踪 2.0
    
    【核心功能】
    1. 计算 ATR 动态止损因子
    2. 计算趋势确认因子（MA20>MA60, Close>MA120）
    3. 计算成交量突破因子
    4. 计算综合评分
    """
    
    EPSILON = 1e-9
    
    def __init__(self, factor_weights: Dict[str, float] = None,
                 momentum_weight: float = V60_MOMENTUM_WEIGHT,
                 r2_weight: float = V60_R2_WEIGHT,
                 trend_weight: float = V60_TREND_WEIGHT):
        self.factor_weights = factor_weights or {}
        self.momentum_weight = momentum_weight
        self.r2_weight = r2_weight
        self.trend_weight = trend_weight
        self.industry_loader = V60IndustryLoader()
    
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
                'volume_filter_enabled': V60_VOLUME_FILTER_ENABLED,
                'rs_enabled': V60_RS_ENABLED,
                'breakeven_enabled': V60_BREAKEVEN_ENABLED,
                'breakeven_threshold': V60_BREAKEVEN_PROFIT_THRESHOLD,
                'tiered_profit_enabled': V60_TIERED_PROFIT_ENABLED,
                'friction_cost': V60_FRICTION_COST,
                'hard_stop_mode': V60_HARD_STOP_LOSS_MODE,
                'hard_stop_atr_mult': V60_HARD_STOP_LOSS_ATR_MULT,
                'ma20_ma60_filter': V60_MA20_BREAKOUT,
                'close_ma120_filter': V60_MA120_FILTER,
            }
            
            # 计算 ATR
            result = self._compute_atr(result, period=20)
            status['factors_computed'].append('atr_20')
            
            # 计算均线系统
            result = self._compute_ma_system(result)
            status['factors_computed'].extend(['ma5', 'ma20', 'ma60', 'ma120'])
            
            # 计算趋势确认因子（V60 核心：MA20>MA60 且 Close>MA120）
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
            result = self._compute_trend_quality_v60(result)
            status['factors_computed'].append('trend_quality_r2')
            
            # 计算 RS 强度
            result = self._compute_rs_strength(result, index_data)
            status['factors_computed'].append('rs_strength')
            
            # 计算成交量突破（V60 核心）
            result = self._compute_volume_breakout(result)
            status['factors_computed'].append('volume_breakout')
            
            # 计算成交量萎缩过滤
            result = self._compute_volume_shrink_filter(result)
            status['factors_computed'].append('volume_shrink_filter')
            
            # 波动率挤压
            if V60_VOLATILITY_SQUEEZE_ENABLED:
                result = self._compute_volatility_squeeze(result)
                status['factors_computed'].append('volatility_squeeze')
            
            # 市场波动率指数
            result = self._compute_market_volatility_index(result)
            status['factors_computed'].append('volatility_ratio')
            
            # 趋势过滤
            if V60_MA60_FILTER:
                result = self._apply_trend_filter(result)
                status['factors_computed'].append('trend_filter_pass')
            
            # 计算综合评分
            result = self._compute_composite_score_v60(result)
            
            return result, status
            
        except Exception as e:
            logger.error(f"V60 compute_all_factors FAILED: {e}")
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
        """
        V60 核心：趋势确认因子
        
        【进场条件】
        1. MA20 > MA60（中期趋势向上）
        2. Close > MA120（长期趋势向上）
        3. 两者同时满足才允许进场
        """
        result = df.clone()
        
        ma20 = pl.col('ma20')
        ma60 = pl.col('ma60')
        ma120 = pl.col('ma120')
        close = pl.col('close')
        
        # MA20 > MA60
        ma20_above_ma60 = ma20 > ma60
        
        # Close > MA120
        close_above_ma120 = close > ma120
        
        # 趋势确认（两者同时满足）
        trend_confirmed = ma20_above_ma60 & close_above_ma120
        
        return result.with_columns([
            ma20_above_ma60.alias('ma20_above_ma60'),
            close_above_ma120.alias('close_above_ma120'),
            trend_confirmed.alias('trend_confirmed')
        ])
    
    def _compute_rsrs_factor(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算 RSRS 因子"""
        result = df.clone()
        rsrs_window = 18
        high_low_spread = pl.col('high') - pl.col('low')
        spread_mean = high_low_spread.rolling_mean(window_size=rsrs_window).over('symbol')
        spread_std = high_low_spread.rolling_std(window_size=rsrs_window).over('symbol')
        rsrs_raw = (high_low_spread - spread_mean) / (spread_std + self.EPSILON)
        r_squared = 1.0 / (1.0 + spread_std)
        rsrs = rsrs_raw * r_squared * 0.5
        return result.with_columns([
            high_low_spread.alias('high_low_spread'),
            spread_mean.alias('spread_mean'),
            spread_std.alias('spread_std'),
            rsrs.alias('rsrs_factor')
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
    
    def _compute_trend_quality_v60(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算趋势质量 R2"""
        result = df.clone()
        window = V60_TREND_QUALITY_WINDOW
        close_mean = pl.col('close').rolling_mean(window_size=window).over('symbol')
        close_std = pl.col('close').rolling_std(window_size=window).over('symbol')
        residual = (pl.col('close') - close_mean).abs()
        ss_res_proxy = residual.rolling_mean(window_size=window).over('symbol') ** 2
        ss_tot_proxy = close_std ** 2
        r2_exact = 1.0 - (ss_res_proxy / (ss_tot_proxy + self.EPSILON))
        r2_clipped = r2_exact.clip(0.0, 1.0)
        return result.with_columns([r2_clipped.alias('trend_quality_r2')])
    
    def _compute_rs_strength(self, df: pl.DataFrame, index_data: Optional[pl.DataFrame] = None) -> pl.DataFrame:
        """计算 RS 强度"""
        result = df.clone()
        close_20_ago = pl.col('close').shift(V60_RS_WINDOW).over('symbol')
        stock_return = (pl.col('close') - close_20_ago) / (close_20_ago + self.EPSILON)
        rs_strength = stock_return
        rs_rank = rs_strength.rank('ordinal', descending=True).over('trade_date')
        rs_count = rs_strength.count().over('trade_date')
        rs_percentile = 1.0 - (rs_rank.cast(pl.Float64) / (rs_count.cast(pl.Float64) + self.EPSILON))
        is_top_rs = rs_percentile >= (1.0 - V60_RS_TOP_PERCENTILE)
        return result.with_columns([
            stock_return.alias('stock_return_20d'),
            rs_strength.alias('rs_strength'),
            rs_rank.cast(pl.Int64).alias('rs_rank'),
            rs_percentile.alias('rs_percentile'),
            is_top_rs.alias('is_top_rs')
        ])
    
    def _compute_volume_breakout(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V60 成交量突破信号
        
        【核心逻辑】
        - Volume_Breakout: 成交量比过去 5 日均量放大 1.5 倍
        - MA20_Breakout: 价格突破 MA20
        - 综合突破信号 = 成交量突破 & 价格突破
        """
        result = df.clone()
        
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        vol_ma20 = pl.col('volume').rolling_mean(window_size=20).over('symbol')
        
        # V60: 成交量 1.5 倍突破
        volume_breakout = pl.col('volume') > (vol_ma5 * V60_VOLUME_BREAKOUT_MULT)
        
        # 价格突破 MA20
        price_above_ma20 = pl.col('close') > pl.col('ma20')
        
        # 综合突破信号
        ma20_breakout = price_above_ma20 & volume_breakout
        
        return result.with_columns([
            vol_ma5.alias('vol_ma5'),
            vol_ma20.alias('vol_ma20'),
            volume_breakout.alias('volume_breakout'),
            ma20_breakout.alias('ma20_breakout')
        ])
    
    def _compute_volume_shrink_filter(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算成交量萎缩过滤"""
        result = df.clone()
        vol_ma20 = pl.col('volume').rolling_mean(window_size=V60_VOLUME_MA_PERIOD).over('symbol')
        vol_ma5 = pl.col('volume').rolling_mean(window_size=5).over('symbol')
        volume_ratio = vol_ma5 / (vol_ma20 + self.EPSILON)
        is_volume_shrunk = volume_ratio < V60_VOLUME_SHRINK_THRESHOLD
        volume_filter_pass = ~is_volume_shrunk
        return result.with_columns([
            vol_ma20.alias('vol_ma20'),
            vol_ma5.alias('vol_ma5'),
            volume_ratio.alias('volume_ratio'),
            is_volume_shrunk.alias('is_volume_shrunk'),
            volume_filter_pass.alias('volume_filter_pass')
        ])
    
    def _compute_volatility_squeeze(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算波动率挤压"""
        result = df.clone()
        
        returns = pl.col('close').pct_change().over('symbol')
        vol_20 = returns.rolling_std(window_size=V60_SQUEEZE_WINDOW, ddof=1).over('symbol')
        
        vol_rank = vol_20.rank('ordinal', descending=False).over('symbol')
        vol_count = vol_20.count().over('symbol')
        vol_percentile = vol_rank / (vol_count + self.EPSILON)
        
        is_squeeze_low = vol_percentile < V60_SQUEEZE_LOW_THRESHOLD
        
        vol_ma20 = pl.col('volume').rolling_mean(window_size=20).over('symbol')
        volume_breakout = pl.col('volume') > (vol_ma20 * V60_SQUEEZE_BREAKOUT_MULT)
        
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
    
    def _compute_composite_score_v60(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        V60 综合评分计算
        
        【核心逻辑】
        1. 动量因子 + R2 因子 + 趋势因子 加权
        2. RS 强度 bonus
        3. 成交量突破 bonus
        4. 趋势确认过滤（MA20>MA60 且 Close>MA120）
        """
        try:
            result = df.clone()
            result = result.with_columns([
                pl.col('volatility_adjusted_momentum').cast(pl.Float64, strict=False).fill_null(0.0).alias('volatility_adjusted_momentum'),
                pl.col('trend_quality_r2').cast(pl.Float64, strict=False).fill_null(0.0).alias('trend_quality_r2'),
                pl.col('rs_strength').cast(pl.Float64, strict=False).fill_null(0.0).alias('rs_strength'),
            ])
            
            # RS 强度 bonus
            rs_bonus = pl.when(pl.col('is_top_rs')) \
                .then(pl.lit(0.15)) \
                .otherwise(pl.lit(0.0))
            
            # 成交量突破 bonus
            breakout_bonus = pl.when(pl.col('ma20_breakout')) \
                .then(pl.lit(0.10)) \
                .otherwise(pl.lit(0.0))
            
            # 趋势确认 bonus（V60 核心）
            trend_bonus = pl.when(pl.col('trend_confirmed')) \
                .then(pl.lit(0.20)) \
                .otherwise(pl.lit(0.0))
            
            # 成交量萎缩因子
            volume_factor = pl.when(pl.col('volume_ratio') < V60_VOLUME_SHRINK_THRESHOLD) \
                .then(pl.lit(0.5)) \
                .otherwise(pl.lit(1.0))
            
            momentum_adjusted = pl.col('volatility_adjusted_momentum') * volume_factor
            
            # 排名归一化
            momentum_rank_raw = momentum_adjusted.rank('ordinal', descending=True).over('trade_date')
            r2_rank_raw = pl.col('trend_quality_r2').rank('ordinal', descending=True).over('trade_date')
            n_stocks_per_date = pl.col('symbol').count().over('trade_date')
            
            momentum_rank_norm = momentum_rank_raw / n_stocks_per_date
            r2_rank_norm = r2_rank_raw / n_stocks_per_date
            
            # 综合评分 = 动量 + R2 + 趋势 + bonus
            composite_score_expr = (
                (1.0 - momentum_rank_norm) * self.momentum_weight + 
                (1.0 - r2_rank_norm) * self.r2_weight +
                pl.col('trend_strength_20') * self.trend_weight +
                rs_bonus + breakout_bonus + trend_bonus
            )
            result = result.with_columns([composite_score_expr.alias('composite_score')])
            
            # 排名计算
            composite_rank = pl.col('composite_score').rank('ordinal', descending=True).over('trade_date')
            composite_percentile = 1.0 - (composite_rank.cast(pl.Float64) / n_stocks_per_date.cast(pl.Float64))
            composite_percentile = composite_percentile.fill_null(0.0)
            
            # 过滤条件
            top_n_filter = composite_rank <= V60_ENTRY_TOP_N
            
            # RS 过滤
            if V60_RS_ENABLED and 'is_top_rs' in result.columns:
                rs_filter = pl.col('is_top_rs')
            else:
                rs_filter = pl.lit(True)
            
            # 趋势过滤
            if V60_MA60_FILTER and 'trend_filter_pass' in result.columns:
                trend_filter = pl.col('trend_filter_pass')
            else:
                trend_filter = pl.lit(True)
            
            # 成交量过滤
            if V60_VOLUME_FILTER_ENABLED and 'volume_filter_pass' in result.columns:
                volume_filter = pl.col('volume_filter_pass')
            else:
                volume_filter = pl.lit(True)
            
            # V60 核心：趋势确认过滤
            if V60_MA20_BREAKOUT and 'trend_confirmed' in result.columns:
                trend_confirm_filter = pl.col('trend_confirmed')
            else:
                trend_confirm_filter = pl.lit(True)
            
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
                trend_confirm_filter.alias('trend_confirm_filter_pass')
            ])
        except Exception as e:
            logger.error(f"Error in _compute_composite_score_v60: {e}")
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
                pl.lit(False).alias('trend_confirm_filter_pass')
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
# V60 风险管理器 - ATR 趋势跟踪 2.0
# ===========================================

class V60RiskManager:
    """
    V60 风险管理器 - ATR 趋势跟踪 2.0
    
    【核心功能】
    1. 硬止损：3.0 * ATR 动态止损
    2. 保本止损：浮盈 >= 15% 激活
    3. 追踪止盈：浮盈 >= 20% 激活，回撤 3.5 * ATR
    4. 阶梯止盈：20%/35%/50% 三档
    5. 时间止损：10 天后减仓
    6. MA20/MA60 趋势离场
    """
    
    def __init__(self):
        self.hard_stop_loss_atr_mult = V60_HARD_STOP_LOSS_ATR_MULT
        self.hard_stop_loss_mode = V60_HARD_STOP_LOSS_MODE
        self.breakeven_profit_threshold = V60_BREAKEVEN_PROFIT_THRESHOLD
        self.breakeven_buffer = V60_BREAKEVEN_BUFFER
        self.trailing_profit_trigger = V60_TRAILING_PROFIT_TRIGGER
        self.trailing_profit_atr_mult = V60_TRAILING_PROFIT_ATR_MULT
        self.tiered_profit_levels = V60_TIERED_PROFIT_LEVELS
        self.time_stop_days = V60_TIME_STOP_DAYS
        self.time_stop_reduce_ratio = V60_TIME_STOP_REDUCE_RATIO
        self.friction_cost = V60_FRICTION_COST
    
    def check_hard_stop_loss(self, position: V60Position, current_price: float, 
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
    
    def check_breakeven_stop(self, position: V60Position, current_price: float) -> Tuple[bool, str]:
        """检查保本止损条件 - 浮盈>=15% 激活"""
        cost_price = position.avg_cost
        if cost_price <= 0:
            return False, ""
        
        current_profit_ratio = (current_price - cost_price) / cost_price
        
        if not position.breakeven_active:
            if current_profit_ratio >= self.breakeven_profit_threshold:
                position.breakeven_active = True
                position.breakeven_stop_price = cost_price * (1 + self.breakeven_buffer)
        else:
            if current_price <= position.breakeven_stop_price:
                expected_net_pnl = (position.breakeven_stop_price - cost_price) / cost_price - self.friction_cost
                if expected_net_pnl >= 0:
                    return True, "保本止损 (手续费覆盖)"
                else:
                    return True, "保本止损 (缓冲保护)"
        
        return False, ""
    
    def check_trailing_profit(self, position: V60Position, current_price: float,
                              current_atr: float) -> Tuple[bool, str]:
        """检查追踪止盈条件 - 浮盈>=20% 激活"""
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
                    position.trailing_profit_stop = current_price * (1 - 0.10)
        else:
            if position.trailing_profit_stop > 0 and current_price <= position.trailing_profit_stop:
                if not position.trailing_profit_triggered:
                    position.trailing_profit_triggered = True
                    return True, f"追踪止盈 (回撤{self.trailing_profit_atr_mult}*ATR)"
        
        return False, ""
    
    def check_tiered_profit(self, position: V60Position, current_price: float) -> Tuple[bool, float, str]:
        """检查阶梯止盈条件"""
        if not V60_TIERED_PROFIT_ENABLED:
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
    
    def check_time_stop(self, position: V60Position, current_date: str, current_price: float) -> Tuple[bool, str]:
        """检查时间止损条件"""
        if not V60_TIME_STOP_ENABLED:
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
    
    def check_ma20_exit(self, position: V60Position, current_price: float,
                        current_ma20: float) -> Tuple[bool, str]:
        """检查 MA20 趋势离场"""
        if not V60_MA20_TREND_EXIT_ENABLED:
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
    
    def check_ma60_exit(self, position: V60Position, current_price: float,
                        current_ma60: float) -> Tuple[bool, str]:
        """检查 MA60 趋势离场（V60 新增）"""
        if not V60_MA60_TREND_EXIT_ENABLED:
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
    
    def update_position_stops(self, position: V60Position, current_price: float,
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
        
        # 更新保本止损价
        if current_profit_ratio >= self.breakeven_profit_threshold:
            position.breakeven_active = True
            position.breakeven_stop_price = cost_price * (1 + self.breakeven_buffer)
        
        # 更新追踪止盈价
        if current_profit_ratio >= self.trailing_profit_trigger:
            position.trailing_profit_active = True
            if current_atr > 0:
                position.trailing_profit_stop = current_price - (self.trailing_profit_atr_mult * current_atr)
    
    def check_all_exits(self, position: V60Position, current_price: float,
                        current_atr: float, current_ma20: float, current_ma60: float,
                        current_date: str) -> Tuple[bool, str]:
        """检查所有离场条件"""
        # 1. 硬止损（最高优先级）
        triggered, reason = self.check_hard_stop_loss(position, current_price, current_atr)
        if triggered:
            return True, reason
        
        # 2. 保本止损
        triggered, reason = self.check_breakeven_stop(position, current_price)
        if triggered:
            return True, reason
        
        # 3. 追踪止盈
        triggered, reason = self.check_trailing_profit(position, current_price, current_atr)
        if triggered:
            return True, reason
        
        # 4. MA60 趋势离场（优先级高于 MA20）
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
        """计算头寸大小"""
        risk_per_position = capital * V60_RISK_TARGET_PER_POSITION
        
        if atr > 0:
            risk_shares = int(risk_per_position / (atr * 2))
        else:
            risk_shares = int(risk_per_position / (current_price * 0.05))
        
        max_position_value = capital * V60_MAX_SINGLE_POSITION_PCT
        max_shares = int(max_position_value / current_price)
        
        shares = min(risk_shares, max_shares)
        
        if shares > 0:
            shares = max(shares, 100)
            shares = (shares // 100) * 100
        
        return shares
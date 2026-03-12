"""
Daily Trade Advisor - 本地量化实战决策系统

集成"量化模型评分 + 双引擎 AI 审计 + 大盘择时"的实战决策系统，
严格执行"宁缺毋滥"原则，为 5 万元小额本金提供每日操作指令。

核心流程（分层过滤）:
    1. 第一层：量化门槛与 Top 10 提取
       - 获取 T 日收盘行情，加载模型并推理
       - 仅保留预测上涨概率 P >= 70% 的股票
       - 若无股票满足 70%，触发"空仓策略"
       - 若满足条件的股票多于 10 只，仅取 Top 10 进入审计环节
    
    2. 第二层：混合 AI 分级审计（黄金组合策略）
       - 步骤 A: 免费模型初筛（Qwen）- 快速总结新闻要点
       - 步骤 B: DeepSeek 核心精审 - 严格风控审计
       - 判定原则：宁缺毋滥，PASS 或 REJECT
    
    3. 第三层：大盘环境择时（仓位控制器）
       - 计算中证 500 指数 20 日均线
       - 若价格在均线下，判定为"防守模式"，预算减半

资金执行（5 万元约束）:
    - 预算分配：总本金 50,000 元，最多 3 只股票
    - 碎股取整：严格执行 100 股整数倍向下取整
    - 成本计算：预扣除佣金（每笔最低 5 元）和印花税
    - 自动补位：剩余现金自动买入国债 ETF (511010)

使用示例:
    >>> python src/daily_trade_advisor.py
    # 输出明日决策清单（Markdown 格式）
"""

import os
import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from decimal import Decimal

import numpy as np
import polars as pl
import yaml
import akshare as ak
from loguru import logger
from dotenv import load_dotenv

# 加载环境变量（优先从 .env 文件读取）
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import DatabaseManager
from factor_engine import FactorEngine


# ===========================================
# Enums and Data Classes
# ===========================================

class AuditStatus(Enum):
    """AI 审计状态枚举"""
    PENDING = "pending"       # 待审计
    PASS = "pass"             # 通过
    REJECT = "reject"         # 驳回
    ERROR = "error"           # 审计出错


class RunMode(Enum):
    """运行模式枚举 - 两段式任务支持"""
    DRAFT = "draft"           # 初稿模式（15:05 运行，基于收盘表现预警）
    FINAL = "final"           # 终稿模式（21:00 运行，结合盘后公告终审）


class MarketMode(Enum):
    """市场模式枚举"""
    NORMAL = "normal"         # 正常模式
    DEFENSIVE = "defensive"   # 防守模式（空仓避险）


@dataclass
class StockCandidate:
    """股票候选对象"""
    symbol: str
    name: str
    close: float
    predict_prob: float       # 预测上涨概率
    rank: int                 # 排名
    
    # AI 审计相关
    audit_status: AuditStatus = AuditStatus.PENDING
    audit_reason: str = ""    # 审计原因/摘要
    news_summary: str = ""    # 新闻摘要（初筛后）
    
    # 交易执行相关
    recommended_shares: int = 0       # 建议股数
    estimated_amount: float = 0.0     # 预估金额
    commission: float = 0.0           # 佣金
    stamp_duty: float = 0.0           # 印花税


@dataclass
class AuditResult:
    """AI 审计结果"""
    symbol: str
    status: AuditStatus
    reason: str
    news_summary: str = ""
    token_usage: int = 0


@dataclass
class TradeDecision:
    """交易决策对象"""
    symbol: str
    name: str
    shares: int               # 股数
    price: float              # 价格
    amount: float             # 金额
    commission: float         # 佣金
    stamp_duty: float         # 印花税
    predict_prob: float       # 预测概率
    audit_summary: str        # AI 审计摘要


@dataclass
class ReportContext:
    """报表上下文"""
    trade_date: str
    market_mode: MarketMode
    regime_ma_value: Optional[float]
    current_price: Optional[float]
    run_mode: RunMode = RunMode.FINAL  # 运行模式
    
    # 候选股票
    top_10_candidates: list[StockCandidate] = field(default_factory=list)
    
    # 最终决策
    decisions: list[TradeDecision] = field(default_factory=list)
    
    # 审计回溯
    audit_trail: list[dict] = field(default_factory=list)
    
    # 资金统计
    total_capital: float = 50000.0
    used_capital: float = 0.0
    remaining_capital: float = 0.0
    bond_etf_shares: int = 0
    bond_etf_amount: float = 0.0
    
    # API 统计
    api_stats: dict = field(default_factory=dict)


# ===========================================
# Configuration Manager
# ===========================================

class ConfigManager:
    """配置管理器 - 支持环境变量优先读取和安全检测"""
    
    # 敏感配置键名列表
    SENSITIVE_KEYS = [
        'api_key', 'password', 'secret', 'token', 'key',
        'DEEPSEEK_API_KEY', 'QWEN_API_KEY', 'KIMI_API_KEY', 'MINIMAX_API_KEY'
    ]
    
    # 占位符模式列表
    PLACEHOLDER_PATTERNS = [
        'YOUR_', '${', '<YOUR', '{{', 'REPLACE_', 'CHANGE_'
    ]
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        初始化配置管理器。
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._expand_env_vars()
        self._security_check()  # 安全检查
        
    def _load_config(self) -> dict:
        """加载 YAML 配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def _is_placeholder(self, value: str) -> bool:
        """
        检查值是否为占位符。
        
        Args:
            value: 要检查的值
            
        Returns:
            True 如果是占位符，False 否则
        """
        if not isinstance(value, str):
            return False
        
        # 检查是否包含占位符模式
        for pattern in self.PLACEHOLDER_PATTERNS:
            if pattern in value.upper():
                return True
        
        # 检查是否为空或过短
        if len(value) < 8 or value in ['', 'None', 'null', 'NULL', 'none']:
            return True
        
        return False
    
    def _is_hardcoded_secret(self, key: str, value: str) -> bool:
        """
        检查是否为硬编码的敏感信息。
        
        Args:
            key: 配置键名
            value: 配置值
            
        Returns:
            True 如果是硬编码的敏感信息，False 否则
        """
        # 检查键名是否敏感
        key_upper = key.upper()
        is_sensitive_key = any(s in key_upper for s in self.SENSITIVE_KEYS)
        
        if not is_sensitive_key:
            return False
        
        # 如果是敏感键且不是占位符，则是硬编码的秘密
        return not self._is_placeholder(value)
    
    def _check_secrets_recursive(self, obj: Any, path: str = "") -> list[str]:
        """
        递归检查配置中的硬编码秘密。
        
        Args:
            obj: 要检查的对象
            path: 当前路径
            
        Returns:
            发现的硬编码秘密路径列表
        """
        hardcoded_secrets = []
        
        if isinstance(obj, dict):
            for key, value in obj.items():
                current_path = f"{path}.{key}" if path else key
                if isinstance(value, str):
                    if self._is_hardcoded_secret(key, value):
                        hardcoded_secrets.append(current_path)
                elif isinstance(value, (dict, list)):
                    hardcoded_secrets.extend(self._check_secrets_recursive(value, current_path))
        elif isinstance(obj, list):
            for i, item in enumerate(obj):
                current_path = f"{path}[{i}]"
                if isinstance(item, str):
                    if self._is_hardcoded_secret(f"{path}[{i}]", item):
                        hardcoded_secrets.append(current_path)
                elif isinstance(item, (dict, list)):
                    hardcoded_secrets.extend(self._check_secrets_recursive(item, current_path))
        
        return hardcoded_secrets
    
    def _security_check(self) -> None:
        """
        安全检查：检测配置文件中是否包含硬编码的敏感信息。
        
        如果检测到真实的 API Key 而非占位符，将发出明显的控制台警告。
        """
        hardcoded_secrets = self._check_secrets_recursive(self.config)
        
        if hardcoded_secrets:
            # 打印明显的警告信息
            print("\n" + "=" * 70)
            print("⚠️  ⚠️  ⚠️  SECURITY WARNING - 安全警告  ⚠️  ⚠️  ⚠️")
            print("=" * 70)
            print("\n检测到配置文件中包含硬编码的敏感信息！")
            print("Detected hardcoded sensitive information in configuration file!")
            print("\n以下配置项包含真实的 API Key 或密码（而非占位符）：\n")
            
            for secret_path in hardcoded_secrets:
                print(f"  ❌ {secret_path}")
            
            print("\n" + "-" * 70)
            print("建议操作 / Recommended Actions:")
            print("-" * 70)
            print("1. 立即将 config/settings.yaml 中的真实 Key 移至环境变量")
            print("2. 使用 .env 文件存储敏感信息（确保 .env 已添加到 .gitignore）")
            print("3. 将 config/settings.yaml 中的 Key 替换为占位符")
            print("\n示例 - 使用环境变量:")
            print("  api_key: ${QWEN_API_KEY}")
            print("  或")
            print("  api_key: ${DEEPSEEK_API_KEY}")
            print("\n示例 - .env 文件格式:")
            print("  QWEN_API_KEY=sk-your-actual-api-key-here")
            print("  DEEPSEEK_API_KEY=sk-your-actual-api-key-here")
            print("=" * 70 + "\n")
            
            # 同时记录到日志
            logger.warning(f"Security check: Found {len(hardcoded_secrets)} hardcoded sensitive config(s)")
    
    def _expand_env_vars(self) -> None:
        """
        展开环境变量 - 优先从环境变量读取，若不存在再尝试读取配置文件。
        
        优先级顺序：
        1. 环境变量 (Environment Variables)
        2. .env 文件 (通过 python-dotenv 加载)
        3. 配置文件中的 ${VAR} 占位符
        """
        def _expand(obj: Any) -> Any:
            if isinstance(obj, str):
                # 处理 ${VAR} 格式的环境变量引用
                if obj.startswith("${") and obj.endswith("}"):
                    env_var = obj[2:-1]
                    # 优先从环境变量读取
                    env_value = os.getenv(env_var)
                    if env_value is not None:
                        return env_value
                    # 如果环境变量不存在，返回原始占位符（会在安全检查中警告）
                    return obj
                return obj
            elif isinstance(obj, dict):
                return {k: _expand(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [_expand(item) for item in obj]
            return obj
        
        self.config = _expand(self.config)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        获取配置值。
        
        Args:
            key: 配置键（支持点号分隔，如 'trading.capital'）
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value


# ===========================================
# AI Agent Base Class
# ===========================================

class AIAgentBase:
    """AI Agent 基类"""
    
    def __init__(self, config: dict):
        """
        初始化 AI Agent。
        
        Args:
            config: AI 配置字典
        """
        self.config = config
        self.provider = config.get('provider', 'unknown')
        self.api_key = config.get('api_key', '')
        self.base_url = config.get('base_url', '')
        self.model = config.get('model', 'default')
        self.max_tokens = config.get('max_tokens', 500)
    
    def call(self, messages: list[dict]) -> tuple[str, int]:
        """
        调用 AI API。
        
        Args:
            messages: 消息列表
            
        Returns:
            (response_text, token_usage)
        """
        raise NotImplementedError
    
    def summarize_news(self, news_list: list[str]) -> str:
        """
        总结新闻。
        
        Args:
            news_list: 新闻列表
            
        Returns:
            新闻摘要
        """
        raise NotImplementedError
    
    def audit_stock(self, stock_info: dict, news_summary: str) -> AuditResult:
        """
        审计股票。
        
        Args:
            stock_info: 股票信息
            news_summary: 新闻摘要
            
        Returns:
            审计结果
        """
        raise NotImplementedError


class MockQwenAgent(AIAgentBase):
    """Mock 通义千问 AI Agent（用于测试，无需 API 密钥）"""
    
    def __init__(self, config: dict):
        super().__init__(config)
    
    def call(self, messages: list[dict]) -> tuple[str, int]:
        """Mock 调用"""
        return "Mock 新闻摘要：无重大负面信息，公司经营正常。", 50
    
    def summarize_news(self, news_list: list[str]) -> str:
        """Mock 新闻总结"""
        if not news_list:
            return "无相关新闻"
        return f"Mock 摘要：共{len(news_list)}条新闻，无重大负面信息。"


class MockDeepSeekAgent(AIAgentBase):
    """Mock DeepSeek AI Agent（用于测试，无需 API 密钥）"""
    
    def __init__(self, config: dict):
        super().__init__(config)
    
    def call(self, messages: list[dict]) -> tuple[str, int]:
        """Mock 调用"""
        return '{"status": "PASS", "reason": "Mock 审计通过，无风险", "risk_level": "无", "keywords_found": []}', 100
    
    def audit_stock(self, stock_info: dict, news_summary: str) -> AuditResult:
        """Mock 审计 - 默认通过"""
        symbol = stock_info.get('symbol', '')
        
        # Mock 审计逻辑：默认通过，但可以模拟特定股票被驳回
        # 例如：如果股票代码包含 "ST"，则驳回
        if "ST" in symbol.upper():
            return AuditResult(
                symbol=symbol,
                status=AuditStatus.REJECT,
                reason="Mock 审计：发现 ST 标识，存在退市风险",
                news_summary=news_summary,
                token_usage=50
            )
        
        return AuditResult(
            symbol=symbol,
            status=AuditStatus.PASS,
            reason="Mock 审计通过，无风险",
            news_summary=news_summary,
            token_usage=100
        )


class QwenAgent(AIAgentBase):
    """
    通义千问 AI Agent（用于初筛）
    
    按照阿里云 Coding Plan 规范强制配置:
    - base_url: https://coding.dashscope.aliyuncs.com/v1 (硬编码)
    - model: qwen3.5-plus (硬编码)
    - api_key: 直接从环境变量 QWEN_API_KEY 读取 (必须 sk-sp-开头)
    """
    
    # Coding Plan 专用 Key 格式前缀
    CODING_PLAN_KEY_PREFIX = "sk-sp-"
    
    def __init__(self, config: dict):
        # 不调用 super().__init__(config)，完全接管配置加载
        self.config = config
        self.provider = 'qwen'
        self.max_tokens = config.get('max_tokens', 500)
        
        from openai import OpenAI
        
        # ========== 强制 Key 加载 (Key Enforcement) ==========
        # 直接从环境变量读取，不依赖 config.get('api_key')
        self.api_key = os.getenv("QWEN_API_KEY", "")
        
        # ========== 安全校验 (Security Validation) ==========
        # Coding Plan 必须使用 sk-sp- 开头的 Key
        if not self.api_key:
            logger.error("❌ QWEN_API_KEY 未配置！请在 .env 文件中设置 QWEN_API_KEY")
            raise ValueError("QWEN_API_KEY is required for Coding Plan")
        
        if not self.api_key.startswith(self.CODING_PLAN_KEY_PREFIX):
            logger.error(
                f"❌ Qwen API Key 格式错误！Coding Plan 的 Key 必须以 '{self.CODING_PLAN_KEY_PREFIX}' 开头。"
                f"当前 Key 以 '{self.api_key[:6] if len(self.api_key) >= 6 else self.api_key}' 开头。"
                f"请确认您使用的是阿里云 Coding Plan 的 API Key，而非其他服务。"
            )
            # 继续执行但记录警告，让用户知道可能有问题
        
        # ========== 强制 URL 与模型对齐 (Endpoint Alignment) ==========
        # 硬编码 base_url，废弃 settings.yaml 中的配置
        self.base_url = "https://coding.dashscope.aliyuncs.com/v1"
        
        # 硬编码 model 名称
        self.model = "qwen3.5-plus"
        
        # ========== OpenAI 客户端实例化加固 ==========
        logger.debug(f"Qwen Client Auth: Key starting with {self.api_key[:7]}, URL: {self.base_url}")
        logger.info(f"Qwen Agent initialized with model={self.model}, base_url={self.base_url}")
        
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url
        )
    
    def call(self, messages: list[dict]) -> tuple[str, int]:
        """
        调用通义千问 API（使用 OpenAI SDK）
        
        Args:
            messages: 消息列表
            
        Returns:
            (response_text, token_usage)
        """
        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            content = completion.choices[0].message.content
            usage = completion.usage.total_tokens if completion.usage else 0
            
            return content, usage
            
        except Exception as e:
            logger.error(f"Qwen API call failed: {e}")
            return f"API 调用失败：{e}", 0
    
    def summarize_news(self, news_list: list[str]) -> str:
        """
        快速总结新闻要点，剔除无关信息。
        
        Args:
            news_list: 新闻列表
            
        Returns:
            新闻摘要
        """
        if not news_list:
            return "无相关新闻"
        
        news_text = "\n".join([f"{i+1}. {news}" for i, news in enumerate(news_list[:10])])
        
        system_prompt = """你是一位专业的财经新闻分析师。请快速总结以下股票新闻的要点：

要求：
1. 用简洁的语言总结核心信息（100 字以内）
2. 剔除"行业研报"和"普涨简讯"等无关信息
3. 如果存在明显负面信息，请明确指出
4. 只输出摘要，不要其他内容"""

        user_prompt = f"请总结以下新闻：\n\n{news_text}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        summary, tokens = self.call(messages)
        return summary.strip()


class DeepSeekAgent(AIAgentBase):
    """
    DeepSeek AI Agent（用于核心精审）
    
    使用 OpenAI SDK 兼容模式调用 DeepSeek API
    关键配置:
    - base_url: https://api.deepseek.com
    - model: deepseek-chat
    - api_key: 从环境变量 DEEPSEEK_API_KEY 读取
    
    注意：此模块的配置被原封不动保留，仅将底层请求封装进 OpenAI SDK
    """
    
    # 审计关键词
    NEGATIVE_KEYWORDS = [
        "立案调查", "违规担保", "财务造假", "面值退市", 
        "实控人变更", "信披违规", "被证监会", "行政处罚",
        "重大违法", "强制退市", "资金占用", "关联交易"
    ]
    
    # "小作文"识别关键词
    RUMOR_KEYWORDS = [
        "传闻", "消息称", "据悉", "网传", "社交媒体",
        "高送转", "送转", "蹭热点", "蹭概念"
    ]
    
    def __init__(self, config: dict):
        super().__init__(config)
        from openai import OpenAI
        
        # DeepSeek 专用配置 - 保持原有配置不变
        # 如果配置中没有指定 base_url，使用默认的 DeepSeek API 地址
        deepseek_base_url = self.base_url
        if not deepseek_base_url:
            deepseek_base_url = "https://api.deepseek.com"
        
        # 如果配置中没有指定 model，使用 deepseek-chat
        deepseek_model = self.model
        if deepseek_model == 'default' or not deepseek_model:
            deepseek_model = "deepseek-chat"
        
        # 重试配置
        self.retry_config = config.get('retry', {
            'max_attempts': 3,
            'base_delay': 1.0,
            'max_delay': 30.0
        })
        
        # 使用 OpenAI SDK 初始化
        self.client = OpenAI(
            api_key=self.api_key,
            base_url=deepseek_base_url
        )
        self.model = deepseek_model
    
    def _exponential_backoff_retry(self, func, *args, **kwargs):
        """指数退避重试机制"""
        max_attempts = self.retry_config.get('max_attempts', 3)
        base_delay = self.retry_config.get('base_delay', 1.0)
        max_delay = self.retry_config.get('max_delay', 30.0)
        
        last_exception = None
        
        for attempt in range(1, max_attempts + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < max_attempts:
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    logger.warning(f"DeepSeek API call failed (attempt {attempt}), retrying in {delay:.1f}s...")
                    time.sleep(delay)
                else:
                    logger.error(f"DeepSeek API call failed after {max_attempts} attempts: {e}")
        
        raise last_exception
    
    def call(self, messages: list[dict]) -> tuple[str, int]:
        """
        调用 DeepSeek API（使用 OpenAI SDK，带重试）
        
        Args:
            messages: 消息列表
            
        Returns:
            (response_text, token_usage)
        """
        def _call_internal():
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                max_tokens=self.max_tokens
            )
            
            content = completion.choices[0].message.content
            usage = completion.usage.total_tokens if completion.usage else 0
            
            return content, usage
        
        return self._exponential_backoff_retry(_call_internal)
    
    def audit_stock(self, stock_info: dict, news_summary: str) -> AuditResult:
        """
        DeepSeek 核心精审 - 严格风控审计。
        
        审计指令包含：
        1. 关键词扫描：排查"立案调查、违规担保、财务造假..."
        2. "小作文"识别：对比官方公告与社交媒体传闻
        3. 利好陷阱：识别是否为"掩护大股东减持"的虚假利好
        
        Args:
            stock_info: 股票信息（包含代码、名称等）
            news_summary: 初筛后的新闻摘要
            
        Returns:
            审计结果
        """
        symbol = stock_info.get('symbol', '')
        name = stock_info.get('name', '')
        
        system_prompt = """你是一位专业的股票风控审计员，负责识别潜在风险。请严格审计以下股票信息：

【审计指令】
1. 关键词扫描：严格排查以下负面关键词
   - 立案调查、违规担保、财务造假、面值退市
   - 实控人变更、信披违规、被证监会处罚
   - 重大违法强制退市、资金占用、关联交易

2. "小作文"识别：
   - 对比官方公告与社交媒体传闻
   - 官方公告权重最高
   - 对模棱两可的利好（如"高送转"数字游戏）保持高度警惕

3. 利好陷阱识别：
   - 识别是否为"掩护大股东减持"的虚假利好
   - 警惕"蹭热点"式公告

【判定原则】
- 宁缺毋滥：有任何重大风险直接 REJECT
- PASS 标准：无明显负面 + 无"小作文"嫌疑 + 非虚假利好

【输出格式】
请严格按以下 JSON 格式输出：
{
    "status": "PASS" 或 "REJECT",
    "reason": "简短说明原因（50 字以内）",
    "risk_level": "无/低/中/高",
    "keywords_found": ["发现的关键词列表，若无则为空"]
}

只输出 JSON，不要其他内容。"""

        user_prompt = f"""【股票信息】
代码：{symbol}
名称：{name}

【初筛新闻摘要】
{news_summary}

请进行核心精审。"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        try:
            response, tokens = self.call(messages)
            
            # 解析 JSON 响应
            try:
                # 尝试提取 JSON
                json_start = response.find('{')
                json_end = response.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response[json_start:json_end]
                    result = json.loads(json_str)
                else:
                    result = json.loads(response)
                
                status = AuditStatus.PASS if result.get('status') == 'PASS' else AuditStatus.REJECT
                reason = result.get('reason', '未知原因')
                risk_level = result.get('risk_level', '未知')
                keywords = result.get('keywords_found', [])
                
                # 构建详细摘要
                audit_summary = f"{reason} (风险等级：{risk_level})"
                if keywords:
                    audit_summary += f" [关键词：{', '.join(keywords)}]"
                
                return AuditResult(
                    symbol=symbol,
                    status=status,
                    reason=audit_summary,
                    news_summary=news_summary,
                    token_usage=tokens
                )
                
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse DeepSeek JSON response: {response[:200]}")
                # 如果无法解析 JSON，默认驳回
                return AuditResult(
                    symbol=symbol,
                    status=AuditStatus.REJECT,
                    reason=f"AI 响应格式异常：{response[:50]}...",
                    news_summary=news_summary,
                    token_usage=tokens
                )
                
        except Exception as e:
            logger.error(f"DeepSeek audit failed for {symbol}: {e}")
            return AuditResult(
                symbol=symbol,
                status=AuditStatus.ERROR,
                reason=f"审计失败：{e}",
                news_summary=news_summary,
                token_usage=0
            )


# ===========================================
# Main Advisor Class
# ===========================================

class DailyTradeAdvisor:
    """
    每日交易顾问 - 核心决策系统
    
    集成三层过滤系统：
    1. 量化门槛与 Top 10 提取
    2. 混合 AI 分级审计
    3. 大盘环境择时
    """
    
    def __init__(self, config_path: str = "config/settings.yaml"):
        """
        初始化交易顾问。
        
        Args:
            config_path: 配置文件路径
        """
        self.config = ConfigManager(config_path)
        self.db = DatabaseManager.get_instance()
        
        # 交易参数
        self.capital = self.config.get('trading.capital', 50000)
        self.max_positions = self.config.get('trading.max_positions', 3)
        self.min_prob = self.config.get('trading.min_prob', 0.70)
        self.regime_ma = self.config.get('trading.regime_ma', 20)
        
        # 交易成本
        self.commission_rate = self.config.get('trading.commission_rate', 0.0003)
        self.commission_min = self.config.get('trading.commission_min', 5.0)
        self.stamp_duty_rate = self.config.get('trading.stamp_duty_rate', 0.001)
        
        # 标的
        self.index_code = self.config.get('assets.index_code', '000905.SH')
        self.bond_etf = self.config.get('assets.bond_etf', '511010')
        
        # 模型配置
        self.model_path = self.config.get('model.path', 'data/models/stock_model.txt')
        self.feature_columns = self.config.get('model.feature_columns', [])
        
        # AI Agents
        self._init_ai_agents()
        
        # 报告上下文
        self.report_ctx = ReportContext(
            trade_date=datetime.now().strftime('%Y-%m-%d'),
            market_mode=MarketMode.NORMAL,
            regime_ma_value=None,
            current_price=None,
            total_capital=self.capital
        )
        
        logger.info("DailyTradeAdvisor initialized")
    
    def _init_ai_agents(self) -> None:
        """初始化 AI Agents"""
        # 检查是否启用 Mock 模式
        mock_mode = os.getenv('AI_MOCK_MODE', 'false').lower() == 'true'
        
        # 初筛 Agent（Qwen）
        pre_filter_config = self.config.get('ai_agents.pre_filter', {})
        if pre_filter_config.get('enabled', False):
            api_key = pre_filter_config.get('api_key', '')
            # 如果 API 密钥为空或是占位符，使用 Mock
            if mock_mode or not api_key or api_key.startswith('${') or api_key == 'YOUR_QWEN_KEY':
                logger.info("Using Mock Qwen Agent (no valid API key)")
                self.pre_filter_agent = MockQwenAgent(pre_filter_config)
            else:
                logger.info("Using real Qwen Agent with OpenAI SDK")
                self.pre_filter_agent = QwenAgent(pre_filter_config)
        else:
            self.pre_filter_agent = None
        
        # 精审 Agent（DeepSeek）
        deep_audit_config = self.config.get('ai_agents.deep_audit', {})
        if deep_audit_config.get('enabled', False):
            api_key = deep_audit_config.get('api_key', '')
            # 如果 API 密钥为空或是占位符，使用 Mock
            if mock_mode or not api_key or api_key.startswith('${') or api_key == 'YOUR_DEEPSEEK_KEY':
                logger.info("Using Mock DeepSeek Agent (no valid API key)")
                self.deep_audit_agent = MockDeepSeekAgent(deep_audit_config)
            else:
                logger.info("Using real DeepSeek Agent with OpenAI SDK")
                self.deep_audit_agent = DeepSeekAgent(deep_audit_config)
        else:
            self.deep_audit_agent = None
    
    def get_latest_trade_date(self) -> str:
        """获取最新交易日"""
        query = """
            SELECT MAX(trade_date) as latest_date 
            FROM stock_daily
        """
        result = self.db.read_sql(query)
        if result.is_empty():
            logger.warning("Database is empty, using current date as fallback")
            # 数据库为空时，使用当前日期作为 fallback
            return datetime.now().strftime('%Y-%m-%d')
        
        latest_date = result["latest_date"][0]
        # 处理 None 值
        if latest_date is None or str(latest_date) == 'None':
            logger.warning("Latest trade date is None, using current date as fallback")
            return datetime.now().strftime('%Y-%m-%d')
        # 处理日期类型
        if isinstance(latest_date, datetime):
            latest_date = latest_date.strftime('%Y-%m-%d')
        elif hasattr(latest_date, 'strftime'):
            latest_date = latest_date.strftime('%Y-%m-%d')
        return str(latest_date)
    
    def load_and_predict(self, trade_date: str, test_mode: bool = False) -> pl.DataFrame:
        """
        加载数据并执行模型预测。
        
        Args:
            trade_date: 交易日
            test_mode: 是否启用测试模式（强制选取 2 只固定股票，赋予 P=0.85 概率）
            
        Returns:
            预测结果 DataFrame
            
        【鲁棒性优化】:
            - 当某只股票历史数据不足时，自动跳过该股而非中断
            - 因子计算失败时记录警告并继续处理其他股票
        """
        logger.info("=" * 50)
        logger.info("Layer 1: 量化门槛与 Top 10 提取")
        logger.info("=" * 50)
        
        # ========== 测试模式 (Test Mode) ==========
        # 用于 AI 审计全链路测试，强制选取 2 只固定股票并赋予高概率
        if test_mode:
            logger.info("TEST MODE: Forcing 2 fixed stocks with P=0.85 for AI audit testing")
            return self._create_test_mode_candidates(trade_date)
        
        # 读取因子配置
        factor_engine = FactorEngine("config/factors.yaml")
        
        # 计算回溯窗口 - 增加窗口大小以容纳更多数据
        lookback_days = 90  # 增加到 90 天，确保因子计算有足够数据
        start_date = (datetime.strptime(trade_date, '%Y-%m-%d') - timedelta(days=lookback_days)).strftime('%Y-%m-%d')
        
        # 加载数据 - 使用 symbol 字段，不再使用 ts_code
        query = f"""
            SELECT * FROM stock_daily
            WHERE trade_date >= '{start_date}'
            AND trade_date <= '{trade_date}'
            ORDER BY symbol, trade_date
        """
        
        try:
            df = self.db.read_sql(query)
        except Exception as db_error:
            logger.error(f"Database query failed: {db_error}")
            return pl.DataFrame()
        
        if df.is_empty():
            logger.warning("No data loaded for prediction")
            return pl.DataFrame()
        
        logger.info(f"Loaded {len(df)} rows of data from {df['symbol'].n_unique()} stocks")
        
        # ========== 鲁棒性优化：检查每只股票的数据天数 ==========
        # 过滤掉历史数据不足的股票（因子计算需要至少 20 天数据）
        min_history_days = 25  # 至少需要 20 天数据计算因子，加上缓冲
        stock_counts = df.group_by("symbol").agg(pl.len().alias("row_count"))
        valid_stocks = stock_counts.filter(pl.col("row_count") >= min_history_days)["symbol"].to_list()
        dropped_stocks = stock_counts.filter(pl.col("row_count") < min_history_days)
        
        if not dropped_stocks.is_empty():
            for row in dropped_stocks.iter_rows():
                logger.warning(f"⚠️ 跳过 {row[0]}: 只有 {row[1]} 天数据（需要至少 {min_history_days} 天）")
        
        if not valid_stocks:
            logger.error("No stocks have sufficient historical data for factor calculation")
            return pl.DataFrame()
        
        # 过滤出有效股票数据
        df = df.filter(pl.col("symbol").is_in(valid_stocks))
        logger.info(f"After filtering: {len(df)} rows from {len(valid_stocks)} stocks")
        
        # 准备数据
        try:
            df = self._prepare_data_for_factors(df)
        except Exception as e:
            logger.error(f"Failed to prepare data for factors: {e}")
            return pl.DataFrame()
        
        # 计算因子 - 包裹在 try-except 中
        try:
            df_with_factors = factor_engine.compute_factors(df)
        except Exception as factor_error:
            logger.error(f"Factor engine failed: {factor_error}")
            # 降级：返回原始数据，预测时跳过
            df_with_factors = df
        
        # 获取最新一天数据 - 直接使用字符串比较
        latest_df = df_with_factors.filter(pl.col("trade_date").cast(pl.Utf8) == trade_date)
        
        if latest_df.is_empty():
            logger.warning(f"No data for trade_date={trade_date}")
            return pl.DataFrame()
        
        logger.info(f"Latest day data: {len(latest_df)} stocks")
        
        # ========== 加载模型并预测 ==========
        try:
            import lightgbm as lgb
            
            model_path = Path(self.model_path)
            if not model_path.exists():
                # ========== 增强模型路径检查提示 ==========
                logger.warning(f"⚠️ 模型文件缺失：{self.model_path}")
                logger.warning("请先运行模型训练：python src/model_trainer.py")
                logger.warning("当前将使用随机预测值作为占位")
                # 无模型时使用预测值占位
                latest_df = latest_df.with_columns(
                    pl.lit(0.5).alias("predict_prob")
                )
            else:
                model = lgb.Booster(model_file=str(model_path))
                
                # 准备预测数据
                feature_cols = self.feature_columns if self.feature_columns else list(set(df_with_factors.columns) & set(model.feature_name()))
                
                # 确保特征列存在
                available_cols = [c for c in feature_cols if c in latest_df.columns]
                
                if available_cols:
                    # 删除有空值的行
                    valid_mask = ~latest_df.select(pl.any_horizontal(pl.col(available_cols).is_null()))
                    latest_df_valid = latest_df.filter(valid_mask)
                    
                    if not latest_df_valid.is_empty():
                        X_pred = latest_df_valid.select(available_cols).to_numpy()
                        predictions = model.predict(X_pred)
                        
                        # 将预测值转换为概率（假设模型输出是对数几率）
                        # 如果是分类模型，直接使用预测概率
                        predict_probs = 1 / (1 + np.exp(-predictions))  # Sigmoid 转换
                        
                        # 创建预测结果 DataFrame
                        pred_df = latest_df_valid.with_columns(
                            pl.Series("predict_prob", predict_probs)
                        )
                        
                        # 合并回原 DataFrame
                        latest_df = latest_df.join(
                            pred_df.select(["symbol", "predict_prob"]),
                            on="symbol",
                            how="left"
                        )
                    else:
                        latest_df = latest_df.with_columns(
                            pl.lit(0.5).alias("predict_prob")
                        )
                else:
                    latest_df = latest_df.with_columns(
                        pl.lit(0.5).alias("predict_prob")
                    )
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            latest_df = latest_df.with_columns(
                pl.lit(0.5).alias("predict_prob")
            )
        
        # 确保 predict_prob 列存在
        if "predict_prob" not in latest_df.columns:
            latest_df = latest_df.with_columns(
                pl.lit(0.5).alias("predict_prob")
            )
        
        # 应用准入门槛：P >= 70%
        qualified = latest_df.filter(pl.col("predict_prob") >= self.min_prob)
        
        logger.info(f"Qualified stocks (P >= {self.min_prob:.0%}): {len(qualified)}")
        
        if qualified.is_empty():
            logger.warning("No stocks meet the probability threshold, triggering defensive mode")
            return pl.DataFrame()
        
        # 按预测概率排序，取 Top 10
        top_10 = qualified.sort("predict_prob", descending=True).head(10)
        
        logger.info(f"Top 10 candidates selected for AI audit")
        
        return top_10
    
    def _create_test_mode_candidates(self, trade_date: str) -> pl.DataFrame:
        """
        创建测试模式候选股票（用于 AI 审计全链路测试）。
        
        强制选取 2 只固定股票：
        - 贵州茅台 (600519.SH)
        - 宁德时代 (300750.SZ)
        
        并赋予它们 P = 0.85 的概率。
        
        Args:
            trade_date: 交易日
            
        Returns:
            包含测试候选股票的 DataFrame
        """
        logger.info("Creating test mode candidates...")
        
        # 测试股票列表
        test_stocks = [
            {"symbol": "600519.SH", "name": "贵州茅台"},
            {"symbol": "300750.SZ", "name": "宁德时代"},
        ]
        
        # 尝试从数据库获取这些股票的最新收盘价
        try:
            # 注意：stock_daily 表没有 name 字段，直接从代码获取
            query = f"""
                SELECT symbol, close 
                FROM stock_daily 
                WHERE trade_date = '{trade_date}'
                AND symbol IN ('600519.SH', '300750.SZ')
            """
            df = self.db.read_sql(query)
            
            candidates_data = []
            for stock in test_stocks:
                symbol = stock["symbol"]
                name = stock["name"]
                
                # 尝试从数据库获取价格
                row = df.filter(pl.col("symbol") == symbol)
                if not row.is_empty():
                    close = float(row["close"][0])
                else:
                    # 如果数据库中没有，使用默认价格
                    close = 1700.0 if "茅台" in name else 200.0
                
                candidates_data.append({
                    "symbol": symbol,
                    "name": name,
                    "close": close,
                    "predict_prob": 0.85,  # 强制赋予 85% 概率
                })
            
            # 创建 DataFrame
            result_df = pl.DataFrame(candidates_data)
            logger.info(f"Test mode candidates created: {candidates_data}")
            return result_df
            
        except Exception as e:
            logger.warning(f"Failed to get stock prices from DB, using defaults: {e}")
            # 使用默认数据
            candidates_data = [
                {"symbol": "600519.SH", "name": "贵州茅台", "close": 1700.0, "predict_prob": 0.85},
                {"symbol": "300750.SZ", "name": "宁德时代", "close": 200.0, "predict_prob": 0.85},
            ]
            return pl.DataFrame(candidates_data)
    
    def _prepare_data_for_factors(self, df: pl.DataFrame) -> pl.DataFrame:
        """准备数据用于因子计算"""
        # 确保必要的列
        required_columns = ["symbol", "trade_date", "open", "high", "low", "close", "volume"]
        
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # 转换数值列为 Float64
        numeric_columns = ["open", "high", "low", "close", "volume", "amount", "adj_factor"]
        for col in numeric_columns:
            if col in df.columns:
                df = df.with_columns(
                    pl.col(col).cast(pl.Float64, strict=False)
                )
        
        # 计算 pct_change
        if "pct_change" not in df.columns:
            df = df.with_columns(
                (pl.col("close") / pl.col("close").shift(1) - 1)
                .over("symbol")
                .alias("pct_change")
            )
        
        # 排序
        df = df.sort(["symbol", "trade_date"])
        
        return df
    
    def fetch_stock_news(self, symbol: str, name: str) -> list[str]:
        """
        获取股票新闻。
        
        Args:
            symbol: 股票代码
            name: 股票名称
            
        Returns:
            新闻列表
        """
        try:
            # 使用 akshare 获取新闻
            news_df = ak.stock_news_em(symbol=symbol)
            
            if news_df is None or len(news_df) == 0:
                return []
            
            # 获取最新 N 条新闻
            lookback = self.config.get('data.news_lookback_days', 3)
            max_news = self.config.get('data.max_news_per_stock', 10)
            
            cutoff_date = (datetime.now() - timedelta(days=lookback)).strftime('%Y-%m-%d')
            
            news_list = []
            for _, row in news_df.head(max_news).iterrows():
                news_date = str(row.get('publish_date', ''))
                title = str(row.get('title', ''))
                content = str(row.get('content', ''))
                
                # 过滤日期
                if news_date >= cutoff_date:
                    # 修复：使用原始字符串避免\u 转义问题
                    news_item = f"[{news_date}] {title}: {content[:200]}"
                    news_list.append(news_item)
            
            return news_list
            
        except Exception as e:
            # 修复：使用 r 前缀避免正则表达式转义问题
            logger.error(f"Failed to fetch news for {symbol}: {str(e)}")
            return []
    
    def run_ai_audit(self, candidates: list[StockCandidate]) -> list[StockCandidate]:
        """
        执行混合 AI 分级审计。
        
        Args:
            candidates: 候选股票列表
            
        Returns:
            审计后的候选股票列表
        """
        logger.info("=" * 50)
        logger.info("Layer 2: 混合 AI 分级审计")
        logger.info("=" * 50)
        
        total_tokens = 0
        
        for i, candidate in enumerate(candidates):
            logger.info(f"\n[{i+1}/{len(candidates)}] Auditing {candidate.symbol} ({candidate.name})")
            
            # Step A: 获取新闻
            news_list = self.fetch_stock_news(candidate.symbol, candidate.name)
            
            # Step B: 免费模型初筛（如果有配置）
            news_summary = ""
            if self.pre_filter_agent and news_list:
                try:
                    logger.info("  -> Pre-filter: Summarizing news...")
                    news_summary = self.pre_filter_agent.summarize_news(news_list)
                    candidate.news_summary = news_summary
                    logger.info(f"  -> Summary: {news_summary[:100]}...")
                except Exception as e:
                    logger.warning(f"Qwen 初筛失败，跳过初筛直接使用原始新闻：{e}")
                    candidate.news_summary = "\n".join(news_list[:5]) if news_list else "无相关新闻"
            else:
                # 无初筛，直接使用原始新闻
                candidate.news_summary = "\n".join(news_list[:5]) if news_list else "无相关新闻"
            
            # Step C: DeepSeek 核心精审
            if self.deep_audit_agent:
                logger.info("  -> DeepSeek: Core audit...")
                
                stock_info = {
                    'symbol': candidate.symbol,
                    'name': candidate.name
                }
                
                audit_result = self.deep_audit_agent.audit_stock(
                    stock_info=stock_info,
                    news_summary=candidate.news_summary
                )
                
                candidate.audit_status = audit_result.status
                candidate.audit_reason = audit_result.reason
                total_tokens += audit_result.token_usage
                
                logger.info(f"  -> Result: {audit_result.status.value} - {audit_result.reason[:80]}")
            else:
                # 无精审，默认通过
                candidate.audit_status = AuditStatus.PASS
                candidate.audit_reason = "未配置精审，默认通过"
            
            # 记录审计轨迹
            self.report_ctx.audit_trail.append({
                'rank': candidate.rank,
                'symbol': candidate.symbol,
                'name': candidate.name,
                'predict_prob': candidate.predict_prob,
                'status': candidate.audit_status.value,
                'reason': candidate.audit_reason
            })
        
        # 更新 API 统计
        self.report_ctx.api_stats['deepseek_tokens'] = total_tokens
        
        # 过滤出通过的股票
        passed = [c for c in candidates if c.audit_status == AuditStatus.PASS]
        
        logger.info(f"\nAudit complete: {len(passed)}/{len(candidates)} passed")
        
        return passed
    
    def check_market_regime(self, trade_date: str) -> MarketMode:
        """
        检查大盘环境（第三层：仓位控制器）。
        
        计算中证 500 指数 20 日均线：
        - 若价格在均线下，判定为"防守模式"
        
        注意：index_daily 表使用 symbol 字段（新规范）
        
        Args:
            trade_date: 交易日
            
        Returns:
            市场模式
        """
        logger.info("=" * 50)
        logger.info("Layer 3: 大盘环境择时")
        logger.info("=" * 50)
        
        try:
            # 获取中证 500 指数数据
            index_code = self.index_code
            
            # 尝试从数据库查询指数数据
            # 注意：index_daily 表使用 symbol 字段作为主键（新规范）
            try:
                query = f"""
                    SELECT trade_date, close 
                    FROM index_daily 
                    WHERE symbol = '{index_code}'
                    AND trade_date <= '{trade_date}'
                    ORDER BY trade_date DESC
                    LIMIT {self.regime_ma + 10}
                """
                
                index_df = self.db.read_sql(query)
            except Exception as db_error:
                logger.warning(f"Index data not available: {db_error}")
                index_df = pl.DataFrame()
            
            if index_df.is_empty() or len(index_df) < self.regime_ma:
                logger.warning(f"Insufficient index data for {index_code}, using default mode")
                self.report_ctx.market_mode = MarketMode.NORMAL
                self.report_ctx.regime_ma_value = None
                self.report_ctx.current_price = None
                return MarketMode.NORMAL
            
            # 计算 20 日均线
            index_df = index_df.sort("trade_date")
            index_df = index_df.with_columns(
                pl.col("close").rolling_mean(window_size=self.regime_ma).alias("ma20")
            )
            
            latest = index_df.tail(1)
            current_price_raw = latest["close"][0]
            ma20_raw = latest["ma20"][0]
            
            # ========== 修复：Decimal 类型转换为 float ==========
            # 数据库返回的 DECIMAL 类型会被 polars 解析为 Decimal，需要转换为 float
            if isinstance(current_price_raw, Decimal):
                current_price = float(current_price_raw)
            elif hasattr(current_price_raw, '__float__'):
                current_price = float(current_price_raw)
            else:
                current_price = float(current_price_raw)
            
            if isinstance(ma20_raw, Decimal):
                ma20 = float(ma20_raw)
            elif hasattr(ma20_raw, '__float__'):
                ma20 = float(ma20_raw)
            else:
                ma20 = float(ma20_raw)
            
            self.report_ctx.current_price = current_price
            self.report_ctx.regime_ma_value = ma20
            
            # ========== 增强日志：输出价格与均线的数值对比 ==========
            price_vs_ma = current_price - ma20
            price_vs_ma_pct = (current_price / ma20 - 1) * 100
            
            if current_price < ma20:
                self.report_ctx.market_mode = MarketMode.DEFENSIVE
                logger.info(f"📉 DEFENSIVE MODE: 中证 500 价格 ({current_price:.2f}) < 20 日均线 ({ma20:.2f})")
                logger.info(f"   价差：{price_vs_ma:.2f} ({price_vs_ma_pct:+.2f}%) - 低于均线，建议防守")
            else:
                self.report_ctx.market_mode = MarketMode.NORMAL
                logger.info(f"📈 NORMAL MODE: 中证 500 价格 ({current_price:.2f}) >= 20 日均线 ({ma20:.2f})")
                logger.info(f"   价差：{price_vs_ma:.2f} ({price_vs_ma_pct:+.2f}%) - 高于均线，可以进攻")
            
            return self.report_ctx.market_mode
            
        except Exception as e:
            logger.error(f"Failed to check market regime: {e}")
            self.report_ctx.market_mode = MarketMode.NORMAL
            return MarketMode.NORMAL
    
    def calculate_allocation(self, passed_candidates: list[StockCandidate]) -> list[StockCandidate]:
        """
        计算资金分配。
        
        预算分配规则：
        - 总本金 50,000 元
        - 若 PASS 标的为 N 只（N <= 3），每只分配 50000/3 的预算
        - 防守模式下预算减半
        - 碎股取整：100 股整数倍向下取整
        - 预扣除佣金和印花税
        
        Args:
            passed_candidates: 通过的候选股票
            
        Returns:
            包含分配信息的候选股票列表
        """
        logger.info("=" * 50)
        logger.info("Capital Allocation")
        logger.info("=" * 50)
        
        # 确定实际可买入数量（最多 3 只）
        actual_count = min(len(passed_candidates), self.max_positions)
        
        # 确定每只股票的预算
        per_stock_budget = self.capital / self.max_positions
        
        # 防守模式下减半
        if self.report_ctx.market_mode == MarketMode.DEFENSIVE:
            per_stock_budget *= 0.5
            logger.info(f"Defensive mode: budget halved to {per_stock_budget:.2f} per stock")
        
        logger.info(f"Per stock budget: {per_stock_budget:.2f}")
        logger.info(f"Max positions: {actual_count}")
        
        used_capital = 0.0
        affordable_count = 0
        
        for i, candidate in enumerate(passed_candidates[:actual_count]):
            # 计算可买入股数（100 股整数倍向下取整）
            raw_shares = int(per_stock_budget / candidate.close)
            shares = (raw_shares // 100) * 100
            
            # 确保至少 100 股
            if shares < 100:
            # 小额本金买不起高价股的明确提示
                min_price_for_100 = per_stock_budget / 100
                required_amount = candidate.close * 100
                logger.warning(
                    f"💸 {candidate.symbol}: 股价 {candidate.close:.2f} 过高，"
                    f"预算 {per_stock_budget:.2f} 元无法买入 100 股（至少需要 {required_amount:.2f} 元）"
                )
                logger.warning(
                    f"   提示：单股预算 {per_stock_budget:.2f} 元，只能买股价 < {min_price_for_100:.2f} 元的股票"
                )
                candidate.recommended_shares = 0
                continue
            
            affordable_count += 1
            
            # 计算金额
            amount = shares * candidate.close
            
            # 计算佣金（最低 5 元）
            commission = max(amount * self.commission_rate, self.commission_min)
            
            # 计算印花税（卖出时收取，买入时不收取，但预留）
            stamp_duty = amount * self.stamp_duty_rate
            
            # 总成本
            total_cost = amount + commission
            
            candidate.recommended_shares = shares
            candidate.estimated_amount = amount
            candidate.commission = commission
            candidate.stamp_duty = stamp_duty
            
            used_capital += total_cost
            
            logger.info(f"{candidate.symbol}: {shares} shares, {amount:.2f} + {commission:.2f} commission")
        
        # 更新报告上下文
        self.report_ctx.used_capital = used_capital
        self.report_ctx.remaining_capital = self.capital - used_capital
        
        logger.info(f"Capital allocation: used={used_capital:.2f}, remaining={self.report_ctx.remaining_capital:.2f}")
        logger.info(f"Affordable stocks: {affordable_count}/{actual_count}")
        
        # 计算国债 ETF 补位
        self._calculate_bond_etf_allocation()
        
        return passed_candidates[:actual_count]
    
    def _calculate_bond_etf_allocation(self) -> None:
        """
        计算国债 ETF 补位（空仓逻辑兜底）。
        
        兜底策略:
        - 若数据库无 ETF 数据，仅提示"建议买入国债 ETF"，不尝试计算具体股数
        - 避免因数据同步延迟导致程序崩溃
        
        注意：etf_daily 表使用 symbol 字段作为主键
        """
        remaining = self.report_ctx.remaining_capital
        
        # ========== 修复：确保空仓时也计算剩余现金 ==========
        if remaining <= 0:
            self.report_ctx.bond_etf_shares = 0
            self.report_ctx.bond_etf_amount = 0.0
            logger.info(f"No remaining capital for bond ETF allocation (remaining={remaining:.2f})")
            return
        
        try:
            # 获取国债 ETF 最新价格
            bond_etf_code = self.bond_etf
            
            # 注意：etf_daily 表使用 symbol 字段
            query = f"""
                SELECT close FROM etf_daily
                WHERE symbol = '{bond_etf_code}'
                ORDER BY trade_date DESC
                LIMIT 1
            """
            
            etf_df = self.db.read_sql(query)
            
            # ========== 兜底逻辑：数据库无数据时的处理 ==========
            if etf_df.is_empty():
                logger.warning(f"⚠️ Bond ETF data not available for {bond_etf_code}")
                logger.warning("ETF 数据可能尚未同步，建议先运行：python src/sync_etf_data.py")
                # 兜底：仅提示建议买入，不计算具体股数，避免程序崩溃
                self.report_ctx.bond_etf_shares = 0
                self.report_ctx.bond_etf_amount = 0.0
                logger.info(f"💡 建议：使用剩余资金 {remaining:.2f} 元买入国债 ETF ({bond_etf_code}) 避险")
                return
            
            # ========== 修复：正确处理 Decimal 类型转换 ==========
            etf_price_raw = etf_df["close"][0]
            # 处理 Decimal 类型（数据库返回）到 float 的转换
            if isinstance(etf_price_raw, Decimal):
                etf_price = float(etf_price_raw)
            elif hasattr(etf_price_raw, '__float__'):
                etf_price = float(etf_price_raw)
            else:
                etf_price = float(etf_price_raw)
            
            # 计算可买入股数（100 股整数倍）
            raw_shares = int(remaining / etf_price)
            shares = (raw_shares // 100) * 100
            
            if shares >= 100:
                self.report_ctx.bond_etf_shares = shares
                self.report_ctx.bond_etf_amount = shares * etf_price
                logger.info(f"Bond ETF ({bond_etf_code}): {shares} shares, {shares * etf_price:.2f}")
            else:
                self.report_ctx.bond_etf_shares = 0
                self.report_ctx.bond_etf_amount = 0.0
                logger.info(f"Remaining capital ({remaining:.2f}) too low for bond ETF (price: {etf_price:.2f})")
                
        except Exception as e:
            logger.error(f"Failed to calculate bond ETF allocation: {e}")
            # 兜底：出错时不阻塞程序，仅记录错误
            self.report_ctx.bond_etf_shares = 0
            self.report_ctx.bond_etf_amount = 0.0
    
    def generate_report(self, final_candidates: list[StockCandidate]) -> str:
        """
        生成 Markdown 格式报表。
        
        包含：
        1. 明日决策清单
        2. 决策链条全回溯
        3. 风控看板
        4. 风险提示
        
        Args:
            final_candidates: 最终候选股票
            
        Returns:
            Markdown 报表字符串
        """
        logger.info("=" * 50)
        logger.info("Generating Report")
        logger.info("=" * 50)
        
        ctx = self.report_ctx
        
        # ========== 修复：确保资金计算逻辑不被跳过 ==========
        # 如果没有买入建议，确保剩余现金等于总本金
        if not final_candidates or all(c.recommended_shares == 0 for c in final_candidates):
            # 空仓时，剩余现金应等于总本金
            if ctx.used_capital == 0.0:
                ctx.remaining_capital = ctx.total_capital
                logger.info(f"Empty position: remaining_capital set to total_capital ({ctx.total_capital:.2f})")
        
        # 构建报表
        report = []
        
        # 标题 - 根据运行模式动态调整
        if ctx.run_mode == RunMode.DRAFT:
            report.append(f"# 📅 [初稿] 盘后初步交易建议")
        elif ctx.run_mode == RunMode.FINAL:
            report.append(f"# 🏆 [终稿] 盘后决策终审报告")
        else:
            report.append(f"# 📊 明日决策清单")
        
        report.append(f"**交易日期**: {ctx.trade_date}")
        report.append(f"**报告生成时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"**运行模式**: {ctx.run_mode.value.upper()}")
        report.append("")
        
        # 判断是否空仓
        if not final_candidates or all(c.recommended_shares == 0 for c in final_candidates):
            report.append("## 🚫 当前环境不佳，建议全仓避险（国债 ETF）")
            report.append("")
            report.append(f"> **市场模式**: {ctx.market_mode.value.upper()}")
            if ctx.regime_ma_value and ctx.current_price:
                report.append(f"> **中证 500 价格**: {ctx.current_price:.2f} | **20 日均线**: {ctx.regime_ma_value:.2f}")
            report.append("")
        else:
            # 有买入标的
            report.append("## 📈 买入标的")
            report.append("")
            report.append("| 代码 | 名称 | 预测概率 | AI 审计摘要 | 建议股数 | 预估金额 |")
            report.append("|------|------|----------|-----------|----------|----------|")
            
            for candidate in final_candidates:
                if candidate.recommended_shares > 0:
                    audit_summary = candidate.audit_reason[:30] + "..." if len(candidate.audit_reason) > 30 else candidate.audit_reason
                    report.append(
                        f"| {candidate.symbol} | {candidate.name} | {candidate.predict_prob:.1%} | {audit_summary} | {candidate.recommended_shares} | {candidate.estimated_amount:.2f} |"
                    )
            report.append("")
        
        # 国债 ETF 配置
        if ctx.bond_etf_shares > 0:
            report.append("## 🛡️ 避险配置（国债 ETF）")
            report.append("")
            report.append(f"- **代码**: {self.bond_etf}")
            report.append(f"- **建议股数**: {ctx.bond_etf_shares}")
            report.append(f"- **预估金额**: {ctx.bond_etf_amount:.2f}")
            report.append("")
        
        # 决策链条全回溯
        report.append("---")
        report.append("")
        report.append("## 🔍 决策链条全回溯")
        report.append("")
        
        # 概率未达标的股票（如有记录）
        report.append("### 量化层过滤")
        report.append("")
        if ctx.audit_trail:
            rejected_by_quant = [t for t in ctx.audit_trail if t.get('quant_rejected', False)]
            if rejected_by_quant:
                report.append("以下股票因预测概率未达 70% 被剔除：")
                report.append("")
                for t in rejected_by_quant:
                    report.append(f"- {t['symbol']} ({t['name']}): 预测概率 {t['predict_prob']:.1%}")
            else:
                report.append("Top 10 股票均满足概率门槛，进入 AI 审计环节。")
        report.append("")
        
        # AI 审计回溯
        report.append("### AI 审计层")
        report.append("")
        
        if ctx.audit_trail:
            for trail in ctx.audit_trail:
                status_emoji = "✅" if trail['status'] == 'pass' else "❌" if trail['status'] == 'reject' else "⚠️"
                report.append(
                    f"- **No.{trail['rank']} [{trail['symbol']}]** {status_emoji} "
                    f"{trail['status'].upper()}: {trail['reason']}"
                )
        else:
            report.append("无审计记录。")
        
        report.append("")
        
        # 风控看板
        report.append("---")
        report.append("")
        report.append("## 🛡️ 风控看板")
        report.append("")
        report.append(f"| 指标 | 数值 |")
        report.append("|------|------|")
        report.append(f"| **市场模式** | {ctx.market_mode.value.upper()} |")
        
        if ctx.regime_ma_value:
            report.append(f"| **中证 500 价格** | {ctx.current_price:.2f} |")
            report.append(f"| **20 日均线** | {ctx.regime_ma_value:.2f} |")
        
        report.append(f"| **API 调用消耗** | DeepSeek: {ctx.api_stats.get('deepseek_tokens', 0)} tokens |")
        report.append("")
        
        # 风险提示
        report.append("---")
        report.append("")
        report.append("## ⚠️ 风险提示")
        report.append("")
        
        # 计算总换手率（假设全仓买入）
        turnover_rate = ctx.used_capital / ctx.total_capital if ctx.total_capital > 0 else 0
        
        # 计算总成本
        total_commission = sum(c.commission for c in final_candidates if c.recommended_shares > 0)
        total_stamp_duty = sum(c.stamp_duty for c in final_candidates if c.recommended_shares > 0)
        total_cost = total_commission + total_stamp_duty
        
        # ========== 修复：正确显示剩余现金 ==========
        # 剩余现金 = 总本金 - 已用资金 - 国债 ETF 金额
        remaining_cash = ctx.total_capital - ctx.used_capital - ctx.bond_etf_amount
        
        report.append(f"- **预估总换手率**: {turnover_rate:.1%}")
        report.append(f"- **交易总成本**: {total_cost:.2f} 元 (佣金 {total_commission:.2f} + 印花税 {total_stamp_duty:.2f})")
        report.append(f"- **剩余现金**: {remaining_cash:.2f} 元")
        report.append(f"- **已用资金**: {ctx.used_capital:.2f} 元 / {ctx.total_capital:.2f} 元")
        if ctx.bond_etf_amount > 0:
            report.append(f"- **国债 ETF 配置**: {ctx.bond_etf_amount:.2f} 元")
        report.append("")
        
        # 免责声明
        report.append("---")
        report.append("")
        report.append("> ⚠️ **免责声明**: 本报告仅供参考，不构成投资建议。")
        report.append("> 量化模型和 AI 审计均存在局限性，投资需谨慎。")
        report.append("> 过往业绩不代表未来表现，市场有风险，投资需谨慎。")
        
        # ========== DRAFT 模式页脚提示 ==========
        if ctx.run_mode == RunMode.DRAFT:
            report.append("")
            report.append("---")
            report.append("")
            report.append("> 📝 **注**: 本报告为盘后初稿，最终决策请参考 21:00 终稿。")
        
        return "\n".join(report)
    
    def run(self, test_mode: bool = False, report_dir: str = "reports", run_mode: RunMode = RunMode.FINAL) -> str:
        """
        执行完整的决策流程。
        
        Args:
            test_mode: 是否启用测试模式（强制选取 2 只固定股票，赋予 P=0.85 概率）
            report_dir: 报告保存目录
            run_mode: 运行模式（DRAFT 初稿 / FINAL 终稿）
            
        Returns:
            Markdown 报表字符串
        """
        # ========== 自动创建报告目录（包括父目录） ==========
        report_path = Path(report_dir)
        try:
            report_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Report directory ensured: {report_path.absolute()}")
        except Exception as e:
            logger.error(f"Failed to create report directory: {e}")
            raise
        
        logger.info("=" * 60)
        logger.info("DAILY TRADE ADVISOR - Starting Decision Process")
        logger.info(f"Run Mode: {run_mode.value.upper()}")
        if test_mode:
            logger.info("🧪 TEST MODE: AI Audit Full-Chain Testing")
        logger.info("=" * 60)
        
        # 设置运行模式
        self.report_ctx.run_mode = run_mode
        
        try:
            # Step 0: 获取最新交易日
            trade_date = self.get_latest_trade_date()
            self.report_ctx.trade_date = trade_date
            logger.info(f"Trade Date: {trade_date}")
            
            # Step 1: 量化门槛与 Top 10 提取
            top_10_df = self.load_and_predict(trade_date, test_mode=test_mode)
            
            if top_10_df.is_empty():
                logger.warning("No stocks meet the probability threshold")
                # 直接生成空仓报告
                self.report_ctx.market_mode = self.check_market_regime(trade_date)
                self._calculate_bond_etf_allocation()
                return self.generate_report([])
            
            # 转换为候选对象列表
            candidates = []
            for rank, row in enumerate(top_10_df.iter_rows(), 1):
                # 获取股票名称 - 从预定义的股票名称映射获取
                symbol = str(row[0])
                
                # 尝试从预定义的名称映射获取名称
                name = self._get_stock_name(symbol)
                
                # 找到 close 和 predict_prob 列的索引
                close_idx = top_10_df.columns.index('close') if 'close' in top_10_df.columns else -1
                prob_idx = top_10_df.columns.index('predict_prob') if 'predict_prob' in top_10_df.columns else -1
                
                candidate = StockCandidate(
                    symbol=symbol,
                    name=name,
                    close=float(row[close_idx]) if close_idx >= 0 else 0.0,
                    predict_prob=float(row[prob_idx]) if prob_idx >= 0 else 0.5,
                    rank=rank
                )
                candidates.append(candidate)
            
            self.report_ctx.top_10_candidates = candidates
            logger.info(f"Top 10 candidates: {[c.symbol for c in candidates]}")
            
            # Step 2: 混合 AI 分级审计
            passed_candidates = self.run_ai_audit(candidates)
            
            # ========== Test Mode 强化：显示 REJECT 原因 ==========
            if test_mode:
                rejected_candidates = [c for c in candidates if c.audit_status == AuditStatus.REJECT]
                if rejected_candidates:
                    logger.warning("=" * 60)
                    logger.warning("🚨 TEST MODE - REJECTED CANDIDATES:")
                    for rc in rejected_candidates:
                        logger.warning(f"  ❌ **{rc.symbol} ({rc.name})**: {rc.audit_reason}")
                    logger.warning("=" * 60)
            
            if not passed_candidates:
                logger.warning("All candidates rejected by AI audit")
                self.report_ctx.market_mode = self.check_market_regime(trade_date)
                self._calculate_bond_etf_allocation()
                return self.generate_report([])
            
            # Step 3: 大盘环境择时
            self.report_ctx.market_mode = self.check_market_regime(trade_date)
            
            # Step 4: 资金分配计算
            final_candidates = self.calculate_allocation(passed_candidates)
            
            # Step 5: 生成报表
            report = self.generate_report(final_candidates)
            
            logger.info("=" * 60)
            logger.info("Decision Process Complete")
            logger.info("=" * 60)
            
            # ========== 保存报告到文件 ==========
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            
            # 根据运行模式生成文件名
            if run_mode == RunMode.DRAFT:
                report_filename = f"decision_{timestamp}_draft.md"
            elif run_mode == RunMode.FINAL:
                report_filename = f"decision_{timestamp}_final.md"
            elif test_mode:
                report_filename = f"decision_{timestamp}_test.md"
            else:
                report_filename = f"decision_{timestamp}.md"
            
            report_path = Path(report_dir) / report_filename
            try:
                with open(report_path, 'w', encoding='utf-8') as f:
                    f.write(report)
                logger.info(f"✅ Report saved to: {report_path.absolute()}")
            except Exception as e:
                logger.error(f"Failed to save report: {e}")
                raise
            
            return report
            
        except Exception as e:
            logger.error(f"Decision process failed: {e}")
            # 返回错误报告
            return f"""# ⚠️ 系统错误

**错误信息**: {e}

**建议**: 请检查系统配置和数据完整性后重试。
"""
    
    def _get_stock_name(self, symbol: str) -> str:
        """
        获取股票名称（从预定义的映射或数据库）。
        
        Args:
            symbol: 股票代码
            
        Returns:
            股票名称
        """
        # 常见股票名称映射
        stock_names = {
            "600519.SH": "贵州茅台",
            "300750.SZ": "宁德时代",
            "000858.SZ": "五粮液",
            "601318.SH": "中国平安",
            "600036.SH": "招商银行",
            "000333.SZ": "美的集团",
            "002415.SZ": "海康威视",
            "601888.SH": "中国中免",
            "600276.SH": "恒瑞医药",
            "601166.SH": "兴业银行",
        }
        
        # 先查映射表
        if symbol in stock_names:
            return stock_names[symbol]
        
        # 尝试从数据库获取
        try:
            # 注意：stock_daily 表没有 name 字段，需要其他方式获取
            # 这里返回默认名称
            return f"股票_{symbol}"
        except Exception:
            return f"股票_{symbol}"


# ===========================================
# Entry Point
# ===========================================

def main(test_mode: bool = False, run_mode: RunMode = RunMode.FINAL):
    """
    主函数
    
    Args:
        test_mode: 是否启用测试模式
        run_mode: 运行模式（DRAFT 初稿 / FINAL 终稿）
    """
    # 配置日志
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 可选：添加文件日志
    log_file = "logs/trade_advisor.log"
    Path("logs").mkdir(parents=True, exist_ok=True)
    logger.add(
        log_file,
        rotation="1 day",
        retention="7 days",
        level="DEBUG"
    )
    
    # 创建顾问并运行
    advisor = DailyTradeAdvisor("config/settings.yaml")
    
    # 确定报告保存目录
    report_dir = "reports" if test_mode else "docs"
    report = advisor.run(test_mode=test_mode, report_dir=report_dir, run_mode=run_mode)
    
    # 输出报表
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)
    
    # 保存到文件
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime('%Y%m%d')
    
    # 根据运行模式生成文件名
    if run_mode == RunMode.DRAFT:
        report_file = f"{report_dir}/decision_{date_str}_draft.md"
    elif run_mode == RunMode.FINAL:
        report_file = f"{report_dir}/decision_{date_str}_final.md"
    elif test_mode:
        report_file = f"{report_dir}/decision_{date_str}_test_mode.md"
    else:
        report_file = f"{report_dir}/daily_decision_{date_str}.md"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    logger.info(f"Report saved to: {report_file}")


def run_with_args():
    """
    命令行入口函数 - 支持通过命令行参数运行
    
    使用示例:
        python src/daily_trade_advisor.py                    # 默认 FINAL 模式
        python src/daily_trade_advisor.py --mode draft       # DRAFT 初稿模式
        python src/daily_trade_advisor.py --mode final       # FINAL 终稿模式
        python src/daily_trade_advisor.py --test             # 测试模式
        python src/daily_trade_advisor.py --test --mode draft  # 测试模式 + DRAFT
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Daily Trade Advisor - 每日交易决策系统')
    parser.add_argument('--mode', type=str, default='final', choices=['draft', 'final'],
                        help='运行模式：draft (初稿) 或 final (终稿)')
    parser.add_argument('--test', action='store_true', help='启用测试模式')
    
    args = parser.parse_args()
    
    # 确定运行模式
    run_mode = RunMode.DRAFT if args.mode == 'draft' else RunMode.FINAL
    
    # 运行
    main(test_mode=args.test, run_mode=run_mode)


if __name__ == "__main__":
    run_with_args()
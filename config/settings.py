# config/settings.py
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()


# 验证必需的配置
MYSQL_CONFIG = {
    'host': os.getenv('MYSQL_HOST'),
    'port': int(os.getenv('MYSQL_PORT', '3306')),
    'user': os.getenv('MYSQL_USER'),
    'password': os.getenv('MYSQL_PASSWORD'),
    'database': os.getenv('MYSQL_DATABASE'),
    'charset': os.getenv('MYSQL_CHARSET', 'utf8mb4'),
}

missing = [key for key, value in MYSQL_CONFIG.items() if value is None]
if missing:
    raise ValueError(f"缺少必需的配置项: {missing}. 请检查 .env 文件")

# 回测配置
BACKTEST_CONFIG = {
    'default_symbol': '510300.SH',
    'initial_capital': 100000,
    'commission_rate': 0.0003,  # 万分之三
    'slippage_rate': 0.0001,    # 万分之一
    'test_periods': {
        'short': ('2023-01-01', '2024-01-01'),
        'medium': ('2021-01-01', '2024-01-01'),
        'long': ('2018-01-01', '2024-01-01')
    }
}

# 策略参数
STRATEGY_PARAMS = {
    'dual_ma': {
        'fast_period': 10,
        'slow_period': 30
    }
}
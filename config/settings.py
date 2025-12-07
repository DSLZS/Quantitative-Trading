# config/settings.py
import os
from dotenv import load_dotenv

# 加载.env文件
load_dotenv()

# 必需的配置 - 没有默认值，确保必须配置
MYSQL_HOST = os.getenv('MYSQL_HOST')
MYSQL_USER = os.getenv('MYSQL_USER')
MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
MYSQL_DATABASE = os.getenv('MYSQL_DATABASE')
MYSQL_PORT = int(os.getenv('MYSQL_PORT', '3306'))

# 验证必需的配置
required_configs = {
    'MYSQL_HOST': MYSQL_HOST,
    'MYSQL_USER': MYSQL_USER,
    'MYSQL_PASSWORD': MYSQL_PASSWORD,
    'MYSQL_DATABASE': MYSQL_DATABASE,
}

missing = [key for key, value in required_configs.items() if value is None]
if missing:
    raise ValueError(f"缺少必需的配置项: {missing}. 请检查 .env 文件")
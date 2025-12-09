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
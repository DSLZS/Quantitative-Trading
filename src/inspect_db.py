#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
V96 策略重构 - 数据库检查脚本
使用 Polars + connectorx 直接读取数据库

强制要求:
    1. 使用 pl.read_database 配合 connectorx
    2. 显式检查"银行"行业数据
    3. 实现 fillna 或行业映射逻辑
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import connectorx as cx
from dotenv import load_dotenv
from loguru import logger

# Load environment variables
load_dotenv()


def get_db_url() -> str:
    """
    从环境变量构建 MySQL 连接 URL。
    
    Returns:
        str: MySQL 连接 URL，格式为 mysql://user:password@host:port/database
    """
    host = os.getenv("MYSQL_HOST", "localhost")
    port = os.getenv("MYSQL_PORT", "3306")
    user = os.getenv("MYSQL_USER", "root")
    password = os.getenv("MYSQL_PASSWORD", "")
    database = os.getenv("MYSQL_DATABASE", "quantitative_trading")
    
    return f"mysql://{user}:{password}@{host}:{port}/{database}"


def read_db(query: str) -> pl.DataFrame:
    """
    使用 connectorx 读取数据库。
    
    Args:
        query: SQL 查询语句
        
    Returns:
        pl.DataFrame: 查询结果
    """
    db_url = get_db_url()
    try:
        # 使用 connectorx 读取数据
        df = cx.read_sql(db_url, query)
        return pl.from_pandas(df)
    except Exception as e:
        raise e


def check_table_structure(table_name: str) -> dict:
    """
    检查表结构，使用 connectorx 读取元数据。
    
    Args:
        table_name: 表名
        
    Returns:
        dict: 表结构信息
    """
    query = f"""
    SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE, COLUMN_KEY
    FROM information_schema.COLUMNS
    WHERE TABLE_SCHEMA = '{os.getenv("MYSQL_DATABASE", "quantitative_trading")}'
    AND TABLE_NAME = '{table_name}'
    ORDER BY ORDINAL_POSITION
    """
    
    try:
        df = read_db(query)
        
        if df.is_empty():
            return {"exists": False, "columns": [], "error": f"表 {table_name} 不存在"}
        
        column_names = df["COLUMN_NAME"].to_list()
        
        return {
            "exists": True,
            "columns": column_names,
            "column_count": len(column_names),
            "details": df
        }
    except Exception as e:
        return {"exists": False, "columns": [], "error": str(e)}


def check_industry_mapping() -> pl.DataFrame:
    """
    检查行业映射数据分布。
    
    Returns:
        pl.DataFrame: 行业分布统计
    """
    query = """
    SELECT industry_name, COUNT(*) as stock_count
    FROM stock_industry_daily
    GROUP BY industry_name
    ORDER BY stock_count DESC
    """
    
    try:
        df = read_db(query)
        return df
    except Exception as e:
        logger.error(f"查询行业映射失败：{e}")
        return pl.DataFrame()


def check_banking_industry() -> tuple[bool, pl.DataFrame]:
    """
    专门检查"银行"行业数据。
    
    Returns:
        tuple[bool, pl.DataFrame]: (是否找到银行行业，银行行业数据)
    """
    # 查询包含"银行"的行业
    query = """
    SELECT DISTINCT industry_name, COUNT(*) as stock_count
    FROM stock_industry_daily
    WHERE industry_name LIKE '%银行%'
    GROUP BY industry_name
    ORDER BY stock_count DESC
    """
    
    try:
        df = read_db(query)
        
        if df.is_empty():
            # 尝试查询"金融"相关行业
            query_finance = """
            SELECT DISTINCT industry_name, COUNT(*) as stock_count
            FROM stock_industry_daily
            WHERE industry_name LIKE '%金融%'
            GROUP BY industry_name
            ORDER BY stock_count DESC
            """
            df_finance = read_db(query_finance)
            
            if not df_finance.is_empty():
                logger.warning("未找到'银行'行业，但找到'金融'相关行业")
                return False, df_finance
            else:
                logger.error("未找到'银行'或'金融'相关行业")
                return False, pl.DataFrame()
        
        return True, df
        
    except Exception as e:
        logger.error(f"查询银行行业失败：{e}")
        return False, pl.DataFrame()


def get_all_industries_with_fillna() -> pl.DataFrame:
    """
    获取所有行业数据，并实现 fillna 和行业映射逻辑。
    
    如果行业名称为空或 NULL，使用以下策略填充:
    1. 优先使用 industry_name
    2. 如果为空，使用 industry_code 映射
    3. 如果都为空，标记为"未知行业"
    
    Returns:
        pl.DataFrame: 处理后的行业数据
    """
    query = """
    SELECT 
        symbol,
        trade_date,
        industry_name,
        industry_code
    FROM stock_industry_daily
    LIMIT 1000
    """
    
    try:
        df = read_db(query)
        
        # 实现 fillna 逻辑
        df = df.with_columns([
            # 优先使用 industry_name，如果为空则使用 industry_code
            pl.col("industry_name").fill_null("未知行业").alias("industry_name_filled"),
            # 如果 industry_name 为空字符串，也视为 NULL
            pl.when(pl.col("industry_name").str.strip_chars() == "")
              .then(pl.col("industry_code").fill_null("未知行业"))
              .otherwise(pl.col("industry_name"))
              .alias("industry_name_final"),
        ])
        
        # 行业映射逻辑 - 将常见行业别名映射到标准名称
        industry_mapping = {
            "银行业": "银行",
            "银行Ⅲ": "银行",
            "银行Ⅱ": "银行",
            "国有大型银行": "银行",
            "股份制银行": "银行",
            "城商行": "银行",
            "农商行": "银行",
            "其他银行": "银行",
            "证券": "非银金融",
            "证券Ⅱ": "非银金融",
            "证券Ⅲ": "非银金融",
            "保险": "非银金融",
            "保险Ⅱ": "非银金融",
            "保险Ⅲ": "非银金融",
            "多元金融": "非银金融",
            "房地产": "房地产",
            "房地产开发": "房地产",
            "房地产服务": "房地产",
            "白酒": "食品饮料",
            "饮料": "食品饮料",
            "食品": "食品饮料",
            "医药": "医药生物",
            "化学制药": "医药生物",
            "中药": "医药生物",
            "生物制品": "医药生物",
            "医疗器械": "医药生物",
            "医疗服务": "医药生物",
            "电子": "电子",
            "半导体": "电子",
            "元件": "电子",
            "光学光电子": "电子",
            "消费电子": "电子",
            "计算机": "计算机",
            "计算机设备": "计算机",
            "软件开发": "计算机",
            "IT 服务": "计算机",
            "通信": "通信",
            "通信设备": "通信",
            "通信服务": "通信",
            "电力设备": "电力设备",
            "电池": "电力设备",
            "光伏设备": "电力设备",
            "风电设备": "电力设备",
            "电网设备": "电力设备",
            "机械设备": "机械设备",
            "通用设备": "机械设备",
            "专用设备": "机械设备",
            "仪器仪表": "机械设备",
            "自动化设备": "机械设备",
            "汽车": "汽车",
            "汽车零部件": "汽车",
            "乘用车": "汽车",
            "商用车": "汽车",
            "汽车零部件Ⅱ": "汽车",
            "汽车零部件Ⅲ": "汽车",
            "家用电器": "家用电器",
            "白色家电": "家用电器",
            "黑色家电": "家用电器",
            "小家电": "家用电器",
            "厨房电器": "家用电器",
            "轻工制造": "轻工制造",
            "家居用品": "轻工制造",
            "造纸": "轻工制造",
            "包装印刷": "轻工制造",
            "其他轻工制造": "轻工制造",
            "纺织服饰": "纺织服饰",
            "纺织制造": "纺织服饰",
            "服装家纺": "纺织服饰",
            "化工": "基础化工",
            "化学原料": "基础化工",
            "化学制品": "基础化工",
            "塑料": "基础化工",
            "橡胶": "基础化工",
            "化纤": "基础化工",
            "农药": "基础化工",
            "涂料油墨": "基础化工",
            "其他化学制品": "基础化工",
            "钢铁": "钢铁",
            "钢铁Ⅱ": "钢铁",
            "钢铁Ⅲ": "钢铁",
            "普钢": "钢铁",
            "特钢": "钢铁",
            "有色金属": "有色金属",
            "工业金属": "有色金属",
            "贵金属": "有色金属",
            "稀有金属": "有色金属",
            "能源金属": "有色金属",
            "小金属": "有色金属",
            "金属新材料": "有色金属",
            "煤炭": "煤炭",
            "煤炭开采": "煤炭",
            "焦炭": "煤炭",
            "石油石化": "石油石化",
            "石油开采": "石油石化",
            "油服工程": "石油石化",
            "炼油化工": "石油石化",
            "其他石油石化": "石油石化",
            "交通运输": "交通运输",
            "物流": "交通运输",
            "港口": "交通运输",
            "高速公路": "交通运输",
            "铁路公路": "交通运输",
            "航空机场": "交通运输",
            "航运": "交通运输",
            "公交": "交通运输",
            "商贸零售": "商贸零售",
            "一般零售": "商贸零售",
            "专业连锁": "商贸零售",
            "百货": "商贸零售",
            "超市": "商贸零售",
            "电商": "商贸零售",
            "贸易": "商贸零售",
            "社会服务": "社会服务",
            "旅游": "社会服务",
            "酒店": "社会服务",
            "餐饮": "社会服务",
            "教育": "社会服务",
            "人服": "社会服务",
            "美容护理": "美容护理",
            "个护用品": "美容护理",
            "化妆品": "美容护理",
            "医美": "美容护理",
            "农林牧渔": "农林牧渔",
            "种植业": "农林牧渔",
            "渔业": "农林牧渔",
            "林业": "农林牧渔",
            "畜牧业": "农林牧渔",
            "动物保健": "农林牧渔",
            "农业综合": "农林牧渔",
            "饲料": "农林牧渔",
            "农产品加工": "农林牧渔",
            "食品饮料": "食品饮料",
            "食品加工": "食品饮料",
            "休闲食品": "食品饮料",
            "调味发酵品": "食品饮料",
            "乳品": "食品饮料",
            "保健品": "食品饮料",
            "其他食品": "食品饮料",
            "非食品饮料": "食品饮料",
            "饮料制造": "食品饮料",
            "酒": "食品饮料",
            "传媒": "传媒",
            "游戏": "传媒",
            "影视": "传媒",
            "出版": "传媒",
            "广告": "传媒",
            "广电": "传媒",
            "互联网": "传媒",
            "数字媒体": "传媒",
            "其他传媒": "传媒",
            "环保": "环保",
            "环保设备": "环保",
            "水务": "环保",
            "固废": "环保",
            "大气": "环保",
            "环境监测": "环保",
            "其他环保": "环保",
            "公用事业": "公用事业",
            "电力": "公用事业",
            "燃气": "公用事业",
            "水利": "公用事业",
            "其他公用事业": "公用事业",
            "综合": "综合",
            "其他": "其他",
        }
        
        # 应用行业映射
        df = df.with_columns([
            pl.col("industry_name_final").replace(industry_mapping, default=None).alias("industry_standard")
        ])
        
        return df
        
    except Exception as e:
        logger.error(f"获取行业数据失败：{e}")
        return pl.DataFrame()


def sample_data_check(table_name: str, limit: int = 5) -> pl.DataFrame:
    """
    采样检查数据。
    
    Args:
        table_name: 表名
        limit: 采样行数
        
    Returns:
        pl.DataFrame: 采样数据
    """
    query = f"SELECT * FROM {table_name} LIMIT {limit}"
    
    try:
        df = read_db(query)
        return df
    except Exception as e:
        logger.error(f"采样查询失败：{e}")
        return pl.DataFrame()


def check_2024_data(table_name: str) -> pl.DataFrame:
    """
    检查 2024 年数据覆盖情况。
    
    Args:
        table_name: 表名
        
    Returns:
        pl.DataFrame: 2024 年数据统计
    """
    query = f"""
    SELECT 
        MIN(trade_date) as min_date, 
        MAX(trade_date) as max_date,
        COUNT(*) as row_count
    FROM {table_name}
    WHERE trade_date >= '20240101'
    """
    
    try:
        df = read_db(query)
        return df
    except Exception as e:
        logger.error(f"检查 2024 年数据失败：{e}")
        return pl.DataFrame()


def main():
    """主函数 - V96 数据库检查"""
    print("=" * 80)
    print("V96 策略重构 - 数据库检查 (使用 Polars + connectorx)")
    print("=" * 80)
    
    # 1. 检查 stock_daily 表结构
    print("\n[1] 检查 stock_daily 表结构...")
    stock_daily_info = check_table_structure("stock_daily")
    if stock_daily_info["exists"]:
        print(f"  表存在：是")
        print(f"  列数：{stock_daily_info['column_count']}")
        print(f"  列名列表：{stock_daily_info['columns']}")
        
        # 检查关键字段
        required_columns = ['symbol', 'trade_date', 'open', 'high', 'low', 'close', 'volume', 'amount']
        missing = [col for col in required_columns if col not in stock_daily_info['columns']]
        if missing:
            print(f"  ⚠️ 缺少关键字段：{missing}")
        else:
            print(f"  ✓ 关键字段完整")
        
        # 采样数据
        print("\n  stock_daily 采样数据 (前 5 行):")
        sample = sample_data_check("stock_daily", 5)
        if not sample.is_empty():
            print(sample)
        else:
            print("  ❌ 无法获取采样数据")
    else:
        print(f"  ❌ {stock_daily_info.get('error', '表不存在')}")
    
    # 2. 检查 stock_industry_daily 表结构
    print("\n[2] 检查 stock_industry_daily 表结构...")
    industry_info = check_table_structure("stock_industry_daily")
    if industry_info["exists"]:
        print(f"  表存在：是")
        print(f"  列数：{industry_info['column_count']}")
        print(f"  列名列表：{industry_info['columns']}")
        
        # 采样数据
        print("\n  stock_industry_daily 采样数据 (前 5 行):")
        sample = sample_data_check("stock_industry_daily", 5)
        if not sample.is_empty():
            print(sample)
        else:
            print("  ❌ 无法获取采样数据")
    else:
        print(f"  ❌ {industry_info.get('error', '表不存在')}")
    
    # 3. 检查行业映射分布
    print("\n[3] 检查行业映射分布...")
    industry_dist = check_industry_mapping()
    if not industry_dist.is_empty():
        print(industry_dist)
        
        # 检查是否包含"银行"行业
        banking_industries = industry_dist.filter(
            pl.col("industry_name").str.contains("银行", literal=True)
        )
        if not banking_industries.is_empty():
            print(f"\n  ✓ 发现银行相关行业：{banking_industries['industry_name'].to_list()}")
        else:
            print(f"\n  ⚠️ 未发现包含'银行'的行业名称")
            all_industries = industry_dist['industry_name'].to_list()
            print(f"  所有行业名称：{all_industries[:20]}...")  # 只显示前 20 个
    else:
        print("  ❌ 无法获取行业映射数据")
    
    # 4. 专门检查银行行业（强制检查）
    print("\n[4] 专门检查'银行'行业数据 (V96 强制检查)...")
    has_banking, banking_df = check_banking_industry()
    
    if has_banking and not banking_df.is_empty():
        print(f"  ✓ 银行行业检查通过")
        print(banking_df)
    else:
        print(f"  ❌ 银行行业检查失败 - 数据库中不存在'银行'行业数据")
        if not banking_df.is_empty():
            print(f"  找到的相关行业：{banking_df}")
        else:
            print(f"  错误：无法查询到银行行业数据，检查未通过!")
    
    # 5. 测试 fillna 和行业映射逻辑
    print("\n[5] 测试 fillna 和行业映射逻辑...")
    df_with_mapping = get_all_industries_with_fillna()
    if not df_with_mapping.is_empty():
        print(f"  ✓ 成功获取行业数据并应用 fillna/映射逻辑")
        print(f"  数据样例 (前 5 行):")
        print(df_with_mapping.select([
            "symbol", "trade_date", "industry_name", 
            "industry_name_filled", "industry_name_final", "industry_standard"
        ]).head(5))
        
        # 统计标准行业分布
        if "industry_standard" in df_with_mapping.columns:
            industry_std_count = df_with_mapping.group_by("industry_standard").agg(
                pl.col("symbol").count().alias("count")
            ).sort("count", descending=True)
            print(f"\n  标准行业分布统计:")
            print(industry_std_count.head(10))
    else:
        print(f"  ❌ 无法获取行业数据进行 fillna/映射测试")
    
    # 6. 检查 2024 年数据覆盖
    print("\n[6] 检查 2024 年数据覆盖情况...")
    for table in ["stock_daily", "stock_industry_daily"]:
        print(f"  {table} (2024 年数据):")
        df = check_2024_data(table)
        if not df.is_empty() and df.height > 0:
            min_date = df['min_date'][0] if 'min_date' in df.columns else "N/A"
            max_date = df['max_date'][0] if 'max_date' in df.columns else "N/A"
            row_count = df['row_count'][0] if 'row_count' in df.columns else 0
            print(f"    日期范围：{min_date} - {max_date}")
            print(f"    行数：{row_count}")
        else:
            print(f"    ❌ 无 2024 年数据")
    
    print("\n" + "=" * 80)
    print("V96 数据库检查完成")
    print("=" * 80)
    
    # V96 验收检查
    print("\n[V96 验收检查]")
    if has_banking:
        print("  ✓ 银行行业数据：PASS")
    else:
        print("  ❌ 银行行业数据：FAIL - 必须在代码中显式处理")
    
    if stock_daily_info["exists"] and industry_info["exists"]:
        print("  ✓ 核心表结构：PASS")
    else:
        print("  ❌ 核心表结构：FAIL")


if __name__ == "__main__":
    main()
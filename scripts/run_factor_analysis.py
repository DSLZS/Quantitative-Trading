#!/usr/bin/env python3
"""
因子分析主脚本 - 适配现有项目结构
"""

import sys
import os
import logging
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
from src.data.database_manager import DatabaseManager
from src.factors.factor_library import (
    momentum, volatility, ma_ratio, rsi, macd, 
    turnover_ratio, price_to_earnings, book_to_price, liquidity
)
from src.factors.factor_analyzer import FactorAnalyzer

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def prepare_data_from_db(symbol: str = '510300.SH', 
                         start_date: str = '2018-01-01',
                         end_date: str = '2023-12-31') -> pd.DataFrame:
    """
    从数据库准备分析数据。
    
    注意：由于我们目前只有单标的，这里返回的是单标的DataFrame。
    真正的截面分析需要多股票数据。
    """
    logger.info(f"从数据库加载数据: {symbol}")
    
    db_manager = DatabaseManager()
    
    # 获取股票数据 - 使用你现有的fetch_stock_data方法
    # 注意：需要确保database_manager中的fetch_stock_data方法能正确处理'510300.SH'格式
    stock_data = db_manager.fetch_stock_data(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date,
        fields=['trade_date', 'close', 'volume', 'amount', 'pct_change'],
        use_adj='forward'  # 使用前复权价格
    )
    
    if stock_data.empty:
        logger.error(f"未获取到 {symbol} 的数据")
        return pd.DataFrame()
    
    # 设置索引
    if 'trade_date' in stock_data.columns:
        stock_data.set_index('trade_date', inplace=True)
        stock_data.sort_index(inplace=True)
    
    logger.info(f"数据加载完成: {len(stock_data)} 条记录, {stock_data.index.min()} 至 {stock_data.index.max()}")
    return stock_data

def create_factor_config() -> dict:
    """创建因子计算配置。"""
    # 注意：这里的设计假设每个因子函数接收price_data DataFrame
    # 你可能需要根据实际因子函数的签名调整这个配置
    
    return {
        'momentum_20': {
            'func': lambda price_data: momentum(price_data['close'], window=20),
            'args': {}
        },
        'momentum_60': {
            'func': lambda price_data: momentum(price_data['close'], window=60),
            'args': {}
        },
        'volatility_20': {
            'func': lambda price_data: volatility(price_data['close'], window=20),
            'args': {}
        },
        'ma_ratio_5_20': {
            'func': lambda price_data: ma_ratio(price_data['close'], short_window=5, long_window=20),
            'args': {}
        },
        'rsi_14': {
            'func': lambda price_data: rsi(price_data['close'], window=14),
            'args': {}
        },
        'macd': {
            'func': lambda price_data: macd(price_data['close']),
            'args': {}
        },
        'liquidity_20': {
            'func': lambda price_data: liquidity(price_data['close'], price_data['volume'], window=20),
            'args': {}
        },
        # 由于我们目前可能没有基本面数据，以下因子可能需要调整
        # 'pe_ratio': {
        #     'func': lambda price_data, pe_data: price_to_earnings(pe_data['pe_ratio']),
        #     'args': {'pe_data': pe_data}  # 需要从其他地方获取PE数据
        # },
    }

def main():
    """主函数"""
    logger.info("开始因子批量分析")
    
    # 1. 准备数据
    stock_data = prepare_data_from_db(
        symbol='510300.SH',
        start_date='2020-01-01',  # 使用近几年的数据
        end_date='2023-12-31'
    )
    
    if stock_data.empty:
        logger.error("数据准备失败，程序退出")
        return
    
    # 2. 计算未来收益率（例如未来5日收益率）
    # 注意：在单标的模式下，这是时间序列分析
    forward_window = 5
    stock_data['forward_returns'] = stock_data['close'].pct_change(forward_window).shift(-forward_window)
    
    # 3. 准备价格数据（保持简单）
    price_data = stock_data[['close', 'volume']].copy()
    
    # 4. 创建因子配置
    factor_config = create_factor_config()
    
    # 5. 初始化分析器
    analyzer = FactorAnalyzer(
        price_data=price_data,
        forward_returns=stock_data['forward_returns']
    )
    
    # 6. 批量计算因子
    analyzer.calculate_all_factors(factor_config)
    
    # 7. 分析所有因子
    results_df = analyzer.analyze_all_factors()
    
    if results_df.empty:
        logger.warning("没有有效的分析结果")
        return
    
    # 8. 生成报告
    reports_dir = project_root / 'outputs' / 'factor_reports'
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = reports_dir / f'factor_analysis_{pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")}.txt'
    analyzer.generate_report(results_df, str(report_path))
    
    # 9. 打印重要发现
    print("\n" + "="*60)
    print("关键发现:")
    print("="*60)
    
    # 找出IC最高的因子
    if not results_df['IC'].isnull().all():
        best_ic_factor = results_df.loc[results_df['IC'].abs().idxmax()]
        print(f"IC绝对值最高的因子: {best_ic_factor['factor_name']}")
        print(f"  IC值: {best_ic_factor['IC']:.4f}")
        print(f"  Rank IC: {best_ic_factor['Rank_IC']:.4f}")
        print(f"  观察点数: {best_ic_factor['obs_count']}")
    
    # 找出ICIR最高的因子
    if not results_df['ICIR'].isnull().all():
        best_icir_factor = results_df.loc[results_df['ICIR'].abs().idxmax()]
        print(f"\nICIR最高的因子: {best_icir_factor['factor_name']}")
        print(f"  ICIR值: {best_icir_factor['ICIR']:.4f}")
        print(f"  IC均值: {best_icir_factor['IC_mean']:.4f}")
        print(f"  IC标准差: {best_icir_factor['IC_std']:.4f}")
    
    print("\n注意: 当前为单标的分析模式，结果仅供参考。")
    print("真正的因子有效性检验需要在多股票截面中进行。")
    print("="*60)
    
    logger.info("因子分析流程完成")

if __name__ == '__main__':
    main()
#!/usr/bin/env python
"""V81 数据加载诊断脚本"""

from src.db_manager import DatabaseManager
from src.core.v81_logic import V81DataManager, V81AlphaCenter

def main():
    db = DatabaseManager()
    dm = V81DataManager(db=db)
    
    print('=== 数据加载检查 ===')
    
    try:
        stock_df = dm.load_stock_data('2024-01-01', '2024-01-31')
        print(f'股票数据：{stock_df.height} 行，{stock_df.width} 列')
        print(f'日期范围：{stock_df["trade_date"].min()} 至 {stock_df["trade_date"].max()}')
        print(f'股票代码数：{stock_df["symbol"].n_unique()}')
        print(f'列名：{stock_df.columns}')
    except Exception as e:
        print(f'股票数据加载失败：{e}')
    
    try:
        fund_flow_df = dm.load_fund_flow_data('2024-01-01', '2024-01-31')
        print(f'资金流数据：{fund_flow_df.height} 行')
        if fund_flow_df.height > 0:
            print(f'资金流列：{fund_flow_df.columns}')
    except Exception as e:
        print(f'资金流数据加载失败：{e}')
    
    try:
        industry_mapping = dm.load_industry_mapping()
        print(f'行业映射：{industry_mapping.height} 行')
        print(f'行业映射列：{industry_mapping.columns}')
    except Exception as e:
        print(f'行业映射加载失败：{e}')
    
    try:
        index_df = dm.load_index_data('2024-01-01', '2024-01-31')
        print(f'指数数据：{index_df.height} 行')
        if index_df.height > 0:
            print(f'指数列：{index_df.columns}')
    except Exception as e:
        print(f'指数数据加载失败：{e}')
    
    # 测试 AlphaCenter
    print('\n=== AlphaCenter 信号计算检查 ===')
    try:
        if stock_df.height > 0:
            alpha = V81AlphaCenter()
            result, status = alpha.compute_signals(stock_df, fund_flow_df, industry_mapping, index_df)
            print(f'信号计算结果：{result.height} 行')
            print(f'状态：{status}')
            if result.height > 0:
                print(f'结果列：{result.columns}')
                # 检查 composite_score
                if 'composite_score' in result.columns:
                    print(f'composite_score 范围：{result["composite_score"].min()} - {result["composite_score"].max()}')
                # 检查 buy_signal
                if 'buy_signal' in result.columns:
                    buy_count = result.filter(pl.col('buy_signal') == True).height
                    print(f'买入信号数量：{buy_count}')
    except Exception as e:
        print(f'信号计算失败：{e}')
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    import polars as pl
    main()
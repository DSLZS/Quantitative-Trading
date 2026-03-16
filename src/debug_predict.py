"""调试预测分数"""
import sys
sys.path.insert(0, 'd:/PythonProject/Quantitative-Trading')

from datetime import datetime
from loguru import logger
import polars as pl
from final_strategy_v3_final import FinalStrategyV3Final, TRAIN_END_DATE

# 设置日志级别
logger.remove()
logger.add(sys.stderr, level="DEBUG")

# 创建策略实例
strategy = FinalStrategyV3Final(config_path="config/production_params.yaml")

# 训练模型
strategy.train_model(train_end_date=TRAIN_END_DATE)

# 获取回测数据
query = """
    SELECT * FROM stock_daily 
    WHERE trade_date >= '2024-01-01' 
    AND trade_date <= '2024-01-10'
"""
data = strategy.db.read_sql(query)
print(f"获取到 {len(data)} 条记录")

# 按日期分组检查预测
dates = data["trade_date"].unique().to_list()
for date in dates:
    daily_data = data.filter(pl.col("trade_date") == date)
    print(f"\n=== 日期：{date} ===")
    print(f"股票数：{len(daily_data)}")
    
    # 预测
    predicted = strategy.predict(daily_data)
    
    if "predict_score" in predicted.columns:
        score_min = predicted["predict_score"].min()
        score_max = predicted["predict_score"].max()
        score_mean = predicted["predict_score"].mean()
        positive_count = predicted.filter(pl.col("predict_score") > 0).height
        print(f"预测分数：min={score_min:.4f}, max={score_max:.4f}, mean={score_mean:.4f}, 正预测数={positive_count}")
        
        # 显示前 5 个正预测的股票
        positive_stocks = predicted.filter(pl.col("predict_score") > 0).sort("predict_score", descending=True).head(5)
        if len(positive_stocks) > 0:
            print("Top 5 正预测股票:")
            for row in positive_stocks.iter_rows(named=True):
                print(f"  {row.get('symbol')}: {row.get('predict_score', 0):.4f}")
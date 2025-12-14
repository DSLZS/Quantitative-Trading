# src/strategy/dual_ma.py
import pandas as pd
import numpy as np
from typing import Dict, Any

class DualMAStrategy:
    """双均线策略（本地复现版）"""
    
    def __init__(self, fast_period: int = 10, slow_period: int = 30):
        """
        初始化策略
        
        Args:
            fast_period: 快线周期（默认10日）
            slow_period: 慢线周期（默认30日）
        """
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.signals = []  # 记录交易信号
        
    def calculate_ma(self, prices: pd.Series, window: int) -> pd.Series:
        """计算移动平均线"""
        return prices.rolling(window=window).mean()
    
    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        生成交易信号
        
        Args:
            data: 必须包含 'price' 列（价格序列）和 'trade_date' 列
            
        Returns:
            添加了信号列的DataFrame，包含：
            - ma_fast: 快线
            - ma_slow: 慢线  
            - signal: 交易信号（1买入，-1卖出，0持有）
            - position: 仓位（1持有，0空仓）
        """
        df = data.copy()
        
        # 确保按日期排序
        if 'trade_date' in df.columns:
            df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 计算双均线
        df['ma_fast'] = self.calculate_ma(df['price'], self.fast_period)
        df['ma_slow'] = self.calculate_ma(df['price'], self.slow_period)
        
        # 初始化信号列
        df['signal'] = 0
        df['position'] = 0
        
        # 生成交易信号（金叉买，死叉卖）
        for i in range(1, len(df)):
            # 跳过NaN值
            if pd.isna(df.loc[i, 'ma_fast']) or pd.isna(df.loc[i, 'ma_slow']):
                continue
            
            # 金叉买入信号
            if (df.loc[i-1, 'ma_fast'] <= df.loc[i-1, 'ma_slow'] and 
                df.loc[i, 'ma_fast'] > df.loc[i, 'ma_slow']):
                df.loc[i, 'signal'] = 1
                self.signals.append({
                    'date': df.loc[i, 'trade_date'],
                    'type': 'BUY',
                    'price': df.loc[i, 'price'],
                    'reason': '金叉'
                })
            
            # 死叉卖出信号  
            elif (df.loc[i-1, 'ma_fast'] >= df.loc[i-1, 'ma_slow'] and 
                  df.loc[i, 'ma_fast'] < df.loc[i, 'ma_slow']):
                df.loc[i, 'signal'] = -1
                self.signals.append({
                    'date': df.loc[i, 'trade_date'],
                    'type': 'SELL',
                    'price': df.loc[i, 'price'],
                    'reason': '死叉'
                })
        
        # 计算仓位（买入后持有，卖出后空仓）
        position = 0
        for i in range(len(df)):
            if df.loc[i, 'signal'] == 1:
                position = 1
            elif df.loc[i, 'signal'] == -1:
                position = 0
            df.loc[i, 'position'] = position
        
        return df
    
    def get_signals(self) -> pd.DataFrame:
        """获取交易信号记录"""
        return pd.DataFrame(self.signals) if self.signals else pd.DataFrame()
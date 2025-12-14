# src/backtest/engine.py
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

class BacktestEngine:
    """简易回测引擎"""
    
    def __init__(self, initial_capital: float = 100000.0,
                 commission_rate: float = 0.0003,
                 slippage_rate: float = 0.0001):
        """
        初始化回测引擎
        
        Args:
            initial_capital: 初始资金
            commission_rate: 佣金费率（默认万分之三）
            slippage_rate: 滑点费率（默认万分之一）
        """
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        
    def run(self, data_with_signals: pd.DataFrame) -> Dict:
        """
        运行回测
        
        Args:
            data_with_signals: 包含价格和信号的DataFrame，必须有：
                              - trade_date: 日期
                              - price: 价格
                              - position: 仓位（0或1）
                              
        Returns:
            包含回测结果的字典
        """
        df = data_with_signals.copy()
        df = df.sort_values('trade_date').reset_index(drop=True)
        
        # 初始化
        capital = self.initial_capital
        position = 0  # 持股数量
        trades = []   # 交易记录
        equity = []   # 每日资产
        
        for i in range(len(df)):
            date = df.loc[i, 'trade_date']
            price = df.loc[i, 'price']
            target_position = df.loc[i, 'position']  # 目标仓位比例（0或1）
            
            # 计算目标持股数量（全仓或空仓）
            if target_position == 1 and position == 0:
                # 买入信号：全仓买入
                shares_to_buy = int(capital / price)
                if shares_to_buy > 0:
                    # 计算交易成本
                    trade_value = shares_to_buy * price
                    commission = trade_value * self.commission_rate
                    slippage = trade_value * self.slippage_rate
                    total_cost = commission + slippage
                    
                    # 更新持仓和现金
                    position = shares_to_buy
                    capital -= (trade_value + total_cost)
                    
                    trades.append({
                        'date': date,
                        'type': 'BUY',
                        'price': price,
                        'shares': shares_to_buy,
                        'value': trade_value,
                        'commission': commission,
                        'slippage': slippage
                    })
            
            elif target_position == 0 and position > 0:
                # 卖出信号：全仓卖出
                trade_value = position * price
                commission = trade_value * self.commission_rate
                slippage = trade_value * self.slippage_rate
                total_cost = commission + slippage
                
                capital += (trade_value - total_cost)
                
                trades.append({
                    'date': date,
                    'type': 'SELL',
                    'price': price,
                    'shares': position,
                    'value': trade_value,
                    'commission': commission,
                    'slippage': slippage
                })
                
                position = 0
            
            # 计算当日资产
            daily_value = capital + (position * price)
            equity.append({
                'date': date,
                'capital': capital,
                'position': position,
                'price': price,
                'total_value': daily_value
            })
        
        # 构建回测结果
        equity_df = pd.DataFrame(equity)
        trades_df = pd.DataFrame(trades) if trades else pd.DataFrame()
        
        # 计算绩效指标
        metrics = self._calculate_metrics(equity_df, trades_df)
        
        return {
            'equity': equity_df,
            'trades': trades_df,
            'metrics': metrics,
            'initial_capital': self.initial_capital,
            'final_capital': equity_df['total_value'].iloc[-1] if not equity_df.empty else self.initial_capital
        }
    
    def _calculate_metrics(self, equity_df: pd.DataFrame, trades_df: pd.DataFrame) -> Dict:
        """计算回测绩效指标"""
        if equity_df.empty:
            return {}
        
        # 基础数据
        initial = self.initial_capital
        final = equity_df['total_value'].iloc[-1]
        total_return = (final - initial) / initial
        
        # 计算时间跨度（年）
        days = (equity_df['date'].iloc[-1] - equity_df['date'].iloc[0]).days
        years = max(days / 365.25, 1/365.25)
        
        # 年化收益率
        annual_return = (1 + total_return) ** (1 / years) - 1
        
        # 每日收益率
        equity_df['daily_return'] = equity_df['total_value'].pct_change()
        daily_returns = equity_df['daily_return'].dropna()
        
        # 年化波动率
        volatility = daily_returns.std() * np.sqrt(252)
        
        # 夏普比率（假设无风险利率3%）
        risk_free_rate = 0.03
        sharpe_ratio = (annual_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # 最大回撤
        equity_df['cummax'] = equity_df['total_value'].cummax()
        equity_df['drawdown'] = (equity_df['total_value'] - equity_df['cummax']) / equity_df['cummax']
        max_drawdown = equity_df['drawdown'].min()
        
        # 交易统计
        total_trades = len(trades_df)
        win_trades = 0
        
        if total_trades >= 2:
            # 简单计算胜率（卖价 > 买价即为盈利）
            for i in range(0, len(trades_df)-1, 2):
                if i+1 < len(trades_df):
                    buy_price = trades_df.iloc[i]['price']
                    sell_price = trades_df.iloc[i+1]['price']
                    if sell_price > buy_price:
                        win_trades += 1
            
            win_rate = win_trades / (total_trades // 2) if total_trades >= 2 else 0
        else:
            win_rate = 0
        
        return {
            '总收益率': f"{total_return:.2%}",
            '年化收益率': f"{annual_return:.2%}",
            '夏普比率': f"{sharpe_ratio:.2f}",
            '最大回撤': f"{max_drawdown:.2%}",
            '年化波动率': f"{volatility:.2%}",
            '总交易次数': total_trades,
            '胜率': f"{win_rate:.2%}" if total_trades >= 2 else "N/A",
            '初始资金': f"¥{initial:,.2f}",
            '最终资金': f"¥{final:,.2f}",
            '净收益': f"¥{final - initial:,.2f}"
        }
    
    def generate_report(self, backtest_result: Dict, save_path: Optional[str] = None) -> str:
        """生成回测报告"""
        result = backtest_result
        metrics = result['metrics']
        
        report = []
        report.append("=" * 60)
        report.append("双均线策略回测报告")
        report.append("=" * 60)
        
        for key, value in metrics.items():
            report.append(f"{key:>10}: {value}")
        
        report.append("\n交易记录:")
        if not result['trades'].empty:
            report.append(result['trades'].to_string(index=False))
        else:
            report.append("无交易")
        
        report_str = "\n".join(report)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report_str)
            logger.info(f"回测报告已保存至: {save_path}")
        
        return report_str
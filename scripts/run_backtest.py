#!/usr/bin/env python3
# scripts/run_backtest.py
"""
双均线策略本地回测 - 主启动脚本
运行方式
python scripts/run_backtest.py
python scripts/run_backtest.py --symbol 510300.SH
args:
    symbol: 股票代码，如 '510300.SH' 或纯代码 '510300' 510300.SH
    start_date: 开始日期，格式 'YYYY-MM-DD'
    end_date: 结束日期，格式 'YYYY-MM-DD'
    fields: 需要获取的字段列表，如果为None则获取所有字段
    use_adj: 复权类型 'forward' (前复权), 'back' (后复权), 'none' (不复权)
"""
import argparse
import sys
from pathlib import Path
import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

# 添加项目根目录到Python路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.database_manager import DatabaseManager
from src.strategies.dual_ma import DualMAStrategy
from src.backtest.engine import BacktestEngine

def main():
    print("=" * 60)
    print("双均线策略本地回测")
    print("=" * 60)

    # 获取传参
    parser = argparse.ArgumentParser(description='回测策略传参')
    parser.add_argument('--symbol', type=str, help='单只股票代码，如 510300.SH')
    parser.add_argument('--start_date', type=str, default="2020-01-01", help='回测开始日期 yyyy-mm-dd')
    parser.add_argument('--end_date', type=str, default="2024-01-01", help='回测结束日期 yyyy-mm-dd')
    parser.add_argument('--use_adj', type=str, default='forward', help='前后复权')

    args = parser.parse_args()
    
    # 1. 获取数据
    print("\n1. 从数据库获取数据...")
    db = DatabaseManager()
    

    # 获取数据（使用前复权价格）
    symbol=args.symbol
    data = db.fetch_stock_data(
        symbol=symbol,
        start_date=args.start_date,
        end_date=args.end_date,
        use_adj=args.use_adj  # 使用前复权
    )
    
    if data.empty:
        print(f"错误: 未获取到 {symbol} 的数据")
        return
    
    print(f"获取到 {len(data)} 条数据，时间范围: {data['trade_date'].min()} 到 {data['trade_date'].max()}")
    
    # 2. 运行策略
    print("\n2. 运行策略...")
    strategy = DualMAStrategy(fast_period=10, slow_period=30)
    data_with_signals = strategy.generate_signals(data)
    
    signals = strategy.get_signals()
    print(f"生成 {len(signals)} 个交易信号")
    if not signals.empty:
        print("最近5个信号:")
        print(signals.tail())
    
    # 3. 运行回测
    print("\n3. 运行回测引擎...")
    backtest = BacktestEngine(
        initial_capital=100000,
        commission_rate=0.0003,
        slippage_rate=0.0001
    )
    
    result = backtest.run(data_with_signals)
    
    # 4. 输出报告
    print("\n4. 回测结果:")
    report = backtest.generate_report(result)
    print(report)
    
    # 5. 可视化
    print("\n5. 生成图表...")
    plot_results(result, data_with_signals, symbol)
    
    # 6. 保存结果
    print("\n6. 保存结果...")
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    # 保存权益曲线
    equity_file = output_dir / "equity_curve.csv"
    result['equity'].to_csv(equity_file, index=False)
    print(f"权益曲线保存至: {equity_file}")
    
    # 保存交易记录
    if not result['trades'].empty:
        trades_file = output_dir / "trades.csv"
        result['trades'].to_csv(trades_file, index=False)
        print(f"交易记录保存至: {trades_file}")
    
    # 保存报告
    report_file = output_dir / "backtest_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"回测报告保存至: {report_file}")
    
    print("\n✅ 回测完成！")

def plot_results(backtest_result, data_with_signals, symbol):
    """绘制回测结果图表"""
    matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # Windows
    matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))
    
    # 1. 价格与均线
    ax1 = axes[0]
    ax1.plot(data_with_signals['trade_date'], data_with_signals['price'], 
             label='价格', linewidth=1, alpha=0.7)
    ax1.plot(data_with_signals['trade_date'], data_with_signals['ma_fast'], 
             label=f'MA{10}', linewidth=1.5)
    ax1.plot(data_with_signals['trade_date'], data_with_signals['ma_slow'], 
             label=f'MA{30}', linewidth=1.5)
    
    # 标记买卖点
    signals = data_with_signals[data_with_signals['signal'] != 0]
    buy_signals = signals[signals['signal'] == 1]
    sell_signals = signals[signals['signal'] == -1]
    
    if not buy_signals.empty:
        ax1.scatter(buy_signals['trade_date'], buy_signals['price'], 
                   color='green', s=80, marker='^', label='买入', zorder=5)
    if not sell_signals.empty:
        ax1.scatter(sell_signals['trade_date'], sell_signals['price'], 
                   color='red', s=80, marker='v', label='卖出', zorder=5)
    
    ax1.set_title(f'{symbol} - 价格与双均线', fontsize=14)
    ax1.set_ylabel('价格')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 资产净值曲线
    ax2 = axes[1]
    equity = backtest_result['equity']
    ax2.plot(equity['date'], equity['total_value'], label='资产净值', linewidth=2, color='blue')
    ax2.axhline(y=backtest_result['initial_capital'], color='red', linestyle='--', alpha=0.5, label='初始资金')
    ax2.set_title('资产净值曲线', fontsize=14)
    ax2.set_ylabel('资产净值（元）')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 回撤曲线
    ax3 = axes[2]
    equity['cummax'] = equity['total_value'].cummax()
    equity['drawdown'] = (equity['total_value'] - equity['cummax']) / equity['cummax'] * 100
    ax3.fill_between(equity['date'], equity['drawdown'], 0, color='red', alpha=0.3)
    ax3.plot(equity['date'], equity['drawdown'], color='red', linewidth=1)
    ax3.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax3.set_title('回撤曲线', fontsize=14)
    ax3.set_ylabel('回撤 (%)')
    ax3.set_xlabel('日期')
    ax3.grid(True, alpha=0.3)
    
    # 标记最大回撤
    max_dd_idx = equity['drawdown'].idxmin()
    max_dd_date = equity.loc[max_dd_idx, 'date']
    max_dd_value = equity.loc[max_dd_idx, 'drawdown']
    ax3.annotate(f'最大回撤: {max_dd_value:.2f}%',
                xy=(max_dd_date, max_dd_value),
                xytext=(max_dd_date, max_dd_value - 5),
                arrowprops=dict(facecolor='black', arrowstyle='->'),
                fontsize=10)
    
    plt.tight_layout()
    
    # 保存图表
    output_dir = Path(__file__).resolve().parent.parent / "outputs"
    chart_file = output_dir / "backtest_chart.png"
    plt.savefig(chart_file, dpi=300, bbox_inches='tight')
    print(f"图表保存至: {chart_file}")
    
    plt.show()

if __name__ == '__main__':
    main()
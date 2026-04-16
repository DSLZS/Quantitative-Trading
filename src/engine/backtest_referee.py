"""
Backtest Referee Module - V103 Immutable裁判引擎.

【架构强制规范 - 裁判 - 选手机制】
这是通用的、不可变的裁判类。它仅接收 pd.DataFrame 格式的信号，
并严格按 0.15% 费率执行会计核算。

【核心原则】
1. 不可变性：此文件一旦创建，严禁修改内部逻辑
2. 接口标准化：仅接收 DataFrame(trade_date, symbol, score)
3. 费率固定：佣金 0.03%, 印花税 0.1%, 滑点 0.05%
4. 审计透明：所有计算过程可追溯

【禁止事项】
- 严禁外部修改此文件的资金处理逻辑
- 严禁修改撮合算法和手续费设置
- 严禁添加任何未来函数

【验收标准】
- T+1 Rank IC > 0.05
- IC Decay 单调递减
- 模块独立性 100%
"""

from datetime import datetime
from typing import Any, Optional
from pathlib import Path
import json

import pandas as pd
import numpy as np
from loguru import logger

# 内存优化配置
pd.options.mode.chained_assignment = None


class BacktestReferee:
    """
    不可变裁判引擎。
    
    【裁判职责】
    1. 接收选手 (AlphaModule) 提交的信号
    2. 按固定费率执行会计核算
    3. 输出 IC 为核心的审计报告
    
    【接口规范】
    - 输入：pd.DataFrame with columns [trade_date, symbol, score]
    - 输出：dict with IC metrics and backtest results
    
    【不可变参数】
    - 佣金率：0.03% (万分之三)
    - 印花税：0.1% (千分之一，卖出收取)
    - 滑点：0.05% (万分之五)
    - 持仓数量：50 只
    - 单股票仓位：2% (等权配置)
    """
    
    # ==================== 不可变参数 ====================
    # 严禁修改以下参数
    
    COMMISSION_RATE = 0.0003      # 佣金率 (万分之三)
    STAMP_DUTY_RATE = 0.001       # 印花税 (千分之一)
    SLIPPAGE_RATE = 0.0005        # 滑点 (万分之五)
    TOP_N = 50                    # 持仓股票数量
    POSITION_PER_STOCK = 0.02     # 单股票仓位 (2% = 100%/50)
    INITIAL_CAPITAL = 100_000.00  # 初始资金 (10 万) - V104 锁定
    
    # ==================== 版本号 (动态) ====================
    VERSION = "V119"  # 可被子类覆盖
    
    # ==================== 验收阈值 ====================
    IC_THRESHOLD = 0.05           # T+1 IC 阈值
    IC_IR_THRESHOLD = 0.6         # IC IR 阈值
    TOP_FACTOR_IC_THRESHOLD = 0.04  # Top 因子 IC 阈值
    
    def __init__(self, alpha_module: Any, output_dir: str = "reports") -> None:
        """
        初始化裁判引擎。
        
        Args:
            alpha_module: Alpha 模块实例 (选手)
            output_dir: 报告输出目录
            
        【初始化检查】
        - 验证 alpha_module 具有 compute_score 方法
        - 创建输出目录
        """
        self.alpha_module = alpha_module
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 验证 Alpha 模块接口
        if not hasattr(alpha_module, 'compute_score'):
            raise ValueError("Alpha module must have 'compute_score' method")
        
        logger.info("=" * 70)
        logger.info("V103 Backtest Referee Initialized")
        logger.info("=" * 70)
        logger.info(f"  Alpha Module: {type(alpha_module).__name__}")
        logger.info(f"  Commission: {self.COMMISSION_RATE:.2%}")
        logger.info(f"  Stamp Duty: {self.STAMP_DUTY_RATE:.2%}")
        logger.info(f"  Slippage: {self.SLIPPAGE_RATE:.2%}")
        logger.info(f"  Top N Stocks: {self.TOP_N}")
        logger.info(f"  Initial Capital: {self.INITIAL_CAPITAL:,.0f}")
        logger.info("=" * 70)
    
    def validate_signal_input(self, df: pd.DataFrame) -> bool:
        """
        验证输入信号格式。
        
        【强制规范】
        输入 DataFrame 必须包含以下列:
        - trade_date: 交易日期
        - symbol: 股票代码
        - score: 预测评分
        
        Args:
            df: 输入数据
            
        Returns:
            bool: 验证是否通过
        """
        required_columns = {'trade_date', 'symbol', 'score'}
        actual_columns = set(df.columns)
        
        missing = required_columns - actual_columns
        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False
        
        # 检查数据类型
        if df['score'].dtype not in [np.float64, np.float32, np.int64, np.int32]:
            logger.warning(f"Score column dtype {df['score'].dtype} may cause issues")
        
        return True
    
    def generate_signals(self, df: pd.DataFrame, score_column: str = 'score') -> pd.DataFrame:
        """
        生成交易信号。
        
        【信号生成逻辑】
        - 每日按 score 排名，选择前 TOP_N 只股票
        - 信号值：1 = 买入/持有，0 = 不持有
        
        Args:
            df: 包含 score 的数据
            score_column: 评分列名
            
        Returns:
            包含 signal 列的 DataFrame
        """
        result = df.copy()
        
        if score_column not in result.columns:
            logger.error(f"Score column '{score_column}' not found")
            return result
        
        if 'trade_date' not in result.columns:
            logger.error("trade_date column not found")
            return result
        
        # 按日期分组计算排名
        result['score_rank'] = result.groupby('trade_date')[score_column].rank(
            method='dense', ascending=False
        )
        
        # 生成信号：前 TOP_N 名为 1，其余为 0
        result['signal'] = (result['score_rank'] <= self.TOP_N).astype(int)
        
        logger.debug(f"[Generate Signals] Generated {result['signal'].sum()} positions")
        return result
    
    def calculate_transaction_cost(
        self,
        buy_amount: float,
        sell_amount: float,
    ) -> dict[str, float]:
        """
        计算交易成本。
        
        【不可变费率】
        - 买入成本：佣金 + 滑点
        - 卖出成本：佣金 + 印花税 + 滑点
        
        Args:
            buy_amount: 买入金额
            sell_amount: 卖出金额
            
        Returns:
            交易成本明细
        """
        # 买入成本：佣金 + 滑点
        buy_cost = buy_amount * (self.COMMISSION_RATE + self.SLIPPAGE_RATE)
        
        # 卖出成本：佣金 + 印花税 + 滑点
        sell_cost = sell_amount * (self.COMMISSION_RATE + self.STAMP_DUTY_RATE + self.SLIPPAGE_RATE)
        
        return {
            'buy_cost': buy_cost,
            'sell_cost': sell_cost,
            'total_cost': buy_cost + sell_cost,
        }
    
    def run_backtest(self, signals: pd.DataFrame, returns: pd.DataFrame) -> dict[str, Any]:
        """
        运行回测。
        
        【回测流程】
        1. T 日生成信号（基于 score 排名）
        2. T+1 日执行交易（买入/卖出）
        3. T+1 日计算持仓收益（基于 T 日持仓）
        4. 计算交易成本
        5. 更新组合价值
        
        【会计逻辑】
        - current_capital = cash + positions_value (总资金 = 现金 + 持仓)
        - 买入：cash 减少，positions 增加，total 不变
        - 卖出：cash 增加，positions 减少，total 不变
        - 收益：positions 价值变化，total 变化
        
        Args:
            signals: 交易信号 DataFrame (包含 symbol, trade_date, signal, t1_return)
            returns: T+1 收益 DataFrame (包含 symbol, trade_date, t1_return)
            
        Returns:
            回测结果字典
        """
        logger.info("[Backtest] Running backtest...")
        
        # 合并信号和收益数据
        merged = signals.copy()
        if 't1_return' not in merged.columns:
            merged = merged.merge(
                returns[['symbol', 'trade_date', 't1_return']],
                on=['symbol', 'trade_date'],
                how='left'
            )
        
        # 按日期排序
        merged = merged.sort_values(['trade_date', 'symbol'])
        
        unique_dates = sorted(merged['trade_date'].unique())
        
        portfolio_values = []
        total_costs = []
        
        cash = self.INITIAL_CAPITAL  # 现金
        prev_positions = {}  # {symbol: position_value}
        prev_date = None
        
        for i, date in enumerate(unique_dates):
            day_data = merged[merged['trade_date'] == date]
            
            # 获取当日信号（T 日决策）
            positions = day_data[day_data['signal'] == 1]
            
            if len(positions) == 0:
                # 没有持仓，计算组合价值
                portfolio_value = cash + sum(prev_positions.values()) if prev_positions else cash
                
                # 计算昨日持仓的今日收益
                daily_profit = 0.0
                if prev_positions:
                    for sym, pos_value in prev_positions.items():
                        pos_data = day_data[day_data['symbol'] == sym]
                        if not pos_data.empty and 't1_return' in pos_data.columns:
                            ret = pos_data['t1_return'].values[0]
                            if not np.isnan(ret):
                                daily_profit += pos_value * ret
                            # 更新持仓价值
                            prev_positions[sym] = pos_value * (1 + ret)
                
                portfolio_values.append({
                    'trade_date': date,
                    'portfolio_value': portfolio_value + daily_profit,
                    'daily_return': daily_profit / portfolio_value if portfolio_value > 0 else 0,
                    'num_positions': len(prev_positions),
                    'transaction_cost': 0.0,
                })
                prev_positions = {}
                prev_date = date
                continue
            
            # 计算调仓（T 日决策，T+1 日执行）
            current_position_set = set(positions['symbol'].tolist())
            prev_position_set = set(prev_positions.keys()) if prev_positions else set()
            
            # 需要卖出的：之前持有但今日不持有
            to_sell = prev_position_set - current_position_set
            # 需要买入的：今日持有但之前不持有
            to_buy = current_position_set - prev_position_set
            # 需要调整的：继续持有的
            to_hold = current_position_set & prev_position_set
            
            # 计算昨日持仓的今日收益（在调仓前计算）
            daily_profit = 0.0
            if prev_positions and prev_date is not None:
                for sym, pos_value in prev_positions.items():
                    pos_data = day_data[day_data['symbol'] == sym]
                    if not pos_data.empty and 't1_return' in pos_data.columns:
                        ret = pos_data['t1_return'].values[0]
                        if not np.isnan(ret):
                            daily_profit += pos_value * ret
                        # 更新持仓价值
                        prev_positions[sym] = pos_value * (1 + ret)
            
            # 计算当前组合价值（调仓前）
            portfolio_value_before = cash + sum(prev_positions.values())
            
            # 执行调仓
            # 卖出
            sell_amount = 0.0
            for sym in to_sell:
                sell_amount += prev_positions.get(sym, 0)
                del prev_positions[sym]
            cash += sell_amount
            
            # 买入：计算目标仓位
            # 目标：每个持仓股票占总资金的 POSITION_PER_STOCK
            # 总持仓目标 = portfolio_value * POSITION_PER_STOCK * len(current_position_set)
            # 但为了简单，使用固定比例
            target_position_value = portfolio_value_before * self.POSITION_PER_STOCK
            
            # 计算需要买入的金额
            buy_amount = 0.0
            for sym in to_buy:
                buy_amount += target_position_value
            cash -= buy_amount
            
            # 计算交易成本
            cost = self.calculate_transaction_cost(buy_amount, sell_amount)
            total_costs.append(cost)
            cash -= cost['total_cost']
            
            # 更新持仓
            for sym in to_buy:
                prev_positions[sym] = target_position_value
            for sym in to_hold:
                # 继续持有的，调整到目标仓位
                prev_positions[sym] = target_position_value
            
            # 计算调仓后的组合价值
            portfolio_value = cash + sum(prev_positions.values())
            
            prev_date = date
            
            # 记录当日数据
            daily_return = daily_profit / portfolio_value_before if portfolio_value_before > 0 else 0
            portfolio_values.append({
                'trade_date': date,
                'portfolio_value': portfolio_value,
                'daily_return': daily_return,
                'num_positions': len(current_position_set),
                'transaction_cost': cost['total_cost'],
            })
        
        # 计算回测统计
        if not portfolio_values:
            return {'error': 'No portfolio values calculated'}
        
        portfolio_df = pd.DataFrame(portfolio_values)
        
        # 计算累计收益
        portfolio_df['cumulative_return'] = (1 + portfolio_df['daily_return']).cumprod() - 1
        
        # 计算年化收益
        num_days = len(unique_dates)
        if num_days > 0:
            total_return = portfolio_df['cumulative_return'].iloc[-1]
            annual_return = (1 + total_return) ** (252 / num_days) - 1
        else:
            annual_return = 0
            total_return = 0
        
        # 计算波动率
        daily_returns = portfolio_df['daily_return'].values
        volatility = np.std(daily_returns, ddof=1) * np.sqrt(252) if len(daily_returns) > 1 else 0
        
        # 计算夏普比率
        mean_daily_return = np.mean(daily_returns) if len(daily_returns) > 0 else 0
        sharpe = (mean_daily_return * 252) / volatility if volatility > 0 else 0
        
        # 计算最大回撤
        cum_values = (1 + portfolio_df['daily_return']).cumprod()
        running_max = cum_values.cummax()
        drawdown = (cum_values - running_max) / running_max
        max_drawdown = float(drawdown.min()) if len(drawdown) > 0 else 0
        
        # 计算总交易成本
        total_transaction_cost = sum(c['total_cost'] for c in total_costs)
        
        result = {
            'portfolio_df': portfolio_df,
            'total_return': float(total_return),
            'annual_return': float(annual_return),
            'volatility': float(volatility),
            'sharpe_ratio': float(sharpe),
            'max_drawdown': float(max_drawdown),
            'total_transaction_cost': float(total_transaction_cost),
            'num_trading_days': num_days,
            'final_value': float(portfolio_df['portfolio_value'].iloc[-1]) if len(portfolio_df) > 0 else self.INITIAL_CAPITAL,
        }
        
        logger.info(f"[Backtest] Total Return: {total_return:.2%}")
        logger.info(f"[Backtest] Annual Return: {annual_return:.2%}")
        logger.info(f"[Backtest] Sharpe Ratio: {sharpe:.2f}")
        logger.info(f"[Backtest] Max Drawdown: {max_drawdown:.2%}")
        
        return result
    
    def calculate_rank_ic(self, factor_values: pd.Series, label_values: pd.Series) -> float:
        """
        计算 Rank IC (Spearman 相关系数)。
        
        Args:
            factor_values: 因子值
            label_values: 标签值 (T+1 收益)
            
        Returns:
            Rank IC 值
        """
        # 去除空值
        mask = factor_values.notna() & label_values.notna()
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算秩
        factor_ranks = factor_clean.rank(method='average')
        label_ranks = label_clean.rank(method='average')
        
        # 计算 Pearson 相关系数
        if np.std(factor_ranks) < 1e-10 or np.std(label_ranks) < 1e-10:
            return 0.0
        
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_t1_ic(self, df: pd.DataFrame, score_column: str = 'score') -> dict[str, Any]:
        """
        计算 T+1 Rank IC。
        
        Args:
            df: 包含 score 和 t1_return 的数据
            score_column: 评分列名
            
        Returns:
            IC 统计字典
        """
        if score_column not in df.columns:
            logger.error(f"Score column '{score_column}' not found")
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
        
        if 't1_return' not in df.columns:
            logger.error("t1_return column not found")
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
        
        if 'trade_date' not in df.columns:
            logger.error("trade_date column not found")
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
        
        # 按日期分组计算 IC
        unique_dates = sorted(df['trade_date'].unique())
        ic_series = []
        
        for date in unique_dates:
            day_data = df[df['trade_date'] == date]
            
            if len(day_data) < 10:
                continue
            
            score_values = day_data[score_column]
            label_values = day_data['t1_return']
            
            ic = self.calculate_rank_ic(score_values, label_values)
            
            if not np.isnan(ic):
                ic_series.append({
                    'trade_date': date,
                    'ic': ic,
                })
        
        if not ic_series:
            return {'mean_ic': 0.0, 'ic_std': 0.0, 'ic_ir': 0.0, 'num_days': 0}
        
        ic_df = pd.DataFrame(ic_series)
        ic_values = ic_df['ic'].values
        
        mean_ic = float(np.mean(ic_values))
        ic_std = float(np.std(ic_values, ddof=1)) if len(ic_values) > 1 else 0.0
        ic_ir = mean_ic / ic_std if ic_std > 1e-10 else 0.0
        
        return {
            'mean_ic': mean_ic,
            'ic_std': ic_std,
            'ic_ir': ic_ir,
            'num_days': len(ic_values),
            'min_ic': float(np.min(ic_values)),
            'max_ic': float(np.max(ic_values)),
        }
    
    def calculate_ic_decay(self, df: pd.DataFrame, score_column: str = 'score') -> dict[str, Any]:
        """
        计算 IC Decay (T+1 to T+5) - V152 修复版.
        
        【V152 修复】
        IC Decay 应该反映预测能力的衰减，使用单期回报而非累计回报。
        - T+1 IC: 预测第 1 天单期回报的能力
        - T+3 IC: 预测第 3 天单期回报的能力
        - T+5 IC: 预测第 5 天单期回报的能力
        
        对于短期因子，T+1 IC 应该最高，因为预测的是近期回报。
        随着时间延长，预测能力应该衰减。
        
        【验收指标】
        IC 应该单调递减，如果 T+3 IC > T+1 IC，说明存在未来函数泄露
        
        Args:
            df: 包含 score 和各期收益的数据
            score_column: 评分列名
            
        Returns:
            IC Decay 分析结果
        """
        ic_decay = {}
        
        # V152 修复：使用单期回报计算 IC Decay
        # 优先使用 _period 列（单期回报）
        for n in [1, 3, 5]:
            return_col = f't{n}_return_period'
            if return_col in df.columns:
                ic = self.calculate_tn_ic(df, score_column, return_col)
                ic_decay[f't{n}_ic'] = ic
            else:
                # 回退到累计回报列
                return_col = f't{n}_return'
                if return_col in df.columns:
                    ic = self.calculate_tn_ic(df, score_column, return_col)
                    ic_decay[f't{n}_ic'] = ic
        
        # 检查单调性
        t1_ic = ic_decay.get('t1_ic', 0)
        t3_ic = ic_decay.get('t3_ic', 0)
        t5_ic = ic_decay.get('t5_ic', 0)
        
        is_monotonic = (t1_ic >= t3_ic >= t5_ic) if t1_ic > 0 else True
        
        if not is_monotonic and t1_ic > 0:
            logger.warning(f"[IC Decay Warning] IC not monotonically decreasing: T+1={t1_ic:.4f}, T+3={t3_ic:.4f}, T+5={t5_ic:.4f}")
            logger.warning("Possible look-ahead bias detected!")
        
        ic_decay['is_monotonic'] = is_monotonic
        ic_decay['decay_pattern'] = f"T+1({t1_ic:.4f}) -> T+3({t3_ic:.4f}) -> T+5({t5_ic:.4f})"
        
        return ic_decay
    
    def calculate_tn_ic(self, df: pd.DataFrame, score_column: str, return_column: str) -> float:
        """
        计算指定 Horizon 的 IC。
        
        Args:
            df: 输入数据
            score_column: 评分列
            return_column: 收益列
            
        Returns:
            IC 值
        """
        if score_column not in df.columns or return_column not in df.columns:
            return 0.0
        
        if 'trade_date' not in df.columns:
            return 0.0
        
        unique_dates = sorted(df['trade_date'].unique())
        ic_series = []
        
        for date in unique_dates:
            day_data = df[df['trade_date'] == date]
            
            if len(day_data) < 10:
                continue
            
            score_values = day_data[score_column]
            label_values = day_data[return_column]
            
            ic = self.calculate_rank_ic(score_values, label_values)
            
            if not np.isnan(ic):
                ic_series.append(ic)
        
        if not ic_series:
            return 0.0
        
        return float(np.mean(ic_series))
    
    def run_audit(self, df: pd.DataFrame) -> dict[str, Any]:
        """
        运行完整审计流程。
        
        【裁判流程】
        1. 验证输入格式
        2. 调用 Alpha 模块计算评分
        3. 计算 T+1 IC
        4. 计算 IC Decay
        5. 运行回测
        6. 生成报告
        
        Args:
            df: 原始数据 (必须包含 trade_date, symbol 和因子计算所需数据)
            
        Returns:
            完整审计结果
        """
        logger.info("=" * 70)
        logger.info("V103 Backtest Referee - Running Audit")
        logger.info("=" * 70)
        
        # 1. 调用 Alpha 模块计算评分
        logger.info("[Step 1] Computing alpha scores...")
        score_df = self.alpha_module.compute_score(df)
        
        # 2. 验证输出格式
        logger.info("[Step 2] Validating signal format...")
        if not self.validate_signal_input(score_df):
            return {'error': 'Invalid signal format', 'passed': False}
        
        # 3. 计算 T+1 IC
        logger.info("[Step 3] Calculating T+1 IC...")
        t1_ic = self.calculate_t1_ic(score_df, score_column='score')
        
        # 4. 计算 IC Decay
        logger.info("[Step 4] Calculating IC Decay...")
        ic_decay = self.calculate_ic_decay(score_df, score_column='score')
        
        # 5. 生成交易信号
        logger.info("[Step 5] Generating trading signals...")
        signals = self.generate_signals(score_df, score_column='score')
        
        # 6. 准备收益数据
        returns = score_df[['symbol', 'trade_date', 't1_return']].copy()
        
        # 7. 运行回测
        logger.info("[Step 6] Running backtest...")
        backtest_result = self.run_backtest(signals, returns)
        
        # 8. 获取因子 IC (如果 Alpha 模块提供)
        factor_ics = {}
        if hasattr(self.alpha_module, 'get_factor_ics'):
            factor_ics = self.alpha_module.get_factor_ics(score_df)
        
        # 9. 验收判断
        passed = (
            t1_ic['mean_ic'] > self.IC_THRESHOLD and
            t1_ic['ic_ir'] > self.IC_IR_THRESHOLD and
            ic_decay['is_monotonic']
        )
        
        # 10. 生成报告
        logger.info("[Step 7] Generating audit report...")
        report_path = self.generate_report(
            t1_ic=t1_ic,
            ic_decay=ic_decay,
            backtest_result=backtest_result,
            factor_ics=factor_ics,
            passed=passed,
        )
        
        # 汇总结果
        result = {
            't1_ic': t1_ic,
            'ic_decay': ic_decay,
            'backtest_result': backtest_result,
            'factor_ics': factor_ics,
            'passed': passed,
            'report_path': report_path,
            'score_df': score_df,
        }
        
        version = getattr(self, 'VERSION', 'V119')
        logger.info("=" * 70)
        logger.info(f"{version} Audit Complete - Status: {'PASSED ✓' if passed else 'FAILED ✗'}")
        logger.info(f"  T+1 IC: {t1_ic['mean_ic']:.4f} (target > {self.IC_THRESHOLD})")
        logger.info(f"  IC IR: {t1_ic['ic_ir']:.2f} (target > {self.IC_IR_THRESHOLD})")
        logger.info(f"  IC Decay: {ic_decay['decay_pattern']}")
        logger.info("=" * 70)
        
        return result
    
    def generate_report(
        self,
        t1_ic: dict,
        ic_decay: dict,
        backtest_result: dict,
        factor_ics: dict,
        passed: bool,
    ) -> str:
        """
        生成审计报告。
        
        Args:
            t1_ic: T+1 IC 统计
            ic_decay: IC Decay 分析
            backtest_result: 回测结果
            factor_ics: 因子 IC 字典
            passed: 是否通过验收
            
        Returns:
            报告文件路径
        """
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        report_path = self.output_dir / f"{self.VERSION.lower()}_audit_{timestamp}.md"
        
        # 提取回测指标
        total_return = backtest_result.get('total_return', 0)
        annual_return = backtest_result.get('annual_return', 0)
        sharpe = backtest_result.get('sharpe_ratio', 0)
        max_dd = backtest_result.get('max_drawdown', 0)
        total_cost = backtest_result.get('total_transaction_cost', 0)
        
        # 生成 Markdown 报告
        report_content = f"""# V103 Backtest Referee Audit Report

**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Alpha Module**: {type(self.alpha_module).__name__}

---

## 1. Alpha Prediction Metrics (核心指标)

| Metric | Value | Threshold | Status |
|--------|-------|-----------|--------|
| T+1 Rank IC (Mean) | {t1_ic.get('mean_ic', 0):.4f} | > {self.IC_THRESHOLD} | {'✓' if t1_ic.get('mean_ic', 0) > self.IC_THRESHOLD else '✗'} |
| IC IR (Stability) | {t1_ic.get('ic_ir', 0):.2f} | > {self.IC_IR_THRESHOLD} | {'✓' if t1_ic.get('ic_ir', 0) > self.IC_IR_THRESHOLD else '✗'} |
| IC Std | {t1_ic.get('ic_std', 0):.4f} | - | - |
| Num Trading Days | {t1_ic.get('num_days', 0)} | - | - |

### IC Decay Analysis (IC 衰减)

| Horizon | IC | Pattern |
|---------|-----|---------|
| T+1 | {ic_decay.get('t1_ic', 0):.4f} | Baseline |
| T+3 | {ic_decay.get('t3_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t1_ic', 0) >= ic_decay.get('t3_ic', 0) else '✗ Non-monotonic'} |
| T+5 | {ic_decay.get('t5_ic', 0):.4f} | {'✓ Monotonic' if ic_decay.get('t3_ic', 0) >= ic_decay.get('t5_ic', 0) else '✗ Non-monotonic'} |

**Decay Pattern**: {ic_decay.get('decay_pattern', 'N/A')}
**Monotonic Check**: {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED - Possible look-ahead bias'}

---

## 2. Backtest Performance (回测表现)

| Metric | Value |
|--------|-------|
| Initial Capital | {self.INITIAL_CAPITAL:,.0f} |
| Final Value | {backtest_result.get('final_value', 0):,.2f} |
| Total Return | {total_return:.2%} |
| Annual Return | {annual_return:.2%} |
| Sharpe Ratio | {sharpe:.2f} |
| Max Drawdown | {max_dd:.2%} |
| Volatility (Ann.) | {backtest_result.get('volatility', 0):.2%} |
| Trading Days | {backtest_result.get('num_trading_days', 0)} |

---

## 3. Transaction Cost Analysis (交易成本)

| Cost Type | Rate | Description |
|-----------|------|-------------|
| Commission | {self.COMMISSION_RATE:.2%} | Buy + Sell |
| Stamp Duty | {self.STAMP_DUTY_RATE:.2%} | Sell only |
| Slippage | {self.SLIPPAGE_RATE:.2%} | Buy + Sell |
| **Total Cost** | - | {total_cost:,.2f} |

---

## 4. Factor IC Analysis (因子 IC 分析)

"""
        
        # 添加因子 IC 表格
        if factor_ics:
            report_content += """| Factor | IC | Status |
|--------|-----|--------|
"""
            for factor_name, ic in sorted(factor_ics.items(), key=lambda x: abs(x[1]), reverse=True):
                status = '✓' if abs(ic) > self.TOP_FACTOR_IC_THRESHOLD else '✗'
                report_content += f"| {factor_name} | {ic:.4f} | {status} |\n"
        else:
            report_content += "*No factor IC data available*\n"
        
        report_content += f"""
---

## 5. Configuration (配置参数)

| Parameter | Value |
|-----------|-------|
| Top N Stocks | {self.TOP_N} |
| Position per Stock | {self.POSITION_PER_STOCK:.1%} |
| Commission Rate | {self.COMMISSION_RATE:.2%} |
| Stamp Duty Rate | {self.STAMP_DUTY_RATE:.2%} |
| Slippage Rate | {self.SLIPPAGE_RATE:.2%} |

---

## 6. Acceptance Criteria (验收标准)

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| T+1 Rank IC | > {self.IC_THRESHOLD} | {t1_ic.get('mean_ic', 0):.4f} | {'✓ PASSED' if t1_ic.get('mean_ic', 0) > self.IC_THRESHOLD else '✗ FAILED'} |
| IC IR | > {self.IC_IR_THRESHOLD} | {t1_ic.get('ic_ir', 0):.2f} | {'✓ PASSED' if t1_ic.get('ic_ir', 0) > self.IC_IR_THRESHOLD else '✗ FAILED'} |
| IC Decay | Monotonic | {ic_decay.get('decay_pattern', 'N/A')} | {'✓ PASSED' if ic_decay.get('is_monotonic', False) else '✗ FAILED'} |

---

## 7. Conclusion (结论)

### Overall Assessment

**{'PASSED ✓' if passed else 'FAILED ✗'}**

{f'The alpha module demonstrated strong predictive power with T+1 IC of {t1_ic.get("mean_ic", 0):.4f} and proper IC decay pattern.' if passed else 'The alpha module needs further optimization. Key issues:'}
{'' if passed else '- IC below threshold' if t1_ic.get('mean_ic', 0) <= self.IC_THRESHOLD else ''}
{'' if passed else '- IC IR below threshold' if t1_ic.get('ic_ir', 0) <= self.IC_IR_THRESHOLD else ''}
{'' if passed else '- Non-monotonic IC decay (possible look-ahead bias)' if not ic_decay.get('is_monotonic', False) else ''}

---

*Report generated by V103 Backtest Referee Module (Immutable)*
"""
        
        # 保存报告
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Report saved to: {report_path}")
        
        # 同时保存 JSON 结果
        json_result = {
            'alpha_metrics': {
                't1_ic': t1_ic,
                'ic_decay': ic_decay,
                'passed': passed,
            },
            'backtest_metrics': backtest_result,
            'factor_ics': factor_ics,
            'config': {
                'commission_rate': self.COMMISSION_RATE,
                'stamp_duty_rate': self.STAMP_DUTY_RATE,
                'slippage_rate': self.SLIPPAGE_RATE,
                'top_n': self.TOP_N,
                'position_per_stock': self.POSITION_PER_STOCK,
                'initial_capital': self.INITIAL_CAPITAL,
            },
        }
        
        json_path = self.output_dir / f"{self.VERSION.lower()}_audit_{timestamp}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_result, f, indent=2, default=str)
        
        logger.info(f"JSON result saved to: {json_path}")
        
        return str(report_path)


def get_backtest_referee(alpha_module: Any, output_dir: str = 'reports') -> BacktestReferee:
    """获取 BacktestReferee 实例。"""
    return BacktestReferee(alpha_module, output_dir)
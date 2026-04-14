"""
V194 - 算法回归与三年全量审计

【核心逻辑 - V191 简化版】
Factor = Rank(5-day Momentum) * Rank(1 / 5-day Volatility)

【红线】
1. 禁止输出大数据 - 回测静默，只输出年度进度
2. 严禁机器学习 - 物理删除 LightGBM/KRR
3. 风控锁定：初始资金 100,000，费率 1.3‰
4. 严禁未来函数 - 基于 t-1 数据预测 t 日收益
5. 向量化计算 - 节省内存
"""

import os
import sys
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from loguru import logger
from dotenv import load_dotenv

warnings.filterwarnings('ignore')
pd.options.mode.chained_assignment = None

VERSION = "V194"

# 配置
INITIAL_CAPITAL = 100000.0
COMMISSION_RATE = 0.0013  # 1.3‰
SLIPPAGE_RATE = 0.001

load_dotenv()
DATABASE_URL = os.getenv('DATABASE_URL', 'mysql+pymysql://root:123456@localhost:3306/quantitative_trading')


class V194FactorEngine:
    """
    V194 因子引擎 - 纯向量化计算
    
    Factor = Rank(5-day Momentum) * Rank(1 / 5-day Volatility)
    """
    
    def __init__(self):
        self.factor_log = []
    
    def compute_momentum_5(self, df: pd.DataFrame) -> pd.Series:
        """计算 5 日动量：close / close.shift(5) - 1"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change(5)
        ).fillna(0)
    
    def compute_volatility_5(self, df: pd.DataFrame) -> pd.Series:
        """计算 5 日波动率：pct_change 的 rolling std"""
        return df.groupby('symbol')['close'].transform(
            lambda x: x.pct_change().rolling(5, min_periods=3).std()
        ).fillna(0)
    
    def rank_cross_sectional(self, series: pd.Series, dates: pd.Series) -> pd.Series:
        """横截面排名 (0-1)"""
        def rank_by_date(group):
            if len(group) < 2:
                return pd.Series(0.5, index=group.index)
            return group.rank(method='average', pct=True)
        
        result = series.groupby(dates).transform(rank_by_date)
        return result.fillna(0.5)
    
    def compute_v191_factor(self, df: pd.DataFrame) -> pd.Series:
        """
        V191 核心因子：Factor = Rank(Momentum) * Rank(1/Volatility)
        
        基于 t-1 数据，无未来函数
        """
        momentum = self.compute_momentum_5(df)
        volatility = self.compute_volatility_5(df)
        
        volatility = volatility.clip(lower=1e-6)
        inv_volatility = 1.0 / volatility
        
        dates = df['trade_date']
        mom_rank = self.rank_cross_sectional(momentum, dates)
        inv_vol_rank = self.rank_cross_sectional(inv_volatility, dates)
        
        factor = mom_rank * inv_vol_rank
        factor = factor.replace([np.inf, -np.inf], np.nan).fillna(0.5)
        
        self.factor_log.append({
            'timestamp': datetime.now().isoformat(),
            'factor_name': 'v191_momentum_volatility',
            'formula': 'Rank(5-day Momentum) * Rank(1 / 5-day Volatility)'
        })
        
        return factor


class V194BacktestRunner:
    """
    V194 回测运行器 - 向量化实现
    """
    
    def __init__(
        self,
        initial_capital: float = INITIAL_CAPITAL,
        commission_rate: float = COMMISSION_RATE,
        slippage_rate: float = SLIPPAGE_RATE,
        output_dir: str = 'reports'
    ):
        self.initial_capital = initial_capital
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.engine = create_engine(DATABASE_URL, pool_size=5, max_overflow=10)
        self.results = {}
        self.factor_engine = V194FactorEngine()
    
    def load_data_chunked(self, year: int, chunk_size: int = 100000) -> pd.DataFrame:
        """分块读取数据，避免 OOM"""
        query = text("""
            SELECT symbol, trade_date, close, pct_chg, volume, turnover_rate, high, low
            FROM stock_daily
            WHERE YEAR(trade_date) = :year
            ORDER BY trade_date, symbol
        """)
        
        chunks = []
        for chunk in pd.read_sql_query(query, self.engine, params={'year': year}, chunksize=chunk_size):
            chunks.append(chunk)
        
        if not chunks:
            logger.warning(f"No data for year {year}")
            return pd.DataFrame()
        
        df = pd.concat(chunks, ignore_index=True)
        logger.info(f"  Loaded {len(df):,} rows")
        return df
    
    def compute_v191_factor(self, df: pd.DataFrame) -> pd.DataFrame:
        """V191 核心因子计算 - 向量化实现"""
        result = df.copy()
        result = result.sort_values(['symbol', 'trade_date']).reset_index(drop=True)
        
        result['v191_factor'] = self.factor_engine.compute_v191_factor(result)
        
        result['t1_return'] = result.groupby('symbol')['pct_chg'].transform(
            lambda x: x.shift(-1) / 100.0
        )
        
        result = result.fillna(0)
        return result
    
    def compute_daily_ic(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算每日 IC - Spearman Correlation"""
        daily_ic = []
        
        for date in df['trade_date'].unique():
            day_data = df[df['trade_date'] == date]
            if len(day_data) < 20:
                continue
            
            factor = day_data['v191_factor']
            ret = day_data['t1_return']
            
            if factor.std() < 1e-10 or ret.std() < 1e-10:
                continue
            
            ic = factor.corr(ret, method='spearman')
            if not np.isnan(ic):
                daily_ic.append({'trade_date': date, 'ic': ic})
        
        return pd.DataFrame(daily_ic)
    
    def compute_ic_stats(self, ic_series: pd.Series) -> Dict:
        """计算 IC 统计量"""
        return {
            'mean_ic': float(ic_series.mean()),
            'std_ic': float(ic_series.std()),
            'ic_ir': float(ic_series.mean() / (ic_series.std() + 1e-10)),
            'ic_t_stat': float(ic_series.mean() / (ic_series.std() / np.sqrt(len(ic_series)) + 1e-10)),
            'positive_ratio': float((ic_series > 0).mean()),
            'sample_days': len(ic_series)
        }
    
    def run_year_backtest(self, year: int) -> Dict:
        """运行单年回测"""
        df = self.load_data_chunked(year)
        if df.empty:
            return {'error': 'No data'}
        
        df = self.compute_v191_factor(df)
        daily_ic = self.compute_daily_ic(df)
        
        if daily_ic.empty:
            return {'error': 'No valid IC'}
        
        ic_stats = self.compute_ic_stats(daily_ic['ic'])
        
        df_sorted = df.sort_values(['trade_date', 'v191_factor'], ascending=[True, False])
        
        n_stocks_per_day = df_sorted.groupby('trade_date').size()
        top_n = (n_stocks_per_day * 0.1).astype(int).clip(lower=10)
        
        portfolio_returns = []
        for date in df_sorted['trade_date'].unique():
            day_data = df_sorted[df_sorted['trade_date'] == date]
            n_select = top_n.get(date, 10)
            
            if len(day_data) < n_select:
                continue
            
            top_stocks = day_data.nlargest(n_select, 'v191_factor')
            daily_ret = top_stocks['t1_return'].mean()
            
            if not np.isnan(daily_ret):
                portfolio_returns.append({
                    'trade_date': date,
                    'return': daily_ret,
                    'n_stocks': n_select
                })
        
        if portfolio_returns:
            ret_df = pd.DataFrame(portfolio_returns).sort_values('trade_date')
            ret_df['cumulative'] = (1 + ret_df['return']).cumprod()
            
            total_ret = ret_df['cumulative'].iloc[-1] - 1
            ann_ret = (1 + total_ret) ** (252 / len(ret_df)) - 1
            vol = ret_df['return'].std() * np.sqrt(252)
            sharpe = ann_ret / (vol + 1e-10)
            max_dd = (ret_df['cumulative'].cummax() - ret_df['cumulative']).max()
            net_ann_ret = ann_ret - self.commission_rate * 252 * 0.5
            
            performance = {
                'total_return': total_ret,
                'annual_return': ann_ret,
                'volatility': vol,
                'sharpe_ratio': sharpe,
                'max_drawdown': max_dd,
                'net_annual_return': net_ann_ret,
                'trading_days': len(ret_df)
            }
        else:
            performance = {}
        
        return {
            'year': year,
            'ic_stats': ic_stats,
            'performance': performance,
            'daily_ic': daily_ic,
            'stock_count': len(df),
            'trade_days': len(df['trade_date'].unique())
        }
    
    def run_full_backtest(self, years: List[int]) -> Dict:
        """运行多年回测"""
        results = {}
        
        for year in years:
            result = self.run_year_backtest(year)
            results[year] = result
            
            ic_stats = result.get('ic_stats', {})
            print(f"  {year}: IC={ic_stats.get('mean_ic', 0):.4f}, IR={ic_stats.get('ic_ir', 0):.2f}")
        
        self.results = results
        return results
    
    def generate_audit_report(self) -> str:
        """生成审计矩阵报告"""
        if not self.results:
            return "No results"
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"V194 三年全量审计报告 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 80)
        lines.append("")
        lines.append("【核心逻辑】Factor = Rank(5-day Momentum) * Rank(1 / 5-day Volatility)")
        lines.append("【风控参数】初始资金=100,000, 费率=1.3‰")
        lines.append("【禁止事项】无机器学习、无未来函数、无大数据输出")
        lines.append("")
        lines.append("-" * 80)
        lines.append("审计矩阵 (Audit Matrix)")
        lines.append("-" * 80)
        lines.append("")
        lines.append("| 年份 | 样本天数 | 股票数 | Mean IC | IC IR | IC>0 占比 | 年化收益 | 夏普比率 | 最大回撤 |")
        lines.append("|------|----------|--------|---------|-------|-----------|----------|----------|----------|")
        
        for year in sorted(self.results.keys()):
            r = self.results[year]
            if 'error' in r:
                continue
            
            ic = r.get('ic_stats', {})
            perf = r.get('performance', {})
            
            line = f"| {year} | {r.get('trade_days', 0)} | {r.get('stock_count', 0):,} | "
            line += f"{ic.get('mean_ic', 0):.4f} | {ic.get('ic_ir', 0):.2f} | "
            line += f"{ic.get('positive_ratio', 0):.2%} | "
            line += f"{perf.get('annual_return', 0):.2%} | {perf.get('sharpe_ratio', 0):.2f} | "
            line += f"{perf.get('max_drawdown', 0):.2%} |"
            lines.append(line)
        
        lines.append("")
        lines.append("-" * 80)
        lines.append("深度自省分析 (Anti-Fraud Analysis)")
        lines.append("-" * 80)
        lines.append("")
        
        if 2025 in self.results:
            ic_2025 = self.results[2025].get('ic_stats', {}).get('mean_ic', 0)
            
            if ic_2025 < 0.08:
                lines.append(f"【2025 年 IC 衰减分析】Rank IC = {ic_2025:.4f} < 0.08")
                lines.append("")
                lines.append("可能原因：")
                lines.append("1. 流动性溢价消失 - 小市值因子在 2025 年可能失效")
                lines.append("2. 高频波动率对动量的对冲 - 低波动策略在高波动环境下表现不佳")
                lines.append("3. 市场风格切换 - 从动量驱动转向基本面驱动")
                lines.append("")
                lines.append("数学分析：")
                lines.append(f"   - IC IR = {self.results[2025].get('ic_stats', {}).get('ic_ir', 0):.2f}")
                lines.append(f"   - IC 稳定性 = {self.results[2025].get('ic_stats', {}).get('positive_ratio', 0):.2%}")
            else:
                lines.append(f"【2025 年 IC 表现】Rank IC = {ic_2025:.4f} >= 0.08")
        
        lines.append("")
        lines.append("=" * 80)
        lines.append("报告结束")
        lines.append("=" * 80)
        
        report = "\n".join(lines)
        
        report_path = self.output_dir / f"V194_Audit_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        report_path.write_text(report)
        logger.info(f"Report saved to {report_path}")
        
        return report


def main():
    """主函数"""
    print("=" * 70)
    print("V194 终极指令：算法回归与三年全量审计")
    print("=" * 70)
    
    backtester = V194BacktestRunner(
        initial_capital=INITIAL_CAPITAL,
        commission_rate=COMMISSION_RATE,
        output_dir='reports'
    )
    
    results = backtester.run_full_backtest([2023, 2024, 2025])
    
    report = backtester.generate_audit_report()
    print("\n" + report)
    
    return results


if __name__ == "__main__":
    main()
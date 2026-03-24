"""
V62 vs V70 选股能力对比分析报告

【对比目的】
1. 纯粹的选股能力对比（不涉及离场逻辑）
2. 比较 V62 (RS-Pullback) 和 V70 (SNR 驱动) 的信号质量
3. 分析两个版本的 Rank IC、SNR、胜率等核心指标

【对比维度】
1. 信号数量：每日生成的信号数量
2. 信号质量：Rank IC、SNR 等指标
3. 选股胜率：信号后 5 日上涨概率
4. 行业分布：信号在各行业的分布
5. 市值分布：信号在不同市值区间的分布

作者：量化系统
版本：V70.0
日期：2026-03-24
"""

import sys
import os
import json
import traceback
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
import numpy as np
import polars as pl
from loguru import logger

# 尝试导入数据库管理器
try:
    from db_manager import DatabaseManager, get_db
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False
    logger.error("数据库模块未找到")

# 尝试导入 V62 和 V70 模块
try:
    from v62_core import (
        V62DataManager, V62AlphaCenter, V62Signal,
        V62_INITIAL_CAPITAL, V62_MAX_POSITIONS,
    )
    V62_AVAILABLE = True
except ImportError:
    V62_AVAILABLE = False
    logger.error("V62 模块未找到")

try:
    from v70_core import (
        V70DataManager, V70AlphaCenter, V70Signal,
        V70_INITIAL_CAPITAL, V70_MAX_POSITIONS,
        V70_SNR_MIN, V70_RANK_IC_TARGET,
    )
    V70_AVAILABLE = True
except ImportError:
    V70_AVAILABLE = False
    logger.error("V70 模块未找到")


# ===========================================
# 配置常量
# ===========================================

REPORT_DIR = "reports"
COMPARISON_CONFIG = {
    'start_date': '2024-01-01',
    'end_date': '2024-12-31',
    'forward_return_window': 5,  # 5 日远期收益率
    'min_signals_per_day': 1,    # 每日最少信号数
}


# ===========================================
# 数据类定义
# ===========================================

@dataclass
class SignalStatistics:
    """信号统计信息"""
    version: str
    total_signals: int = 0
    avg_signals_per_day: float = 0.0
    unique_stocks: int = 0
    avg_signal_score: float = 0.0
    signal_score_std: float = 0.0


@dataclass
class QualityMetrics:
    """质量指标"""
    version: str
    mean_rank_ic: float = 0.0
    rank_ic_std: float = 0.0
    rank_ic_ir: float = 0.0
    mean_snr: float = 0.0
    snr_pass_ratio: float = 0.0
    signal_win_rate: float = 0.0
    avg_forward_return: float = 0.0
    forward_return_sharpe: float = 0.0


@dataclass
class DistributionAnalysis:
    """分布分析"""
    version: str
    industry_distribution: Dict[str, float] = None
    market_cap_distribution: Dict[str, float] = None
    signal_by_month: Dict[str, int] = None


@dataclass
class ComparisonResult:
    """对比结果"""
    v62_metrics: QualityMetrics = None
    v70_metrics: QualityMetrics = None
    v62_stats: SignalStatistics = None
    v70_stats: SignalStatistics = None
    winner: str = ""
    summary: str = ""


# ===========================================
# 信号分析器
# ===========================================

class SignalAnalyzer:
    """
    信号分析器 - 计算信号质量和分布
    """
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self.forward_return_window = 5
    
    def calculate_forward_returns(self, df: pl.DataFrame) -> pl.DataFrame:
        """计算远期收益率"""
        result = df.clone()
        
        # 计算 5 日远期收益率
        result = result.with_columns([
            ((pl.col('close').shift(-self.forward_return_window)).over('symbol') - pl.col('close')) / 
            (pl.col('close') + 1e-9)
        ].alias('forward_return_5d'))
        
        return result
    
    def calculate_rank_ic(self, df: pl.DataFrame, 
                          signal_col: str = 'composite_score',
                          return_col: str = 'forward_return_5d') -> List[Dict[str, Any]]:
        """计算每日 Rank IC"""
        ic_results = []
        
        unique_dates = sorted(df['trade_date'].unique().to_list())
        
        for trade_date in unique_dates:
            day_data = df.filter(pl.col('trade_date') == trade_date)
            
            if day_data.height < 10:
                continue
            
            if signal_col not in day_data.columns or return_col not in day_data.columns:
                continue
            
            signal_values = day_data[signal_col].to_numpy()
            return_values = day_data[return_col].to_numpy()
            
            # 计算 Rank IC
            rank_ic = self._spearman_rank_ic(signal_values, return_values)
            
            # 计算普通 IC
            ic = np.corrcoef(signal_values, return_values)[0, 1]
            ic = float(ic) if not np.isnan(ic) else 0.0
            
            ic_results.append({
                'trade_date': trade_date,
                'ic': ic,
                'rank_ic': rank_ic,
                'count': len(signal_values),
            })
        
        return ic_results
    
    def _spearman_rank_ic(self, factor_values: np.ndarray,
                          label_values: np.ndarray) -> float:
        """计算 Spearman Rank IC"""
        mask = ~np.isnan(factor_values) & ~np.isnan(label_values)
        factor_clean = factor_values[mask]
        label_clean = label_values[mask]
        
        if len(factor_clean) < 10:
            return 0.0
        
        # 计算排名
        factor_ranks = np.argsort(np.argsort(factor_clean)).astype(float) + 1
        label_ranks = np.argsort(np.argsort(label_clean)).astype(float) + 1
        
        if np.std(factor_ranks) < 1e-9 or np.std(label_ranks) < 1e-9:
            return 0.0
        
        # 计算相关系数
        correlation = np.corrcoef(factor_ranks, label_ranks)[0, 1]
        
        return float(correlation) if not np.isnan(correlation) else 0.0
    
    def calculate_win_rate(self, df: pl.DataFrame,
                           signal_col: str = 'buy_signal',
                           return_col: str = 'forward_return_5d') -> Dict[str, Any]:
        """计算信号胜率"""
        signals_df = df.filter(pl.col(signal_col) == True)
        
        if signals_df.is_empty():
            return {'win_rate': 0.0, 'total_signals': 0, 'avg_return': 0.0}
        
        if return_col not in signals_df.columns:
            return {'win_rate': 0.0, 'total_signals': 0, 'avg_return': 0.0}
        
        returns = signals_df[return_col].drop_nulls().to_numpy()
        
        if len(returns) == 0:
            return {'win_rate': 0.0, 'total_signals': 0, 'avg_return': 0.0}
        
        wins = np.sum(returns > 0)
        win_rate = wins / len(returns)
        avg_return = np.mean(returns)
        return_std = np.std(returns, ddof=1) if len(returns) > 1 else 0.0
        sharpe = avg_return / return_std if return_std > 1e-9 else 0.0
        
        return {
            'win_rate': win_rate,
            'total_signals': len(returns),
            'avg_return': avg_return,
            'return_std': return_std,
            'sharpe': sharpe,
        }
    
    def calculate_industry_distribution(self, df: pl.DataFrame,
                                         signal_col: str = 'buy_signal') -> Dict[str, float]:
        """计算行业分布"""
        signals_df = df.filter(pl.col(signal_col) == True)
        
        if signals_df.is_empty() or 'industry_name' not in signals_df.columns:
            return {}
        
        industry_counts = signals_df.group_by('industry_name').agg(
            pl.count('symbol').alias('count')
        ).sort('count', descending=True)
        
        total = industry_counts['count'].sum()
        if total == 0:
            return {}
        
        distribution = {}
        for row in industry_counts.iter_rows(named=True):
            industry = row.get('industry_name', 'Unknown')
            count = row.get('count', 0)
            distribution[industry] = count / total
        
        return distribution
    
    def calculate_market_cap_distribution(self, df: pl.DataFrame,
                                           signal_col: str = 'buy_signal') -> Dict[str, float]:
        """计算市值分布"""
        signals_df = df.filter(pl.col(signal_col) == True)
        
        if signals_df.is_empty() or 'mv' not in signals_df.columns:
            return {}
        
        # 按市值分组
        signals_df = signals_df.with_columns([
            pl.when(pl.col('mv') < 50)
            .then(pl.lit('Small (<50B)'))
            .when(pl.col('mv') < 200)
            .then(pl.lit('Mid (50-200B)'))
            .when(pl.col('mv') < 500)
            .then(pl.lit('Large (200-500B)'))
            .otherwise(pl.lit('Huge (>500B)'))
        ].alias('mv_category'))
        
        mv_counts = signals_df.group_by('mv_category').agg(
            pl.count('symbol').alias('count')
        ).sort('mv_category')
        
        total = mv_counts['count'].sum()
        if total == 0:
            return {}
        
        distribution = {}
        for row in mv_counts.iter_rows(named=True):
            category = row.get('mv_category', 'Unknown')
            count = row.get('count', 0)
            distribution[category] = count / total
        
        return distribution


# ===========================================
# V62 分析器
# ===========================================

class V62SignalAnalyzer:
    """V62 信号分析器"""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self.signal_analyzer = SignalAnalyzer(db)
    
    def analyze(self, start_date: str, end_date: str) -> Tuple[SignalStatistics, QualityMetrics, DistributionAnalysis]:
        """分析 V62 信号"""
        logger.info("开始分析 V62 信号...")
        
        try:
            # 加载数据
            data_manager = V62DataManager(self.db)
            alpha_center = V62AlphaCenter()
            
            stock_df = data_manager.load_stock_data(start_date, end_date)
            index_df = data_manager.load_index_data(start_date, end_date)
            industry_df = data_manager.load_industry_data(start_date, end_date)
            
            # 计算信号
            result_df, status = alpha_center.compute_signals(stock_df, index_df)
            
            # 合并行业数据
            if not industry_df.is_empty():
                result_df = result_df.join(
                    industry_df.select(['symbol', 'trade_date', 'industry_name']),
                    on=['symbol', 'trade_date'],
                    how='left'
                )
            
            # 计算远期收益率
            result_df = self.signal_analyzer.calculate_forward_returns(result_df)
            
            # 计算统计信息
            stats = self._calculate_statistics(result_df)
            
            # 计算质量指标
            metrics = self._calculate_quality_metrics(result_df)
            
            # 计算分布分析
            distribution = self._calculate_distribution(result_df)
            
            logger.info(f"V62 分析完成：{stats.total_signals} 个信号")
            
            return stats, metrics, distribution
            
        except Exception as e:
            logger.error(f"V62 分析失败：{e}")
            logger.error(traceback.format_exc())
            return self._empty_result()
    
    def _calculate_statistics(self, df: pl.DataFrame) -> SignalStatistics:
        """计算统计信息"""
        buy_signals = df.filter(pl.col('buy_signal') == True)
        
        total_signals = buy_signals.height
        unique_stocks = buy_signals['symbol'].n_unique()
        unique_dates = buy_signals['trade_date'].n_unique()
        
        avg_signals_per_day = total_signals / unique_dates if unique_dates > 0 else 0.0
        avg_score = buy_signals['composite_score'].mean() if 'composite_score' in buy_signals.columns else 0.0
        score_std = buy_signals['composite_score'].std() if 'composite_score' in buy_signals.columns else 0.0
        
        return SignalStatistics(
            version='V62',
            total_signals=total_signals,
            avg_signals_per_day=avg_signals_per_day,
            unique_stocks=unique_stocks,
            avg_signal_score=float(avg_score) if avg_score else 0.0,
            signal_score_std=float(score_std) if score_std else 0.0,
        )
    
    def _calculate_quality_metrics(self, df: pl.DataFrame) -> QualityMetrics:
        """计算质量指标"""
        # Rank IC
        ic_results = self.signal_analyzer.calculate_rank_ic(df)
        
        if ic_results:
            rank_ics = [r['rank_ic'] for r in ic_results]
            mean_rank_ic = np.mean(rank_ics)
            rank_ic_std = np.std(rank_ics, ddof=1) if len(rank_ics) > 1 else 0.0
            rank_ic_ir = mean_rank_ic / rank_ic_std if rank_ic_std > 1e-9 else 0.0
        else:
            mean_rank_ic = 0.0
            rank_ic_std = 0.0
            rank_ic_ir = 0.0
        
        # 胜率
        win_rate_result = self.signal_analyzer.calculate_win_rate(df)
        
        # SNR (简化计算)
        snr_pass_ratio = 0.0
        
        return QualityMetrics(
            version='V62',
            mean_rank_ic=mean_rank_ic,
            rank_ic_std=rank_ic_std,
            rank_ic_ir=rank_ic_ir,
            mean_snr=0.0,
            snr_pass_ratio=snr_pass_ratio,
            signal_win_rate=win_rate_result['win_rate'],
            avg_forward_return=win_rate_result['avg_return'],
            forward_return_sharpe=win_rate_result['sharpe'],
        )
    
    def _calculate_distribution(self, df: pl.DataFrame) -> DistributionAnalysis:
        """计算分布分析"""
        industry_dist = self.signal_analyzer.calculate_industry_distribution(df)
        mv_dist = self.signal_analyzer.calculate_market_cap_distribution(df)
        
        # 按月统计
        buy_signals = df.filter(pl.col('buy_signal') == True)
        monthly_counts = {}
        if not buy_signals.is_empty() and 'trade_date' in buy_signals.columns:
            buy_signals = buy_signals.with_columns([
                pl.col('trade_date').str.slice(0, 7).alias('month')
            ])
            monthly = buy_signals.group_by('month').agg(pl.count('symbol').alias('count'))
            for row in monthly.iter_rows(named=True):
                monthly_counts[row['month']] = row['count']
        
        return DistributionAnalysis(
            version='V62',
            industry_distribution=industry_dist,
            market_cap_distribution=mv_dist,
            signal_by_month=monthly_counts,
        )
    
    def _empty_result(self) -> Tuple[SignalStatistics, QualityMetrics, DistributionAnalysis]:
        """返回空结果"""
        return (
            SignalStatistics(version='V62'),
            QualityMetrics(version='V62'),
            DistributionAnalysis(version='V62'),
        )


# ===========================================
# V70 分析器
# ===========================================

class V70SignalAnalyzer:
    """V70 信号分析器"""
    
    def __init__(self, db: Optional[DatabaseManager] = None):
        self.db = db
        self.signal_analyzer = SignalAnalyzer(db)
    
    def analyze(self, start_date: str, end_date: str) -> Tuple[SignalStatistics, QualityMetrics, DistributionAnalysis]:
        """分析 V70 信号"""
        logger.info("开始分析 V70 信号...")
        
        try:
            # 加载数据
            data_manager = V70DataManager(self.db)
            alpha_center = V70AlphaCenter()
            
            stock_df = data_manager.load_stock_data(start_date, end_date)
            fund_flow_df = data_manager.load_fund_flow_data(start_date, end_date)
            industry_df = data_manager.load_industry_data(start_date, end_date)
            
            # 计算信号
            result_df, status = alpha_center.compute_signals(
                stock_df, fund_flow_df, industry_df
            )
            
            # 计算统计信息
            stats = self._calculate_statistics(result_df)
            
            # 计算质量指标
            metrics = self._calculate_quality_metrics(result_df)
            
            # 计算分布分析
            distribution = self._calculate_distribution(result_df)
            
            logger.info(f"V70 分析完成：{stats.total_signals} 个信号")
            
            return stats, metrics, distribution
            
        except Exception as e:
            logger.error(f"V70 分析失败：{e}")
            logger.error(traceback.format_exc())
            return self._empty_result()
    
    def _calculate_statistics(self, df: pl.DataFrame) -> SignalStatistics:
        """计算统计信息"""
        buy_signals = df.filter(pl.col('buy_signal') == True)
        
        total_signals = buy_signals.height
        unique_stocks = buy_signals['symbol'].n_unique()
        unique_dates = buy_signals['trade_date'].n_unique()
        
        avg_signals_per_day = total_signals / unique_dates if unique_dates > 0 else 0.0
        avg_score = buy_signals['composite_score'].mean() if 'composite_score' in buy_signals.columns else 0.0
        score_std = buy_signals['composite_score'].std() if 'composite_score' in buy_signals.columns else 0.0
        
        return SignalStatistics(
            version='V70',
            total_signals=total_signals,
            avg_signals_per_day=avg_signals_per_day,
            unique_stocks=unique_stocks,
            avg_signal_score=float(avg_score) if avg_score else 0.0,
            signal_score_std=float(score_std) if score_std else 0.0,
        )
    
    def _calculate_quality_metrics(self, df: pl.DataFrame) -> QualityMetrics:
        """计算质量指标"""
        # Rank IC
        ic_results = self.signal_analyzer.calculate_rank_ic(df)
        
        if ic_results:
            rank_ics = [r['rank_ic'] for r in ic_results]
            mean_rank_ic = np.mean(rank_ics)
            rank_ic_std = np.std(rank_ics, ddof=1) if len(rank_ics) > 1 else 0.0
            rank_ic_ir = mean_rank_ic / rank_ic_std if rank_ic_std > 1e-9 else 0.0
        else:
            mean_rank_ic = 0.0
            rank_ic_std = 0.0
            rank_ic_ir = 0.0
        
        # 胜率
        win_rate_result = self.signal_analyzer.calculate_win_rate(df)
        
        # SNR
        snr_values = df['snr_value'].to_numpy() if 'snr_value' in df.columns else np.array([])
        snr_pass = df['snr_pass'].to_numpy() if 'snr_pass' in df.columns else np.array([])
        
        mean_snr = float(np.mean(snr_values)) if len(snr_values) > 0 else 0.0
        snr_pass_ratio = float(np.sum(snr_pass) / len(snr_pass)) if len(snr_pass) > 0 else 0.0
        
        return QualityMetrics(
            version='V70',
            mean_rank_ic=mean_rank_ic,
            rank_ic_std=rank_ic_std,
            rank_ic_ir=rank_ic_ir,
            mean_snr=mean_snr,
            snr_pass_ratio=snr_pass_ratio,
            signal_win_rate=win_rate_result['win_rate'],
            avg_forward_return=win_rate_result['avg_return'],
            forward_return_sharpe=win_rate_result['sharpe'],
        )
    
    def _calculate_distribution(self, df: pl.DataFrame) -> DistributionAnalysis:
        """计算分布分析"""
        industry_dist = self.signal_analyzer.calculate_industry_distribution(df)
        mv_dist = self.signal_analyzer.calculate_market_cap_distribution(df)
        
        # 按月统计
        buy_signals = df.filter(pl.col('buy_signal') == True)
        monthly_counts = {}
        if not buy_signals.is_empty() and 'trade_date' in buy_signals.columns:
            buy_signals = buy_signals.with_columns([
                pl.col('trade_date').str.slice(0, 7).alias('month')
            ])
            monthly = buy_signals.group_by('month').agg(pl.count('symbol').alias('count'))
            for row in monthly.iter_rows(named=True):
                monthly_counts[row['month']] = row['count']
        
        return DistributionAnalysis(
            version='V70',
            industry_distribution=industry_dist,
            market_cap_distribution=mv_dist,
            signal_by_month=monthly_counts,
        )
    
    def _empty_result(self) -> Tuple[SignalStatistics, QualityMetrics, DistributionAnalysis]:
        """返回空结果"""
        return (
            SignalStatistics(version='V70'),
            QualityMetrics(version='V70'),
            DistributionAnalysis(version='V70'),
        )


# ===========================================
# 对比报告生成器
# ===========================================

class ComparisonReporter:
    """对比报告生成器"""
    
    def __init__(self, output_dir: str = REPORT_DIR):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def generate_report(self, v62_result: Tuple, v70_result: Tuple,
                        start_date: str, end_date: str) -> str:
        """生成对比报告"""
        v62_stats, v62_metrics, v62_dist = v62_result
        v70_stats, v70_metrics, v70_dist = v70_result
        
        # 确定胜者
        winner = self._determine_winner(v62_metrics, v70_metrics)
        
        # 生成报告内容
        report = f"""# V62 vs V70 选股能力对比分析报告

## 报告信息
- 生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- 回测区间：[{start_date}, {end_date}]
- 对比维度：纯粹的选股能力（不涉及离场逻辑）

---

## 一、信号统计对比

| 指标 | V62 (RS-Pullback) | V70 (SNR 驱动) |
|------|-------------------|----------------|
| 总信号数 | {v62_stats.total_signals:,} | {v70_stats.total_signals:,} |
| 日均信号数 | {v62_stats.avg_signals_per_day:.2f} | {v70_stats.avg_signals_per_day:.2f} |
| 覆盖股票数 | {v62_stats.unique_stocks:,} | {v70_stats.unique_stocks:,} |
| 平均信号分 | {v62_stats.avg_signal_score:.4f} | {v70_stats.avg_signal_score:.4f} |
| 信号分标准差 | {v62_stats.signal_score_std:.4f} | {v70_stats.signal_score_std:.4f} |

---

## 二、信号质量对比（核心）

| 指标 | V62 (RS-Pullback) | V70 (SNR 驱动) | 达标要求 |
|------|-------------------|----------------|----------|
| Mean Rank IC | {v62_metrics.mean_rank_ic:.4f} | {v70_metrics.mean_rank_ic:.4f} | > 0.02 |
| Rank IC Std | {v62_metrics.rank_ic_std:.4f} | {v70_metrics.rank_ic_std:.4f} | - |
| Rank IC IR | {v62_metrics.rank_ic_ir:.2f} | {v70_metrics.rank_ic_ir:.2f} | > 0.5 |
| Mean SNR | {v62_metrics.mean_snr:.4f} | {v70_metrics.mean_snr:.4f} | > 0.15 |
| SNR 通过率 | {v62_metrics.snr_pass_ratio:.1%} | {v70_metrics.snr_pass_ratio:.1%} | > 80% |
| 信号胜率 | {v62_metrics.signal_win_rate:.1%} | {v70_metrics.signal_win_rate:.1%} | > 50% |
| 平均远期收益 | {v62_metrics.avg_forward_return:.2%} | {v70_metrics.avg_forward_return:.2%} | > 0 |
| 远期收益 Sharpe | {v62_metrics.forward_return_sharpe:.2f} | {v70_metrics.forward_return_sharpe:.2f} | > 0.5 |

---

## 三、行业分布对比

### V62 行业分布
{self._format_distribution(v62_dist.industry_distribution)}

### V70 行业分布
{self._format_distribution(v70_dist.industry_distribution)}

---

## 四、市值分布对比

### V62 市值分布
{self._format_distribution(v62_dist.market_cap_distribution)}

### V70 市值分布
{self._format_distribution(v70_dist.market_cap_distribution)}

---

## 五、月度信号数量对比

### V62 月度信号
{self._format_monthly_counts(v62_dist.signal_by_month)}

### V70 月度信号
{self._format_monthly_counts(v70_dist.signal_by_month)}

---

## 六、结论

### 胜者：{winner}

### 核心发现
1. **Rank IC 对比**: V62 ({v62_metrics.mean_rank_ic:.4f}) vs V70 ({v70_metrics.mean_rank_ic:.4f})
   - {'V70 的 Rank IC 更高，选股能力更强' if v70_metrics.mean_rank_ic > v62_metrics.mean_rank_ic else 'V62 的 Rank IC 更高，选股能力更强'}

2. **SNR 稳定性**: V62 ({v62_metrics.mean_snr:.4f}) vs V70 ({v70_metrics.mean_snr:.4f})
   - {'V70 的 SNR 更高，信号更稳定' if v70_metrics.mean_snr > v62_metrics.mean_snr else 'V62 的 SNR 更高，信号更稳定'}

3. **信号胜率**: V62 ({v62_metrics.signal_win_rate:.1%}) vs V70 ({v70_metrics.signal_win_rate:.1%})
   - {'V70 的胜率更高' if v70_metrics.signal_win_rate > v62_metrics.signal_win_rate else 'V62 的胜率更高'}

### 建议
{self._generate_recommendations(v62_metrics, v70_metrics)}

---

*本报告由 V70 对比分析器自动生成*
"""
        
        # 保存报告
        report_path = os.path.join(
            self.output_dir,
            f"V62_V70_Comparison_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        )
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        logger.info(f"报告已保存至：{report_path}")
        
        return report_path
    
    def _determine_winner(self, v62_metrics: QualityMetrics,
                          v70_metrics: QualityMetrics) -> str:
        """确定胜者"""
        v62_score = 0
        v70_score = 0
        
        # Rank IC 对比 (权重 40%)
        if v62_metrics.mean_rank_ic > v70_metrics.mean_rank_ic:
            v62_score += 40
        else:
            v70_score += 40
        
        # SNR 对比 (权重 30%)
        if v62_metrics.mean_snr > v70_metrics.mean_snr:
            v62_score += 30
        else:
            v70_score += 30
        
        # 胜率对比 (权重 30%)
        if v62_metrics.signal_win_rate > v70_metrics.signal_win_rate:
            v62_score += 30
        else:
            v70_score += 30
        
        if v62_score > v70_score:
            return f"V62 (得分：{v62_score} vs {v70_score})"
        elif v70_score > v62_score:
            return f"V70 (得分：{v70_score} vs {v62_score})"
        else:
            return "平局"
    
    def _format_distribution(self, dist: Dict[str, float]) -> str:
        """格式化分布数据"""
        if not dist:
            return "无数据"
        
        lines = []
        for key, value in sorted(dist.items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- {key}: {value:.1%}")
        return "\n".join(lines)
    
    def _format_monthly_counts(self, monthly: Dict[str, int]) -> str:
        """格式化月度计数"""
        if not monthly:
            return "无数据"
        
        lines = []
        for month, count in sorted(monthly.items()):
            lines.append(f"- {month}: {count:,} 个信号")
        return "\n".join(lines)
    
    def _generate_recommendations(self, v62_metrics: QualityMetrics,
                                   v70_metrics: QualityMetrics) -> str:
        """生成建议"""
        recommendations = []
        
        if v70_metrics.mean_rank_ic > v62_metrics.mean_rank_ic:
            recommendations.append("- V70 的 Rank IC 更高，建议优先使用 V70 进行选股")
        else:
            recommendations.append("- V62 的 Rank IC 更高，建议优先使用 V62 进行选股")
        
        if v70_metrics.mean_snr > 0.15:
            recommendations.append("- V70 的 SNR 达标 (>0.15)，信号稳定性良好")
        
        if v70_metrics.signal_win_rate > v62_metrics.signal_win_rate:
            recommendations.append("- V70 的胜率更高，可考虑增加 V70 的仓位权重")
        
        return "\n".join(recommendations) if recommendations else "- 两个版本表现接近，可根据实际情况选择"


# ===========================================
# 主程序
# ===========================================

def run_comparison(start_date: str = '2024-01-01',
                   end_date: str = '2024-12-31',
                   output_dir: str = REPORT_DIR) -> str:
    """运行对比分析"""
    logger.info("=" * 60)
    logger.info("V62 vs V70 选股能力对比分析")
    logger.info("=" * 60)
    
    if not DB_AVAILABLE:
        logger.error("数据库不可用")
        return ""
    
    if not V62_AVAILABLE:
        logger.error("V62 模块不可用")
        return ""
    
    if not V70_AVAILABLE:
        logger.error("V70 模块不可用")
        return ""
    
    try:
        db = get_db()
    except Exception as e:
        logger.error(f"数据库连接失败：{e}")
        return ""
    
    # 分析 V62
    v62_analyzer = V62SignalAnalyzer(db)
    v62_result = v62_analyzer.analyze(start_date, end_date)
    
    # 分析 V70
    v70_analyzer = V70SignalAnalyzer(db)
    v70_result = v70_analyzer.analyze(start_date, end_date)
    
    # 生成报告
    reporter = ComparisonReporter(output_dir)
    report_path = reporter.generate_report(v62_result, v70_result, start_date, end_date)
    
    logger.info("=" * 60)
    logger.info("对比分析完成")
    logger.info(f"报告路径：{report_path}")
    logger.info("=" * 60)
    
    return report_path


if __name__ == "__main__":
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO"
    )
    
    # 运行对比
    run_comparison()
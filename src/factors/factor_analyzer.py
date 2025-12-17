import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class FactorAnalyzer:
    """因子分析器，负责计算IC、ICIR、换手率等核心指标。"""
    
    def __init__(self, price_data: pd.DataFrame, forward_returns: pd.Series):
        """
        初始化分析器。
        
        Args:
            price_data: 必须包含'close', 'volume'列，索引为日期。
            forward_returns: 未来收益率序列，索引与price_data对齐。
        """
        self.price_data = price_data.copy()
        self.forward_returns = forward_returns.copy()
        self.factor_values = {}  # 存储因子名称 -> 因子值Series
        self.ic_series = {}      # 存储因子名称 -> IC时间序列（单标的时为占位）
        
    def calculate_factor(self, factor_func, **kwargs) -> pd.Series:
        """计算单个因子。"""
        return factor_func(self.price_data, **kwargs)
    
    def calculate_all_factors(self, factor_config: Dict) -> None:
        """批量计算所有配置的因子。"""
        logger.info(f"开始批量计算 {len(factor_config)} 个因子")
        
        for factor_name, config in factor_config.items():
            try:
                func = config['func']
                args = config.get('args', {})
                factor_series = self.calculate_factor(func, **args)
                
                if factor_series is not None and not factor_series.isnull().all():
                    self.factor_values[factor_name] = factor_series
                    logger.debug(f"因子计算成功: {factor_name}")
                else:
                    logger.warning(f"因子计算失败或结果全为空: {factor_name}")
            except Exception as e:
                logger.error(f"计算因子 {factor_name} 时出错: {e}")
    
    def _align_data(self, factor_series: pd.Series) -> Tuple[pd.Series, pd.Series]:
        """对齐因子值和未来收益，剔除缺失值。"""
        aligned = pd.DataFrame({
            'factor': factor_series,
            'forward_ret': self.forward_returns
        }).dropna()
        
        if len(aligned) < 10:  # 数据点太少无统计意义
            logger.warning("对齐后数据点不足10个，无法计算有效IC")
            return pd.Series(dtype=float), pd.Series(dtype=float)
        
        return aligned['factor'], aligned['forward_ret']
    
    def calculate_ic(self, factor_series: pd.Series) -> float:
        """
        计算信息系数IC。
        对于单标的，计算因子值与未来收益率的时间序列相关性。
        """
        factor_aligned, forward_aligned = self._align_data(factor_series)
        
        if len(factor_aligned) == 0:
            return np.nan
        
        # 计算Pearson相关系数
        ic = factor_aligned.corr(forward_aligned)
        return ic
    
    def calculate_rank_ic(self, factor_series: pd.Series) -> float:
        """计算Rank IC（斯皮尔曼相关系数）。"""
        factor_aligned, forward_aligned = self._align_data(factor_series)
        
        if len(factor_aligned) == 0:
            return np.nan
        
        # 计算Spearman秩相关系数
        rank_ic = factor_aligned.corr(forward_aligned, method='spearman')
        return rank_ic
    
    def calculate_ic_series(self, factor_series: pd.Series, window: int = 20) -> pd.Series:
        """
        计算滚动IC时间序列（用于单标的分析）。
        注意：这只是一种近似，真正的截面IC需要多股票数据。
        """
        ic_list = []
        dates = []
        
        # 使用滚动窗口计算相关性
        for i in range(window, len(factor_series)):
            factor_window = factor_series.iloc[i-window:i]
            forward_window = self.forward_returns.iloc[i-window:i]
            
            # 对齐数据
            aligned = pd.DataFrame({
                'factor': factor_window,
                'forward': forward_window
            }).dropna()
            
            if len(aligned) > 5:  # 至少需要5个点
                ic = aligned['factor'].corr(aligned['forward'])
                ic_list.append(ic)
                dates.append(factor_series.index[i])
        
        return pd.Series(ic_list, index=dates) if ic_list else pd.Series(dtype=float)
    
    def estimate_turnover(self, factor_series: pd.Series, period: int = 20) -> float:
        """
        估算因子换手率（针对单标的）。
        通过计算因子值在时间序列上的自相关系数来近似。
        自相关系数越低，表示因子值变化越快，隐含的换手率越高。
        """
        if len(factor_series) < period * 2:
            return np.nan
        
        # 计算因子值的一阶自相关系数
        autocorr = factor_series.autocorr(lag=1)
        
        # 自相关系数转换为换手率近似值 (0~1之间)
        # 自相关接近1表示高度持久（低换手），接近-1表示反转（高换手）
        turnover_approx = (1 - autocorr) / 2 if not np.isnan(autocorr) else 0.5
        return max(0, min(1, turnover_approx))  # 限制在[0,1]范围内
    
    def analyze_single_factor(self, factor_name: str, factor_series: pd.Series) -> Dict:
        """分析单个因子，返回所有关键指标。"""
        # 计算IC
        ic = self.calculate_ic(factor_series)
        rank_ic = self.calculate_rank_ic(factor_series)
        
        # 计算IC时间序列和ICIR
        ic_series = self.calculate_ic_series(factor_series, window=60)
        if len(ic_series) > 0:
            ic_mean = ic_series.mean()
            ic_std = ic_series.std()
            icir = ic_mean / ic_std if ic_std != 0 else np.nan
        else:
            ic_mean, ic_std, icir = np.nan, np.nan, np.nan
        
        # 估算换手率
        turnover = self.estimate_turnover(factor_series)
        
        return {
            'factor_name': factor_name,
            'IC': round(ic, 4) if not np.isnan(ic) else np.nan,
            'Rank_IC': round(rank_ic, 4) if not np.isnan(rank_ic) else np.nan,
            'IC_mean': round(ic_mean, 4) if not np.isnan(ic_mean) else np.nan,
            'IC_std': round(ic_std, 4) if not np.isnan(ic_std) else np.nan,
            'ICIR': round(icir, 4) if not np.isnan(icir) else np.nan,
            'turnover': round(turnover, 4) if not np.isnan(turnover) else np.nan,
            'obs_count': len(factor_series.dropna())
        }
    
    def analyze_all_factors(self) -> pd.DataFrame:
        """分析所有已计算的因子，生成汇总报告。"""
        if not self.factor_values:
            logger.warning("没有可分析的因子数据")
            return pd.DataFrame()
        
        results = []
        logger.info(f"开始分析 {len(self.factor_values)} 个因子")
        
        for factor_name, factor_series in self.factor_values.items():
            try:
                result = self.analyze_single_factor(factor_name, factor_series)
                results.append(result)
                logger.info(f"分析完成: {factor_name} - IC={result['IC']}, ICIR={result['ICIR']}")
            except Exception as e:
                logger.error(f"分析因子 {factor_name} 时出错: {e}")
                continue
        
        if not results:
            return pd.DataFrame()
        
        # 创建DataFrame并排序
        results_df = pd.DataFrame(results)
        
        # 按IC绝对值降序排序
        results_df['IC_abs'] = results_df['IC'].abs()
        results_df = results_df.sort_values('IC_abs', ascending=False)
        results_df = results_df.drop('IC_abs', axis=1)
        
        logger.info(f"因子分析完成，共分析 {len(results_df)} 个有效因子")
        return results_df
    
    def generate_report(self, results_df: pd.DataFrame, output_path: str) -> None:
        """生成详细的因子分析报告。"""
        if results_df.empty:
            logger.warning("无分析结果，跳过报告生成")
            return
        
        report_lines = [
            "=" * 60,
            "因子有效性分析报告",
            "=" * 60,
            f"生成时间: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"分析标的: 单标的分析模式",
            f"数据期间: {self.price_data.index.min()} 至 {self.price_data.index.max()}",
            f"有效因子数量: {len(results_df)}",
            "",
        ]
        
        # 添加IC最高的前5个因子
        top_factors = results_df.head(5)
        report_lines.extend(["IC最高的5个因子:", "-" * 40])
        for _, row in top_factors.iterrows():
            report_lines.append(
                f"{row['factor_name']:20s} | IC={row['IC']:6.4f} | "
                f"Rank_IC={row['Rank_IC']:6.4f} | ICIR={row['ICIR']:6.4f} | "
                f"换手率={row['turnover']:.2%}"
            )
        
        report_lines.extend(["", "完整因子分析结果:", "-" * 40])
        
        # 添加完整表格
        report_df = results_df.copy()
        report_df['turnover_pct'] = report_df['turnover'].apply(lambda x: f"{x:.2%}")
        
        # 选择要显示的列
        display_cols = ['factor_name', 'IC', 'Rank_IC', 'IC_mean', 'IC_std', 'ICIR', 'turnover_pct', 'obs_count']
        display_df = report_df[[c for c in display_cols if c in report_df.columns]]
        
        report_lines.append(display_df.to_string(index=False))
        
        # 添加总结
        report_lines.extend([
            "",
            "-" * 40,
            "报告总结:",
            f"1. 平均IC: {results_df['IC'].mean():.4f}",
            f"2. 平均ICIR: {results_df['ICIR'].mean():.4f}",
            f"3. IC>0的因子比例: {(results_df['IC'] > 0).mean():.2%}",
            f"4. Rank_IC>0的因子比例: {(results_df['Rank_IC'] > 0).mean():.2%}",
            "=" * 60
        ])
        
        # 写入文件
        report_content = "\n".join(report_lines)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"分析报告已保存至: {output_path}")
        
        # 同时在控制台输出摘要
        print("\n" + "=" * 60)
        print("因子分析摘要:")
        print("=" * 60)
        print(f"分析完成，共 {len(results_df)} 个因子")
        print(f"IC最高的因子: {results_df.iloc[0]['factor_name']} (IC={results_df.iloc[0]['IC']:.4f})")
        print(f"ICIR最高的因子: {results_df.loc[results_df['ICIR'].idxmax()]['factor_name']} "
              f"(ICIR={results_df['ICIR'].max():.4f})")
        print("=" * 60)
"""
V63 双回测对比脚本 - V62 vs V63

【脚本功能】
1. 同时运行 V62(回调版) 和 V63(收缩版) 回测
2. 生成对比报告
3. 因子贡献度审计
4. 分析 A 股市场风格适应性

作者：量化系统
版本：V63.0
日期：2026-03-23
"""

import sys
import json
from datetime import datetime
from pathlib import Path
from loguru import logger

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from db_manager import DatabaseManager
from v62_engine import run_v62_backtest, V62BacktestMetrics
from v63_engine import run_v63_backtest, V63BacktestMetrics


def configure_logging():
    """配置日志"""
    logger.remove()
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>",
        level="INFO"
    )


def metrics_to_dict(metrics) -> dict:
    """将指标转换为字典"""
    if hasattr(metrics, '__dict__'):
        return {k: v for k, v in vars(metrics).items() if not k.startswith('_')}
    return {}


def run_dual_backtest(start_date: str = "2024-01-01",
                      end_date: str = "2024-12-31",
                      initial_capital: float = 100000.0,
                      max_positions: int = 10):
    """
    运行双回测对比
    
    Parameters
    ----------
    start_date : str
        回测开始日期
    end_date : str
        回测结束日期
    initial_capital : float
        初始资金
    max_positions : int
        最大持仓数
    """
    configure_logging()
    
    logger.info("=" * 80)
    logger.info("V62 vs V63 双回测对比")
    logger.info("=" * 80)
    logger.info(f"回测区间：[{start_date}, {end_date}]")
    logger.info(f"初始资金：{initial_capital:,.2f}")
    logger.info(f"最大持仓：{max_positions}只")
    logger.info("=" * 80)
    
    # 初始化数据库
    db = DatabaseManager()
    
    results = {
        'v62': None,
        'v63': None,
        'comparison': {},
        'factor_audit': {},
    }
    
    # ========== 运行 V62 回测 ==========
    logger.info("\n" + "=" * 60)
    logger.info("【V62 RS-Pullback 回调版】回测开始")
    logger.info("=" * 60)
    
    try:
        v62_metrics = run_v62_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_positions=max_positions,
            db=db
        )
        results['v62'] = metrics_to_dict(v62_metrics)
        logger.info("V62 回测完成 ✅")
    except Exception as e:
        logger.error(f"V62 回测失败：{e}")
        results['v62'] = {'error': str(e)}
    
    # ========== 运行 V63 回测 ==========
    logger.info("\n" + "=" * 60)
    logger.info("【V63 VCP 波动率收缩版】回测开始")
    logger.info("=" * 60)
    
    try:
        v63_metrics = run_v63_backtest(
            start_date=start_date,
            end_date=end_date,
            initial_capital=initial_capital,
            max_positions=max_positions,
            db=db
        )
        results['v63'] = metrics_to_dict(v63_metrics)
        logger.info("V63 回测完成 ✅")
    except Exception as e:
        logger.error(f"V63 回测失败：{e}")
        results['v63'] = {'error': str(e)}
    
    # ========== 生成对比报告 ==========
    logger.info("\n" + "=" * 80)
    logger.info("V62 vs V63 对比报告")
    logger.info("=" * 80)
    
    comparison = generate_comparison_report(results['v62'], results['v63'])
    results['comparison'] = comparison
    
    # ========== 因子贡献度审计 ==========
    logger.info("\n" + "=" * 80)
    logger.info("因子贡献度审计")
    logger.info("=" * 80)
    
    factor_audit = generate_factor_audit(results['v62'], results['v63'])
    results['factor_audit'] = factor_audit
    
    # ========== 保存报告 ==========
    report_path = save_report(results, start_date, end_date)
    logger.info(f"\n报告已保存至：{report_path}")
    
    # ========== 打印最终总结 ==========
    print_final_summary(results)
    
    return results


def generate_comparison_report(v62: dict, v63: dict) -> dict:
    """生成对比报告"""
    comparison = {
        'metrics': {},
        'winner': '',
        'analysis': []
    }
    
    # 核心指标对比
    core_metrics = [
        'total_return',
        'annualized_return',
        'max_drawdown',
        'sharpe_ratio',
        'win_rate',
        'profit_loss_ratio',
        'total_trades',
        'avg_holding_days',
    ]
    
    for metric in core_metrics:
        v62_val = v62.get(metric, 0) if v62 else 0
        v63_val = v63.get(metric, 0) if v63 else 0
        
        if isinstance(v62_val, (int, float)) and isinstance(v63_val, (int, float)):
            diff = v63_val - v62_val
            diff_pct = (diff / abs(v62_val) * 100) if v62_val != 0 else 0
            
            comparison['metrics'][metric] = {
                'v62': v62_val,
                'v63': v63_val,
                'diff': diff,
                'diff_pct': diff_pct,
                'v63_better': is_better(metric, v63_val, v62_val)
            }
    
    # 打印对比表格
    print_comparison_table(comparison['metrics'])
    
    # 判断赢家
    v62_wins = sum(1 for m in comparison['metrics'].values() if not m['v63_better'])
    v63_wins = sum(1 for m in comparison['metrics'].values() if m['v63_better'])
    
    comparison['winner'] = 'V63' if v63_wins > v62_wins else 'V62'
    comparison['v62_wins'] = v62_wins
    comparison['v63_wins'] = v63_wins
    
    logger.info(f"\n核心指标胜出：V62 {v62_wins} 项，V63 {v63_wins} 项")
    logger.info(f"综合评判：{comparison['winner']} 胜出")
    
    return comparison


def is_better(metric: str, v63_val: float, v62_val: float) -> bool:
    """判断 V63 是否更好"""
    # 越大越好的指标
    higher_is_better = ['total_return', 'annualized_return', 'sharpe_ratio', 'win_rate', 'profit_loss_ratio']
    # 越小越好的指标
    lower_is_better = ['max_drawdown', 'total_trades', 'avg_holding_days']
    
    if metric in higher_is_better:
        return v63_val > v62_val
    elif metric in lower_is_better:
        return v63_val < v62_val
    return v63_val > v62_val


def print_comparison_table(metrics: dict):
    """打印对比表格"""
    logger.info("\n" + "-" * 70)
    logger.info(f"{'指标':<20} {'V62':>12} {'V63':>12} {'差异':>12} {'改善':>10}")
    logger.info("-" * 70)
    
    for name, data in metrics.items():
        v62_str = f"{data['v62']:.2%}" if 'return' in name or 'rate' in name or 'drawdown' in name else f"{data['v62']:.2f}"
        v63_str = f"{data['v63']:.2%}" if 'return' in name or 'rate' in name or 'drawdown' in name else f"{data['v63']:.2f}"
        diff_str = f"{data['diff_pct']:+.1f}%"
        better_str = "✅" if data['v63_better'] else "❌"
        
        logger.info(f"{name:<20} {v62_str:>12} {v63_str:>12} {diff_str:>12} {better_str:>10}")
    
    logger.info("-" * 70)


def generate_factor_audit(v62: dict, v63: dict) -> dict:
    """
    因子贡献度审计
    
    分析哪个因子（RS、VCP 还是大盘择时）拦截了灾难性亏损
    """
    audit = {
        'v62_loss_analysis': {},
        'v63_improvement': {},
        'factor_contribution': {},
        'conclusion': ''
    }
    
    if not v62 or not v63:
        audit['conclusion'] = "数据不足，无法进行因子贡献度审计"
        return audit
    
    # V62 亏损分析
    v62_return = v62.get('total_return', 0)
    v62_max_dd = v62.get('max_drawdown', 0)
    v62_trades = v62.get('total_trades', 0)
    v62_win_rate = v62.get('win_rate', 0)
    
    audit['v62_loss_analysis'] = {
        'total_return': v62_return,
        'max_drawdown': v62_max_dd,
        'total_trades': v62_trades,
        'win_rate': v62_win_rate,
        'is_disaster': v62_return < -0.10 or v62_max_dd > 0.25  # 亏损>10% 或回撤>25% 定义为灾难
    }
    
    # V63 改进分析
    v63_return = v63.get('total_return', 0)
    v63_max_dd = v63.get('max_drawdown', 0)
    v63_trades = v63.get('total_trades', 0)
    v63_win_rate = v63.get('win_rate', 0)
    v63_forced_empty_days = v63.get('forced_empty_days', 0)
    v63_time_stop_count = v63.get('time_stop_count', 0)
    
    audit['v63_improvement'] = {
        'return_improvement': v63_return - v62_return,
        'drawdown_reduction': v62_max_dd - v63_max_dd,
        'trade_reduction': v62_trades - v63_trades,
        'win_rate_improvement': v63_win_rate - v62_win_rate,
        'forced_empty_days': v63_forced_empty_days,
        'time_stop_count': v63_time_stop_count,
    }
    
    # 因子贡献度分析
    factor_contributions = []
    
    # 1. 大盘择时贡献
    if v63_forced_empty_days > 0:
        factor_contributions.append({
            'factor': '大盘择时熔断',
            'description': f'强制空仓{v63_forced_empty_days}天，避开市场大跌',
            'impact': '高' if v63_forced_empty_days > 20 else '中'
        })
    
    # 2. 时间止损贡献
    if v63_time_stop_count > 0:
        factor_contributions.append({
            'factor': '时间止损',
            'description': f'{v63_time_stop_count}次时间止损，避免阴跌亏损',
            'impact': '中' if v63_time_stop_count < 10 else '高'
        })
    
    # 3. VCP 收缩贡献
    if v63_trades < v62_trades:
        trade_reduction = v62_trades - v63_trades
        factor_contributions.append({
            'factor': 'VCP 波动率收缩',
            'description': f'减少{trade_reduction}次交易，提高信号质量',
            'impact': '高' if trade_reduction > 20 else '中'
        })
    
    # 4. 趋势过滤贡献
    if v63_win_rate > v62_win_rate:
        win_rate_improvement = v63_win_rate - v62_win_rate
        factor_contributions.append({
            'factor': '趋势对齐过滤',
            'description': f'胜率提升{win_rate_improvement*100:.1f}%',
            'impact': '高' if win_rate_improvement > 0.05 else '中'
        })
    
    audit['factor_contribution'] = {
        'factors': factor_contributions,
        'top_factor': factor_contributions[0]['factor'] if factor_contributions else '无',
        'total_factors': len(factor_contributions)
    }
    
    # 结论
    if audit['v62_loss_analysis']['is_disaster']:
        if audit['v63_improvement']['return_improvement'] > 0:
            audit['conclusion'] = f"✅ V63 成功拦截 V62 灾难性亏损！主要贡献因子：{audit['factor_contribution']['top_factor']}"
        else:
            audit['conclusion'] = "⚠️ V63 未能有效改善 V62 的灾难性亏损，需要进一步优化"
    else:
        if v63_return > v62_return:
            audit['conclusion'] = f"✅ V63 表现优于 V62，主要贡献因子：{audit['factor_contribution']['top_factor']}"
        else:
            audit['conclusion'] = "⚠️ V63 表现不如 V62，建议分析具体原因"
    
    # 打印审计结果
    logger.info("\n" + "-" * 70)
    logger.info("因子贡献度审计结果")
    logger.info("-" * 70)
    
    logger.info(f"\nV62 状态:")
    logger.info(f"  总收益：{v62_return*100:.2f}%")
    logger.info(f"  最大回撤：{v62_max_dd*100:.2f}%")
    logger.info(f"  交易次数：{v62_trades}")
    logger.info(f"  胜率：{v62_win_rate*100:.2f}%")
    logger.info(f"  是否灾难：{'是' if audit['v62_loss_analysis']['is_disaster'] else '否'}")
    
    logger.info(f"\nV63 改进:")
    logger.info(f"  收益改善：{audit['v63_improvement']['return_improvement']*100:+.2f}%")
    logger.info(f"  回撤降低：{audit['v63_improvement']['drawdown_reduction']*100:.2f}%")
    logger.info(f"  交易减少：{audit['v63_improvement']['trade_reduction']}次")
    logger.info(f"  胜率提升：{audit['v63_improvement']['win_rate_improvement']*100:+.2f}%")
    logger.info(f"  强制空仓：{audit['v63_improvement']['forced_empty_days']}天")
    logger.info(f"  时间止损：{audit['v63_improvement']['time_stop_count']}次")
    
    logger.info(f"\n因子贡献度排名:")
    for i, factor in enumerate(factor_contributions, 1):
        logger.info(f"  {i}. {factor['factor']} - {factor['description']} (影响：{factor['impact']})")
    
    logger.info(f"\n🎯 结论：{audit['conclusion']}")
    logger.info("-" * 70)
    
    return audit


def save_report(results: dict, start_date: str, end_date: str) -> Path:
    """保存报告"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = Path(__file__).parent.parent / 'reports'
    report_dir.mkdir(exist_ok=True)
    
    report_path = report_dir / f"V63_V62_Comparison_Report_{timestamp}.md"
    
    v62 = results.get('v62', {})
    v63 = results.get('v63', {})
    comparison = results.get('comparison', {})
    factor_audit = results.get('factor_audit', {})
    
    content = f"""# V62 vs V63 双回测对比报告

## 基本信息

- 回测区间：[{start_date}, {end_date}]
- 报告生成时间：{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## V62 RS-Pullback 回调版

| 指标 | 数值 |
|------|------|
| 总收益率 | {v62.get('total_return', 'N/A'):.2%} |
| 年化收益 | {v62.get('annualized_return', 'N/A'):.2%} |
| 最大回撤 | {v62.get('max_drawdown', 'N/A'):.2%} |
| 夏普比率 | {v62.get('sharpe_ratio', 'N/A'):.2f} |
| 胜率 | {v62.get('win_rate', 'N/A'):.2%} |
| 交易次数 | {v62.get('total_trades', 'N/A')} |
| 平均持仓天数 | {v62.get('avg_holding_days', 'N/A'):.1f} |

## V63 VCP 波动率收缩版

| 指标 | 数值 |
|------|------|
| 总收益率 | {v63.get('total_return', 'N/A'):.2%} |
| 年化收益 | {v63.get('annualized_return', 'N/A'):.2%} |
| 最大回撤 | {v63.get('max_drawdown', 'N/A'):.2%} |
| 夏普比率 | {v63.get('sharpe_ratio', 'N/A'):.2f} |
| 胜率 | {v63.get('win_rate', 'N/A'):.2%} |
| 交易次数 | {v63.get('total_trades', 'N/A')} |
| 平均持仓天数 | {v63.get('avg_holding_days', 'N/A'):.1f} |
| 强制空仓天数 | {v63.get('forced_empty_days', 'N/A')} |
| 时间止损次数 | {v63.get('time_stop_count', 'N/A')} |

## 核心指标对比

| 指标 | V62 | V63 | 差异 | V63 更优 |
|------|-----|-----|------|----------|
"""
    
    for name, data in comparison.get('metrics', {}).items():
        v62_val = f"{data['v62']:.2%}" if 'return' in name or 'rate' in name or 'drawdown' in name else f"{data['v62']:.2f}"
        v63_val = f"{data['v63']:.2%}" if 'return' in name or 'rate' in name or 'drawdown' in name else f"{data['v63']:.2f}"
        diff_str = f"{data['diff_pct']:+.1f}%"
        better_str = "✅" if data['v63_better'] else "❌"
        content += f"| {name} | {v62_val} | {v63_val} | {diff_str} | {better_str} |\n"
    
    content += f"""
## 因子贡献度审计

### V62 状态分析
- 总收益：{v62.get('total_return', 'N/A'):.2%}
- 最大回撤：{v62.get('max_drawdown', 'N/A'):.2%}
- 是否灾难性亏损：{'是' if factor_audit.get('v62_loss_analysis', {}).get('is_disaster', False) else '否'}

### V63 改进分析
- 收益改善：{factor_audit.get('v63_improvement', {}).get('return_improvement', 0)*100:+.2f}%
- 回撤降低：{factor_audit.get('v63_improvement', {}).get('drawdown_reduction', 0)*100:.2f}%
- 交易减少：{factor_audit.get('v63_improvement', {}).get('trade_reduction', 0)}次
- 胜率提升：{factor_audit.get('v63_improvement', {}).get('win_rate_improvement', 0)*100:+.2f}%
- 强制空仓天数：{factor_audit.get('v63_improvement', {}).get('forced_empty_days', 0)}天
- 时间止损次数：{factor_audit.get('v63_improvement', {}).get('time_stop_count', 0)}次

### 因子贡献度排名

| 排名 | 因子 | 描述 | 影响程度 |
|------|------|------|----------|
"""
    
    for i, factor in enumerate(factor_audit.get('factor_contribution', {}).get('factors', []), 1):
        content += f"| {i} | {factor['factor']} | {factor['description']} | {factor['impact']} |\n"
    
    content += f"""
## 结论

{factor_audit.get('conclusion', '无结论')}

## 综合评判

- 核心指标胜出：V62 {comparison.get('v62_wins', 0)} 项，V63 {comparison.get('v63_wins', 0)} 项
- 综合赢家：**{comparison.get('winner', 'N/A')}**

---
*报告由 V63 双回测对比脚本自动生成*
"""
    
    report_path.write_text(content, encoding='utf-8')
    
    # 同时保存 JSON 数据
    json_path = report_dir / f"V63_V62_Comparison_Data_{timestamp}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    return report_path


def print_final_summary(results: dict):
    """打印最终总结"""
    logger.info("\n" + "=" * 80)
    logger.info("最终总结")
    logger.info("=" * 80)
    
    v62 = results.get('v62', {})
    v63 = results.get('v63', {})
    comparison = results.get('comparison', {})
    factor_audit = results.get('factor_audit', {})
    
    if not v62 or not v63:
        logger.info("⚠️ 回测数据不完整，无法生成总结")
        return
    
    v62_return = v62.get('total_return', 0)
    v63_return = v63.get('total_return', 0)
    v62_max_dd = v62.get('max_drawdown', 0)
    v63_max_dd = v63.get('max_drawdown', 0)
    
    logger.info(f"\n📊 收益对比:")
    logger.info(f"  V62: {v62_return*100:.2f}%")
    logger.info(f"  V63: {v63_return*100:.2f}%")
    logger.info(f"  改善：{(v63_return - v62_return)*100:+.2f}%")
    
    logger.info(f"\n📉 风险对比:")
    logger.info(f"  V62 最大回撤：{v62_max_dd*100:.2f}%")
    logger.info(f"  V63 最大回撤：{v63_max_dd*100:.2f}%")
    logger.info(f"  降低：{(v62_max_dd - v63_max_dd)*100:.2f}%")
    
    logger.info(f"\n🏆 综合评判：{comparison.get('winner', 'N/A')} 胜出")
    logger.info(f"\n💡 核心结论：{factor_audit.get('conclusion', '无结论')}")
    
    logger.info("\n" + "=" * 80)


if __name__ == "__main__":
    # 运行双回测对比
    results = run_dual_backtest(
        start_date="2024-01-01",
        end_date="2024-12-31",
        initial_capital=100000.0,
        max_positions=10
    )
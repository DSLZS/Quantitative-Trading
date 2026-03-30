"""
V91 回测运行脚本

使用方法：
    python src/run_v91_backtest.py

功能：
    1. 运行 V91 非线性特征共振回测
    2. 生成审计报告
    3. 保存结果至 JSON 文件
"""

import os
import sys
import json
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from src.v91_engine import V91Engine, V91EngineConfig, run_v91_backtest, print_v91_report


def main():
    """主函数"""
    # 配置日志
    logger.remove()
    logger.add(
        sink=lambda msg: print(msg, end=""),
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
        level="INFO"
    )
    
    # 添加文件日志
    log_path = "logs/v91_backtest_{time:YYYYMMDD_HHmmss}.log"
    os.makedirs("logs", exist_ok=True)
    logger.add(log_path, level="DEBUG", rotation="100 MB")
    
    logger.info("=" * 70)
    logger.info("V91 非线性特征共振回测")
    logger.info(f"运行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 70)
    
    # 创建配置
    config = V91EngineConfig(
        start_date="2019-01-01",
        end_date="2024-12-31",
        initial_capital=100000.00,  # 严禁修改
        max_positions=30,
        warmup_period=60,  # Reduced warmup to allow more trading days
        oos_years=["2019", "2021", "2024"],
        min_score_threshold=55.0,
        min_single_weight=0.003,
        max_single_weight=0.08,
        enable_resonance=True,  # 启用非线性共振
        enable_regime_switching=True,  # 启用市场状态切换
        enable_integrity_shield=True,  # 启用数据防御
    )
    
    logger.info("")
    logger.info("【配置参数】")
    logger.info(f"  回测区间：{config.start_date} ~ {config.end_date}")
    logger.info(f"  初始资金：{config.initial_capital:,.2f}")
    logger.info(f"  最大持仓：{config.max_positions}")
    logger.info(f"  热身周期：{config.warmup_period}天")
    logger.info(f"  OOS 年份：{config.oos_years}")
    logger.info(f"  非线性共振：{'启用' if config.enable_resonance else '禁用'}")
    logger.info(f"  市场状态切换：{'启用' if config.enable_regime_switching else '禁用'}")
    logger.info(f"  数据防御：{'启用' if config.enable_integrity_shield else '禁用'}")
    logger.info("")
    
    # 运行回测
    result = run_v91_backtest(config)
    
    # 打印报告
    print_v91_report(result)
    
    # 保存结果
    output_path = "reports/v91_backtest_result.json"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 生成可序列化的结果
    ic_audit = result.get('ic_audit', {})
    ic_by_year = ic_audit.get('ic_by_year', {})
    
    serializable_result = {
        'run_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': {
            'start_date': config.start_date,
            'end_date': config.end_date,
            'initial_capital': config.initial_capital,
            'max_positions': config.max_positions,
            'warmup_period': config.warmup_period,
            'oos_years': config.oos_years,
            'enable_resonance': config.enable_resonance,
            'enable_regime_switching': config.enable_regime_switching,
            'enable_integrity_shield': config.enable_integrity_shield,
        },
        'data_integrity': result.get('data_integrity', {}),
        'ic_audit': {
            'ic_t1': ic_audit.get('ic_t1', 0.0),
            'ic_t2': ic_audit.get('ic_t2', 0.0),
            'ic_t3': ic_audit.get('ic_t3', 0.0),
            'std_t1': ic_audit.get('std_t1', 0.0),
            'std_t2': ic_audit.get('std_t2', 0.0),
            'std_t3': ic_audit.get('std_t3', 0.0),
            'ic_ir': ic_audit.get('ic_ir', 0.0),
            'mean_ic_3yr': ic_audit.get('mean_ic_3yr', 0.0),
            'decay_normal': ic_audit.get('decay_normal', False),
            't1_ic_passed': ic_audit.get('t1_ic_passed', False),
            'ic_ir_passed': ic_audit.get('ic_ir_passed', False),
            'mean_ic_passed': ic_audit.get('mean_ic_passed', False),
            'ic_by_year': {
                year: {
                    'mean_ic': data.get('mean_ic', 0.0),
                    'std_ic': data.get('std_ic', 0.0),
                    'ic_count': data.get('ic_count', 0),
                }
                for year, data in ic_by_year.items()
            },
        },
        'lookahead_check': result.get('lookahead_check', {}),
        'trade_results': {
            'total_return': result.get('trade_results', {}).get('total_return', 0.0),
            'final_value': result.get('trade_results', {}).get('final_value', 0.0),
            'max_drawdown': result.get('trade_results', {}).get('max_drawdown', 0.0),
            'annualized_turnover': result.get('trade_results', {}).get('annualized_turnover', 0.0),
            'total_trading_days': result.get('trade_results', {}).get('total_trading_days', 0),
            'total_trades': result.get('trade_results', {}).get('total_trades', 0),
            'rebalance_count': result.get('trade_results', {}).get('rebalance_count', 0),
            'avg_annual_return': result.get('trade_results', {}).get('avg_annual_return', 0.0),
            'annual_returns': result.get('trade_results', {}).get('annual_returns', {}),
        },
        'rebalance_errors': [
            {
                'trade_date': err.trade_date,
                'symbol': err.symbol,
                'error_type': err.error_type,
                'error_message': err.error_message,
                'skipped': err.skipped,
            }
            for err in result.get('rebalance_errors', [])
        ],
        'summary': {
            'trade_count': len(result.get('trade_records', [])),
            'snapshot_count': len(result.get('daily_snapshots', [])),
            'rebalance_count': len(result.get('rebalance_dates', [])),
            'error_count': len(result.get('rebalance_errors', [])),
        },
        'audit_report': result.get('audit_report', ''),
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serializable_result, f, indent=2, ensure_ascii=False, default=str)
    
    logger.info(f"V91: 结果已保存至 {output_path}")
    
    # 生成 Markdown 报告
    md_report_path = generate_markdown_report(serializable_result)
    logger.info(f"V91: Markdown 报告已保存至 {md_report_path}")
    
    logger.info("")
    logger.info("=" * 70)
    logger.info("V91 回测完成")
    logger.info("=" * 70)
    
    return serializable_result


def generate_markdown_report(result: dict) -> str:
    """生成 Markdown 报告"""
    lines = []
    
    lines.append("# V91 非线性特征共振回测报告")
    lines.append("")
    lines.append(f"**运行时间**: {result.get('run_time', 'N/A')}")
    lines.append("")
    
    # 配置参数
    lines.append("## 1. 配置参数")
    lines.append("")
    config = result.get('config', {})
    lines.append(f"- 回测区间：{config.get('start_date', 'N/A')} ~ {config.get('end_date', 'N/A')}")
    lines.append(f"- 初始资金：{config.get('initial_capital', 0):,.2f}")
    lines.append(f"- 最大持仓：{config.get('max_positions', 0)}")
    lines.append(f"- 热身周期：{config.get('warmup_period', 0)}天")
    lines.append(f"- OOS 年份：{', '.join(config.get('oos_years', []))}")
    lines.append(f"- 非线性共振：{'✓ 启用' if config.get('enable_resonance') else '✗ 禁用'}")
    lines.append(f"- 市场状态切换：{'✓ 启用' if config.get('enable_regime_switching') else '✗ 禁用'}")
    lines.append(f"- 数据防御：{'✓ 启用' if config.get('enable_integrity_shield') else '✗ 禁用'}")
    lines.append("")
    
    # IC 审计
    lines.append("## 2. IC 审计")
    lines.append("")
    ic_audit = result.get('ic_audit', {})
    lines.append("| 指标 | 值 | 目标 | 状态 |")
    lines.append("|------|-----|------|------|")
    lines.append(f"| T+1 Rank IC | {ic_audit.get('ic_t1', 0):.4f} | > 0.045 | {'✓' if ic_audit.get('t1_ic_passed') else '✗'} |")
    lines.append(f"| T+2 Rank IC | {ic_audit.get('ic_t2', 0):.4f} | - | - |")
    lines.append(f"| T+3 Rank IC | {ic_audit.get('ic_t3', 0):.4f} | - | - |")
    lines.append(f"| IC IR | {ic_audit.get('ic_ir', 0):.2f} | > 0.6 | {'✓' if ic_audit.get('ic_ir_passed') else '✗'} |")
    lines.append(f"| 三年度平均 IC | {ic_audit.get('mean_ic_3yr', 0):.4f} | > 0.045 | {'✓' if ic_audit.get('mean_ic_passed') else '✗'} |")
    lines.append("")
    
    # 分年度 IC
    ic_by_year = ic_audit.get('ic_by_year', {})
    if ic_by_year:
        lines.append("### 分年度 IC")
        lines.append("")
        lines.append("| 年份 | Mean IC | Std IC | 样本数 |")
        lines.append("|------|---------|--------|--------|")
        for year in ['2019', '2021', '2024']:
            if year in ic_by_year:
                data = ic_by_year[year]
                lines.append(f"| {year} | {data.get('mean_ic', 0):.4f} | {data.get('std_ic', 0):.4f} | {data.get('ic_count', 0)} |")
        lines.append("")
    
    # 交易执行
    lines.append("## 3. 交易执行")
    lines.append("")
    trade_results = result.get('trade_results', {})
    lines.append(f"- 总收益：{trade_results.get('total_return', 0):.2%}")
    lines.append(f"- 最终价值：{trade_results.get('final_value', 0):,.2f}")
    lines.append(f"- 最大回撤：{trade_results.get('max_drawdown', 0):.2%}")
    lines.append(f"- 年化换手率：{trade_results.get('annualized_turnover', 0):.2%}")
    lines.append(f"- 调仓次数：{trade_results.get('rebalance_count', 0)}")
    lines.append(f"- 平均年化收益：{trade_results.get('avg_annual_return', 0):.2%}")
    lines.append("")
    
    # 年度收益
    annual_returns = trade_results.get('annual_returns', {})
    if annual_returns:
        lines.append("### 年度收益")
        lines.append("")
        lines.append("| 年份 | 收益率 |")
        lines.append("|------|--------|")
        for year in ['2019', '2021', '2024']:
            ret = annual_returns.get(year, 0)
            lines.append(f"| {year} | {ret:.2%} |")
        lines.append("")
    
    # 硬性指标验证
    lines.append("## 4. 硬性指标验证")
    lines.append("")
    
    # 指标 A
    ic_by_year = ic_audit.get('ic_by_year', {})
    metric_a_2019 = ic_by_year.get('2019', {}).get('mean_ic', 0) >= 0.045
    metric_a_2021 = ic_by_year.get('2021', {}).get('mean_ic', 0) >= 0.045
    metric_a_2024 = ic_by_year.get('2024', {}).get('mean_ic', 0) >= 0.045
    metric_a_ir = ic_audit.get('ic_ir', 0) >= 0.6
    metric_a_pass = metric_a_2019 and metric_a_2021 and metric_a_2024 and metric_a_ir
    
    lines.append("### 指标 A：三年度 IC > 0.045, IC IR > 0.6")
    lines.append("")
    lines.append(f"- 2019 IC: {ic_by_year.get('2019', {}).get('mean_ic', 0):.4f} {'✓' if metric_a_2019 else '✗'}")
    lines.append(f"- 2021 IC: {ic_by_year.get('2021', {}).get('mean_ic', 0):.4f} {'✓' if metric_a_2021 else '✗'}")
    lines.append(f"- 2024 IC: {ic_by_year.get('2024', {}).get('mean_ic', 0):.4f} {'✓' if metric_a_2024 else '✗'}")
    lines.append(f"- IC IR: {ic_audit.get('ic_ir', 0):.2f} {'✓' if metric_a_ir else '✗'}")
    lines.append(f"- **状态**: {'✓ 通过' if metric_a_pass else '✗ 未通过'}")
    lines.append("")
    
    # 指标 B
    metric_b_2021 = annual_returns.get('2021', 0) > 0.05
    metric_b_2024 = annual_returns.get('2024', 0) > 0.05
    metric_b_pass = metric_b_2021 and metric_b_2024
    
    lines.append("### 指标 B:2021&2024 收益 > 5%")
    lines.append("")
    lines.append(f"- 2021 收益：{annual_returns.get('2021', 0):.2%} {'✓' if metric_b_2021 else '✗'}")
    lines.append(f"- 2024 收益：{annual_returns.get('2024', 0):.2%} {'✓' if metric_b_2024 else '✗'}")
    lines.append(f"- **状态**: {'✓ 通过' if metric_b_pass else '✗ 未通过'}")
    lines.append("")
    
    # 指标 C
    error_count = result.get('summary', {}).get('error_count', 0)
    
    lines.append("### 指标 C:Max_Rebalancing_Error_Log")
    lines.append("")
    lines.append(f"- 错误数量：{error_count}")
    lines.append(f"- **状态**: ✓ 通过（日志已记录）")
    lines.append("")
    
    # 总体评估
    all_passed = metric_a_pass and metric_b_pass
    lines.append("## 5. 总体评估")
    lines.append("")
    lines.append(f"### {'✓ 所有指标通过' if all_passed else '✗ 部分指标未通过'}")
    lines.append("")
    
    # 调仓错误日志
    rebalance_errors = result.get('rebalance_errors', [])
    if rebalance_errors:
        lines.append("## 6. Max_Rebalancing_Error_Log")
        lines.append("")
        lines.append("| 日期 | 标的 | 错误类型 | 错误信息 | 跳过 |")
        lines.append("|------|------|----------|----------|------|")
        for err in rebalance_errors[:20]:  # 限制显示 20 条
            lines.append(f"| {err.get('trade_date', 'N/A')} | {err.get('symbol', 'N/A')} | {err.get('error_type', 'N/A')} | {err.get('error_message', 'N/A')[:50]} | {'是' if err.get('skipped') else '否'} |")
        if len(rebalance_errors) > 20:
            lines.append(f"... 还有 {len(rebalance_errors) - 20} 条错误")
        lines.append("")
    
    return "\n".join(lines)


if __name__ == "__main__":
    main()
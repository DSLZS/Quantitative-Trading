# reports/ 目录说明

## 用途

本目录存放回测审计报告，包括：
- 跨年度汇总报告（Cross-Year Report）
- 年度审计报告（Yearly Audit Report）
- 运行日志（Run Log）

## 文件命名规范

| 文件类型 | 命名格式 | 示例 |
|----------|----------|------|
| 跨年度报告 | `V{version}_Cross_Year_Report_{timestamp}.md` | `V218_Cross_Year_Report_20260427_140611.md` |
| 跨年度JSON | `V{version}_Cross_Year_Report_{timestamp}.json` | `V218_Cross_Year_Report_20260427_140611.json` |
| 年度审计 | `v{version}_year{year}_audit_{timestamp}.md` | `v218_year2024_audit_20260427_140611.md` |
| 年度审计JSON | `v{version}_year{year}_audit_{timestamp}.json` | `v218_year2024_audit_20260427_140611.json` |
| 运行日志 | `v{version}_run_{date}.log` | `v218_run_20260427.log` |

## 当前保留文件

仅保留 V218 版本的报告（2026-04-27生成）：
- `V218_Cross_Year_Report_20260427_140611.md` - 最新跨年度报告
- `V218_Cross_Year_Report_20260427_140611.json` - 最新跨年度JSON
- `v218_year2020_audit_20260427_140522.md` - 2020年审计
- `v218_year2022_audit_20260427_140544.md` - 2022年审计
- `v218_year2024_audit_20260427_140611.md` - 2024年审计
- `v218_run_20260427.log` - 运行日志

## 清理策略

每次新版本回测完成后：
1. 保留最新版本的报告
2. 删除旧版本报告（可通过 Git 历史恢复）
3. 清理超过30天的运行日志

---

*最后更新: 2026-04-27*
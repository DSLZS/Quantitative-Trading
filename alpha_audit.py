#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
alpha_audit.py - 量化策略半自动化审计脚本（纯审计+提示词生成器）

功能：
1. 代码审计：检查违规行为、AI偷懒痕迹、过拟合风险。
2. 回测结果评析：评价最新回测的 IC、IR、年化收益等。
3. 生成下一步完整提示词：包含所有历史经验、约束、禁止行为、文件维护规范。

用法：
    python alpha_audit.py

环境变量：
    DEEPSEEK_API_KEY: DeepSeek API 密钥

输出：
    - next_prompt.txt: 下一步完整提示词（可直接复制给 cline）
    - ALPHA_HISTORY.md 将被自动追加审计结论（如果检测到历史日志格式）
"""

import os
import re
import yaml
import requests
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ====================== 配置 ======================
CONFIG_FILE = Path(__file__).parent / "audit_config.yaml"
if not CONFIG_FILE.exists():
    raise FileNotFoundError(f"配置文件 {CONFIG_FILE} 不存在，请先创建")

with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

PROJECT_ROOT = Path(__file__).parent
CORE_CODE_GLOB = config["core_code_glob"]
HISTORY_FILE = PROJECT_ROOT / config["history_file"]
CURRENT_STATE_FILE = PROJECT_ROOT / config["current_state_file"]
TODOS_FILE = PROJECT_ROOT / config["todos_file"]
REPORT_DIR = PROJECT_ROOT / config["report_dir"]
REPORT_README = REPORT_DIR / "README.md"
GIT_DIFF_COMMITS = config.get("git_diff_commits", 5)

DEEPSEEK_API_URL = config["deepseek"]["api_url"]
MODEL = config["deepseek"]["model"]
MAX_TOKENS = config["deepseek"]["max_tokens"]
TEMPERATURE = config["deepseek"]["temperature"]

TRADING_RULES = "\n".join(config["trading_rules"])


# ====================== 辅助函数 ======================
def get_latest_core_code():
    files = list(PROJECT_ROOT.glob(CORE_CODE_GLOB))
    if not files:
        return None, None
    latest = max(files, key=lambda f: f.stat().st_mtime)
    return latest.name, latest.read_text(encoding='utf-8')


def get_latest_report():
    reports = list(REPORT_DIR.glob("V*_Cross_Year_Report_*.md"))
    if not reports:
        return None, None
    latest = max(reports, key=lambda f: f.stat().st_mtime)
    content = latest.read_text(encoding='utf-8')
    summary = content[:3500] + ("\n...(报告摘要已截断)" if len(content) > 3500 else "")
    return latest.name, summary


def get_git_diff():
    try:
        import subprocess
        diff = subprocess.check_output(
            ["git", "diff", f"HEAD~{GIT_DIFF_COMMITS}", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL
        )
        return diff[:5000]
    except:
        return "无法获取 git diff（可能不在 git 仓库中）"


def read_file_with_limit(path, limit=8000):
    if not path.exists():
        return f"文件不存在: {path}"
    content = path.read_text(encoding='utf-8')
    if len(content) > limit:
        content = content[:limit] + "\n...(文件过长，已截断)"
    return content


def call_deepseek(prompt):
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("请设置环境变量 DEEPSEEK_API_KEY")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE
    }
    response = requests.post(DEEPSEEK_API_URL, headers=headers, json=payload, timeout=120)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def build_audit_prompt(history, state, todos, code_name, code, report_name, report_summary, git_diff, report_readme):
    """构造发送给 DeepSeek 的审计提示词（包含补充功能）"""
    prompt = f"""
你是量化策略研发专家，同时也是项目审计官。你的任务是对当前策略代码和回测结果进行深度审计，并生成下一步的完整提示词。

【核心约束】（你在审计时必须检查以下内容，并在输出中指出违规）
- 严禁 T+0 交易，所有交易必须 T+1 执行。
- 严禁使用未来函数（如 shift(-1)、访问 T+1 数据）。
- 严禁修改 `src/engine/backtest_referee.py` 中的费率、资金、持仓参数。
- 严禁使用 `fillna(0)` 填充缺失值，必须使用数据修复模块。
- 严禁超过 3 个因子的线性叠加，严禁三阶及以上交互项。
- 严禁篡改回测结果（如选择性展示年份、手工调整指标）。
- 必须遵循 Referee-Player 解耦架构：Player 只输出因子得分，Referee 负责回测。

【报告管理规范】（必须遵照执行，内容来自 reports/README.md）
{report_readme[:2000]}

【历史经验】（ALPHA_HISTORY.md）
{history[:10000]}

【当前状态快照】（CURRENT_STATE.md）
{state[:4000]}

【待办事项】（TODOS.md）
{todos[:3000]}

【最新核心代码】：{code_name}
以下是该文件的纯文本内容：
{code[:8000]}

【最新回测报告摘要】：{report_name if report_name else '无'}
{report_summary[:4000] if report_summary else '无最近报告'}

【最近 Git 变更】（用于检测恶意修改）
{git_diff[:3000]}

---
请执行以下审计任务：

1. **合规性检查**：逐条检查上述约束，指出代码中是否存在违反行为。特别关注：
   - 是否有 T+0 或未来函数？
   - 是否修改了裁判引擎参数？
   - 是否使用了 fillna(0)？
   - 是否超过 3 个因子？
   - 是否保持了解耦架构？
   如有违规，明确指出文件、行号和问题。

2. **AI 行为审计**：从历史日志、代码注释、Git 提交信息中，判断之前的 AI（如 cline）是否存在以下"偷懒/不负责"行为：
   - 遇到报错不主动解决，而是绕过或忽略。
   - 遇到数据缺失（如数据库字段不存在）不主动编写脚本拉取或修复。
   - 试图打破解耦架构（如在 Player 中直接计算 IC）。
   - 不运行回测就声称结果改善。
   - 美化结果（如只展示好的年份）。
   - 未遵循 reports/README.md 中的报告命名和清理规范（如未删除旧报告）。
   若发现此类问题，列出具体证据。

3. **过拟合风险检查**：结合历史经验和最新回测结果，评估当前策略是否存在过拟合：
   - 是否仅针对特定年份（如 2024）调参？
   - 因子数量是否过多？
   - IC 是否在跨年度间波动极大？
   - 是否使用了复杂非线性或高频数据？

4. **回测结果评析**：
   - 提取最新报告中的 2020、2022、2024 年 T+1 IC、IC IR、年化收益、最大回撤。
   - 判断策略方向是否正确：若 2024 年 IC 显著低于前两年，说明策略失效，需要调整。
   - 给出整体评价：优秀/合格/不合格。

5. **生成下一步完整提示词**：基于上述审计结果，输出一段可直接复制给 AI（如 cline）的提示词，要求其进行**增量优化**。该提示词必须包含以下所有内容：
   - 所有历史经验和禁止规则（从本项目中的约束继承）。
   - 明确要求基于当前代码迭代，不得推倒重来。
   - 强制要求修改后必须运行回测，并对比修改前后的 IC 变化。
   - 强制要求遇到数据缺失或报错必须主动解决（如补充 SQL、安装依赖、修复数据源）。
   - 严禁美化结果或选择性展示。
   - **强制要求阅读并遵循 `reports/README.md` 中的报告命名规范和清理规范**（新报告按格式命名，旧报告及时清理，仅保留最新版本）。
   - **强制要求：在每次迭代获得关键经验教训（无论成功或失败）时，必须追加到 `ALPHA_HISTORY.md` 中**，格式为 `## V[版本号] - YYYY-MM-DD`，包含核心尝试、结果、教训。
   - **强制要求：在修改代码或完成重要待办后，必须同步更新 `CURRENT_STATE.md`**，包括因子公式、市场状态逻辑、数据依赖、已知问题等。
   - 明确输出格式：需提供修改的代码片段、运行回测的命令、对比结果。

6. **更新历史日志**：输出追加到 ALPHA_HISTORY.md 的内容，格式如下：
   ## V[下一个版本号] - [当前日期]
   - 核心尝试：...
   - 结果：...
   - 教训：...

【输出要求】
- 清晰分段，每部分用【】标识。
- 不要输出多余的解释性前言。
- 【下一步完整提示词】部分必须单独成段，可直接复制使用。
- 确保历史日志追加格式正确。

现在开始审计。
"""
    return prompt


def append_to_history(history_file, content):
    match = re.search(r"(##\s*V\d+\s*-\s*\d{4}-\d{2}-\d{2}.*?)(?=\n##|\Z)", content, re.DOTALL)
    if match:
        append_content = match.group(1).strip()
        with open(history_file, "a", encoding="utf-8") as f:
            f.write("\n\n" + append_content)
        return True
    return False


# ====================== 主流程 ======================
def main():
    print("=" * 60)
    print("Alpha策略审计脚本 alpha_audit.py（纯审计+提示词生成）")
    print("=" * 60)

    # 1. 收集文件
    print("1. 收集项目文件...")
    hist_content = read_file_with_limit(HISTORY_FILE, 12000)
    state_content = read_file_with_limit(CURRENT_STATE_FILE, 8000)
    todos_content = read_file_with_limit(TODOS_FILE, 4000)
    code_name, code_content = get_latest_core_code()
    if not code_content:
        print("   ❌ 未找到核心代码文件，退出")
        return
    report_name, report_summary = get_latest_report()
    git_diff = get_git_diff()
    report_readme = read_file_with_limit(REPORT_README, 2000) if REPORT_README.exists() else "未找到 reports/README.md 文件，请创建。"

    print(f"   ✓ 历史日志: {len(hist_content)} 字符")
    print(f"   ✓ 当前状态: {len(state_content)} 字符")
    print(f"   ✓ 待办清单: {len(todos_content)} 字符")
    print(f"   ✓ 核心代码: {code_name}")
    print(f"   ✓ 最新报告: {report_name if report_name else '无'}")
    print(f"   ✓ Git diff: {len(git_diff)} 字符")
    print(f"   ✓ 报告规范: {'已加载' if REPORT_README.exists() else '缺失'}")

    # 2. 构建审计提示词
    print("\n2. 构建审计提示词...")
    prompt = build_audit_prompt(
        hist_content, state_content, todos_content,
        code_name, code_content,
        report_name, report_summary,
        git_diff,
        report_readme
    )

    # 3. 调用 DeepSeek API
    print("\n3. 调用 DeepSeek 进行审计（可能需要 1-2 分钟）...")
    try:
        audit_result = call_deepseek(prompt)
    except Exception as e:
        print(f"   ❌ API 调用失败: {e}")
        return

    # 4. 保存审计结果（包含下一步提示词）
    print("\n4. 保存审计结果...")
    output_file = PROJECT_ROOT / "next_prompt.txt"
    output_file.write_text(audit_result, encoding='utf-8')
    print(f"   ✓ 审计结果已保存至: {output_file}")

    # 5. 尝试自动追加历史日志
    if append_to_history(HISTORY_FILE, audit_result):
        print(f"   ✓ 已自动追加历史日志到 {HISTORY_FILE}")
    else:
        print("   ⚠️ 未检测到可追加的历史日志格式，请手动检查")

    print("\n" + "=" * 60)
    print("审计完成。请查看 next_prompt.txt，其中包含【下一步完整提示词】。")
    print("复制该提示词给 cline（或您的开发 AI），进行下一轮迭代优化。")
    print("=" * 60)


if __name__ == "__main__":
    main()
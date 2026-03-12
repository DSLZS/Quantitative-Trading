"""
AI 审计锐度压力测试脚本

测试 DeepSeek 在不同风险场景下的审计能力，验证"宁缺毋滥"原则的执行。

测试场景：
1. 明确负面公告（立案调查、财务造假）- 应 REJECT
2. 疑似"小作文"传闻 - 应 REJECT
3. 利好陷阱（高送转 + 大股东减持）- 应 REJECT
4. 正常经营无风险 - 应 PASS
5. 轻微负面但非重大 - 应 PASS（边缘测试）

使用方法:
    python tests/test_stress_audit.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


class StressTestCase:
    """压力测试用例"""
    def __init__(self, name: str, description: str, stock_info: dict, news_summary: str, expected_status: str):
        self.name = name
        self.description = description
        self.stock_info = stock_info
        self.news_summary = news_summary
        self.expected_status = expected_status  # "PASS" or "REJECT"
        
        # 测试结果
        self.success = False
        self.actual_status = None
        self.response_text = ""
        self.response_time = 0.0
        self.tokens_used = 0
        self.error = None


# 定义测试用例
TEST_CASES = [
    # 测试 1: 明确负面 - 立案调查
    StressTestCase(
        name="明确负面：立案调查",
        description="公司因涉嫌信息披露违规被证监会立案调查",
        stock_info={"symbol": "600XXX", "name": "某科技股"},
        news_summary="公司因涉嫌信息披露违规被证监会立案调查，目前正在配合调查。",
        expected_status="REJECT"
    ),
    
    # 测试 2: 明确负面 - 财务造假
    StressTestCase(
        name="明确负面：财务造假",
        description="公司被曝光财务造假，虚增利润",
        stock_info={"symbol": "000XXX", "name": "某消费股"},
        news_summary="媒体曝光公司连续 3 年虚增营业收入和利润，证监会已立案并处罚。",
        expected_status="REJECT"
    ),
    
    # 测试 3: 明确负面 - 大股东减持
    StressTestCase(
        name="明确负面：大股东减持",
        description="大股东高位减持套现",
        stock_info={"symbol": "300XXX", "name": "某新能源"},
        news_summary="公司股价近期大涨后，控股股东宣布减持不超过 5% 股份套现。",
        expected_status="REJECT"
    ),
    
    # 测试 4: 疑似"小作文"传闻
    StressTestCase(
        name="疑似小作文：并购传闻",
        description="社交媒体流传并购消息但无官方公告",
        stock_info={"symbol": "601XXX", "name": "某医药股"},
        news_summary="社交媒体消息称公司将被大型国企收购，但公司未发布任何公告。",
        expected_status="REJECT"
    ),
    
    # 测试 5: 利好陷阱 - 高送转 + 减持
    StressTestCase(
        name="利好陷阱：高送转掩护减持",
        description="发布高送转方案后大股东减持",
        stock_info={"symbol": "002XXX", "name": "某制造股"},
        news_summary="公司发布 10 转 15 的高送转方案，同时公告大股东拟减持 3% 股份。",
        expected_status="REJECT"
    ),
    
    # 测试 6: 正常经营 - 业绩增长
    StressTestCase(
        name="正常经营：业绩增长",
        description="公司发布财报，业绩稳健增长",
        stock_info={"symbol": "600519", "name": "贵州茅台"},
        news_summary="公司发布三季报，营收同比增长 15%，净利润增长 18%，经营正常。",
        expected_status="PASS"
    ),
    
    # 测试 7: 正常经营 - 获得订单
    StressTestCase(
        name="正常经营：获得订单",
        description="公司签订重大合同",
        stock_info={"symbol": "002XXX", "name": "某电子股"},
        news_summary="公司与某大客户签订 5 亿元供货合同，占公司去年营收的 20%。",
        expected_status="PASS"
    ),
    
    # 测试 8: 边缘测试 - 轻微负面
    StressTestCase(
        name="边缘测试：原材料涨价",
        description="原材料价格上涨影响利润",
        stock_info={"symbol": "000XXX", "name": "某化工股"},
        news_summary="受大宗商品价格上涨影响，公司原材料成本上升，预计毛利率下降 2-3 个百分点。",
        expected_status="PASS"
    ),
    
    # 测试 9: ST 股风险
    StressTestCase(
        name="ST 股风险：退市预警",
        description="ST 股票面临退市风险",
        stock_info={"symbol": "ST 某某", "name": "ST 风险股"},
        news_summary="公司股票被实施退市风险警示，连续两年亏损，净资产为负。",
        expected_status="REJECT"
    ),
    
    # 测试 10: 实控人变更
    StressTestCase(
        name="实控人变更风险",
        description="公司实际控制人发生变更",
        stock_info={"symbol": "300XXX", "name": "某软件股"},
        news_summary="公司公告控股股东拟转让股权，实际控制人可能发生变更。",
        expected_status="REJECT"
    ),
]


def run_audit_test(test_case: StressTestCase) -> StressTestCase:
    """运行单个审计测试"""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    
    if not api_key:
        test_case.error = "DEEPSEEK_API_KEY 未设置"
        return test_case
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    system_prompt = """你是一位专业的股票风控审计员，负责识别潜在风险。请严格审计以下股票信息：

【审计指令】
1. 关键词扫描：严格排查以下负面关键词
   - 立案调查、违规担保、财务造假、面值退市
   - 实控人变更、信披违规、被证监会处罚
   - 重大违法强制退市、资金占用、关联交易

2. "小作文"识别：对比官方公告与社交媒体传闻
   - 官方公告权重最高
   - 对模棱两可的利好（如"高送转"数字游戏）保持高度警惕

3. 利好陷阱识别：
   - 识别是否为"掩护大股东减持"的虚假利好
   - 警惕"蹭热点"式公告

【判定原则】
- 宁缺毋滥：有任何重大风险直接 REJECT
- PASS 标准：无明显负面 + 无"小作文"嫌疑 + 非虚假利好

【输出格式】
请严格按以下 JSON 格式输出：
{
    "status": "PASS" 或 "REJECT",
    "reason": "简短说明原因（50 字以内）",
    "risk_level": "无/低/中/高",
    "keywords_found": ["发现的关键词列表，若无则为空"]
}

只输出 JSON，不要其他内容。"""

    user_prompt = f"""【股票信息】
代码：{test_case.stock_info['symbol']}
名称：{test_case.stock_info['name']}

【新闻摘要】
{test_case.news_summary}

请进行核心风控审计。"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 300
    }
    
    start_time = time.time()
    try:
        with httpx.Client(timeout=60.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            test_case.response_time = time.time() - start_time
            test_case.response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            test_case.tokens_used = data.get('usage', {}).get('total_tokens', 0)
            
            # 解析响应
            try:
                response_content = test_case.response_text.strip()
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    test_case.actual_status = parsed.get('status', '')
                    risk_level = parsed.get('risk_level', '')
                    keywords = parsed.get('keywords_found', [])
                    reason = parsed.get('reason', '')
                    
                    # 验证结果
                    if test_case.actual_status == test_case.expected_status:
                        test_case.success = True
                    else:
                        test_case.success = False
                        test_case.error = f"预期 {test_case.expected_status} 但得到 {test_case.actual_status}"
                else:
                    test_case.error = f"无法提取 JSON: {response_content[:100]}"
                    
            except json.JSONDecodeError as e:
                test_case.error = f"JSON 解析失败：{e}"
            
    except httpx.HTTPStatusError as e:
        test_case.error = f"HTTP 错误：{e.response.status_code}"
    except httpx.RequestError as e:
        test_case.error = f"请求错误：{e}"
    except Exception as e:
        test_case.error = f"未知错误：{e}"
    
    return test_case


def print_test_report(results: list[StressTestCase]) -> None:
    """打印测试报告"""
    print("\n" + "=" * 70)
    print("AI 审计锐度压力测试报告")
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    total_time = 0.0
    total_tokens = 0
    
    for result in results:
        status = "✅ PASS" if result.success else "❌ FAIL"
        expected_vs_actual = f"预期:{result.expected_status} → 实际:{result.actual_status or 'N/A'}"
        
        print(f"\n{status} - {result.name}")
        print("-" * 50)
        print(f"  描述：{result.description}")
        print(f"  {expected_vs_actual}")
        
        if result.success:
            total_passed += 1
            print(f"  响应时间：{result.response_time:.2f}秒")
            print(f"  Token 消耗：{result.tokens_used}")
            total_time += result.response_time
            total_tokens += result.tokens_used
            
            if result.response_text:
                try:
                    json_start = result.response_text.find('{')
                    json_end = result.response_text.rfind('}') + 1
                    if json_start >= 0 and json_end > json_start:
                        parsed = json.loads(result.response_text[json_start:json_end])
                        print(f"  审计原因：{parsed.get('reason', '')[:60]}...")
                except:
                    pass
        else:
            total_failed += 1
            print(f"  错误：{result.error}")
            print(f"  原始响应：{result.response_text[:100] if result.response_text else 'N/A'}...")
    
    print("\n" + "=" * 70)
    print(f"测试汇总：{total_passed} 通过 / {total_failed} 失败")
    if total_passed > 0:
        accuracy = total_passed / len(results) * 100
        print(f"准确率：{accuracy:.1f}%")
        print(f"平均响应时间：{total_time / total_passed:.2f}秒")
        print(f"总 Token 消耗：{total_tokens}")
    print("=" * 70)
    
    # 输出详细分析
    if total_failed > 0:
        print("\n⚠️  失败用例分析:")
        print("-" * 50)
        for result in results:
            if not result.success:
                print(f"\n  [{result.name}]")
                print(f"  预期：{result.expected_status}")
                print(f"  实际：{result.actual_status}")
                print(f"  错误：{result.error}")
                print(f"  响应：{result.response_text[:200] if result.response_text else 'N/A'}")


def main():
    """主函数"""
    print("开始 AI 审计锐度压力测试...")
    print(f"共 {len(TEST_CASES)} 个测试用例")
    
    results = []
    
    for i, test_case in enumerate(TEST_CASES, 1):
        print(f"\n[{i}/{len(TEST_CASES)}] 测试：{test_case.name}...")
        result = run_audit_test(test_case)
        results.append(result)
        
        # 短暂延迟避免 API 限流
        time.sleep(0.5)
    
    # 打印报告
    print_test_report(results)
    
    # 返回测试结果
    all_passed = all(r.success for r in results)
    if all_passed:
        print("\n🎉 所有测试通过！AI 审计逻辑表现优秀。")
    else:
        failed_count = sum(1 for r in results if not r.success)
        print(f"\n⚠️  {failed_count} 个测试失败，可能需要优化 SYSTEM_PROMPT。")
    
    return all_passed, results


if __name__ == "__main__":
    success, _ = main()
    sys.exit(0 if success else 1)
"""
AI Agents 连通性测试脚本

测试 DeepSeek 和 Qwen API 的真实连通性，验证 JSON 格式解析能力。

使用方法:
    python -m pytest tests/test_ai_agents.py -v
    或
    python tests/test_ai_agents.py
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


class TestResult:
    """测试结果容器"""
    def __init__(self, name: str):
        self.name = name
        self.success = False
        self.error = None
        self.response_time = 0.0
        self.tokens_used = 0
        self.response_text = ""


def test_qwen_api_connection() -> TestResult:
    """测试通义千问 API 连通性"""
    result = TestResult("Qwen API Connection Test")
    
    api_key = os.getenv("QWEN_API_KEY")
    if not api_key:
        result.error = "QWEN_API_KEY 环境变量未设置"
        return result
    
    # 检查是否为占位符
    if api_key.startswith("${") or "YOUR" in api_key.upper():
        result.error = f"QWEN_API_KEY 为占位符或未正确配置：{api_key[:20]}..."
        return result
    
    # 使用 DashScope 兼容模式 API
    url = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用今日热点话题测试
    messages = [
        {"role": "system", "content": "你是一位财经新闻分析师。请用简洁的语言总结新闻要点。"},
        {"role": "user", "content": "请总结以下新闻：今日 A 股市场震荡，科技股领涨，成交量放大。"}
    ]
    
    # 使用正确的模型名称
    payload = {
        "model": "qwen-turbo",
        "messages": messages,
        "max_tokens": 200
    }
    
    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            result.response_time = time.time() - start_time
            result.response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            result.tokens_used = data.get('usage', {}).get('total_tokens', 0)
            result.success = True
            
    except httpx.HTTPStatusError as e:
        result.error = f"HTTP 错误：{e.response.status_code} - {e.response.text[:100]}"
    except httpx.RequestError as e:
        result.error = f"请求错误：{e}"
    except Exception as e:
        result.error = f"未知错误：{e}"
    
    return result


def test_deepseek_api_connection() -> TestResult:
    """测试 DeepSeek API 连通性"""
    result = TestResult("DeepSeek API Connection Test")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        result.error = "DEEPSEEK_API_KEY 环境变量未设置"
        return result
    
    # 检查是否为占位符
    if api_key.startswith("${") or "YOUR" in api_key.upper():
        result.error = f"DEEPSEEK_API_KEY 为占位符或未正确配置：{api_key[:20]}..."
        return result
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用审计测试 prompt
    messages = [
        {
            "role": "system", 
            "content": """你是一位专业的股票风控审计员。请严格按以下 JSON 格式输出：
{
    "status": "PASS" 或 "REJECT",
    "reason": "简短说明原因（50 字以内）",
    "risk_level": "无/低/中/高",
    "keywords_found": ["发现的关键词列表"]
}
只输出 JSON，不要其他内容。"""
        },
        {
            "role": "user", 
            "content": "股票：贵州茅台 (600519)，新闻摘要：公司发布三季度财报，营收同比增长 15%，净利润增长 18%。"
        }
    ]
    
    payload = {
        "model": "deepseek-chat",
        "messages": messages,
        "max_tokens": 200
    }
    
    start_time = time.time()
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            result.response_time = time.time() - start_time
            result.response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            result.tokens_used = data.get('usage', {}).get('total_tokens', 0)
            
            # 验证 JSON 格式
            try:
                response_content = result.response_text.strip()
                # 尝试提取 JSON
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    parsed = json.loads(json_str)
                    if 'status' in parsed and 'reason' in parsed:
                        result.success = True
                    else:
                        result.error = f"JSON 格式正确但缺少必要字段：{parsed}"
                else:
                    result.error = f"无法从响应中提取 JSON: {response_content[:100]}"
            except json.JSONDecodeError as e:
                result.error = f"JSON 解析失败：{e} - 原始响应：{response_content[:100]}"
            
    except httpx.HTTPStatusError as e:
        result.error = f"HTTP 错误：{e.response.status_code} - {e.response.text[:100]}"
    except httpx.RequestError as e:
        result.error = f"请求错误：{e}"
    except Exception as e:
        result.error = f"未知错误：{e}"
    
    return result


def test_negative_stock_audit() -> TestResult:
    """测试负面股票审计 - 压力点探测"""
    result = TestResult("Negative Stock Audit Test (ST 股风险识别)")
    
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        result.error = "DEEPSEEK_API_KEY 环境变量未设置"
        return result
    
    if api_key.startswith("${") or "YOUR" in api_key.upper():
        result.error = f"DEEPSEEK_API_KEY 为占位符或未正确配置"
        return result
    
    url = "https://api.deepseek.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    # 使用真实负面案例测试
    messages = [
        {
            "role": "system", 
            "content": """你是一位专业的股票风控审计员，负责识别潜在风险。请严格审计以下股票信息：

【审计指令】
1. 关键词扫描：严格排查以下负面关键词
   - 立案调查、违规担保、财务造假、面值退市
   - 实控人变更、信披违规、被证监会处罚
   - 重大违法强制退市、资金占用、关联交易

2. "小作文"识别：对比官方公告与社交媒体传闻

3. 利好陷阱识别：识别是否为"掩护大股东减持"的虚假利好

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
        },
        {
            "role": "user", 
            "content": """【股票信息】
代码：600XXX
名称：某 ST 公司

【负面新闻摘要】
1. 公司因涉嫌信息披露违规被证监会立案调查
2. 实际控制人因涉嫌经济犯罪被公安机关采取强制措施
3. 公司连续三年亏损，面临退市风险
4. 大股东减持计划正在实施中

请进行核心风控审计。"""
        }
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
            
            result.response_time = time.time() - start_time
            result.response_text = data.get('choices', [{}])[0].get('message', {}).get('content', '')
            result.tokens_used = data.get('usage', {}).get('total_tokens', 0)
            
            # 解析并验证审计结果
            try:
                response_content = result.response_text.strip()
                json_start = response_content.find('{')
                json_end = response_content.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = response_content[json_start:json_end]
                    parsed = json.loads(json_str)
                    
                    status = parsed.get('status', '')
                    risk_level = parsed.get('risk_level', '')
                    keywords = parsed.get('keywords_found', [])
                    
                    # 验证是否正确识别风险
                    if status == 'REJECT':
                        result.success = True
                        result.response_text = f"✅ 正确识别风险!\n状态：{status}\n风险等级：{risk_level}\n关键词：{keywords}\n原因：{parsed.get('reason', '')}"
                    else:
                        result.success = False
                        result.error = f"❌ 审计失败！预期 REJECT 但得到 {status}。风险等级：{risk_level}, 关键词：{keywords}"
                else:
                    result.error = f"无法从响应中提取 JSON: {response_content[:200]}"
                    
            except json.JSONDecodeError as e:
                result.error = f"JSON 解析失败：{e} - 原始响应：{response_content[:200]}"
            
    except httpx.HTTPStatusError as e:
        result.error = f"HTTP 错误：{e.response.status_code} - {e.response.text[:100]}"
    except httpx.RequestError as e:
        result.error = f"请求错误：{e}"
    except Exception as e:
        result.error = f"未知错误：{e}"
    
    return result


def print_report(results: list[TestResult]) -> None:
    """打印测试报告"""
    print("\n" + "=" * 70)
    print("AI API 连通性测试报告")
    print(f"测试时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    total_success = 0
    total_failed = 0
    total_time = 0.0
    total_tokens = 0
    
    for result in results:
        status = "✅ PASS" if result.success else "❌ FAIL"
        print(f"\n{status} - {result.name}")
        print("-" * 50)
        
        if result.success:
            total_success += 1
            print(f"  响应时间：{result.response_time:.2f}秒")
            print(f"  Token 消耗：{result.tokens_used}")
            total_time += result.response_time
            total_tokens += result.tokens_used
            
            if result.response_text:
                print(f"  响应内容：{result.response_text[:200]}...")
        else:
            total_failed += 1
            print(f"  错误信息：{result.error}")
    
    print("\n" + "=" * 70)
    print(f"测试汇总：{total_success} 通过 / {total_failed} 失败")
    if total_success > 0:
        print(f"平均响应时间：{total_time / total_success:.2f}秒")
        print(f"总 Token 消耗：{total_tokens}")
    print("=" * 70)


def main():
    """主函数"""
    print("开始 AI API 连通性测试...")
    
    results = []
    
    # 测试 1: Qwen API
    print("\n[1/3] 测试 Qwen API 连通性...")
    results.append(test_qwen_api_connection())
    
    # 测试 2: DeepSeek API
    print("\n[2/3] 测试 DeepSeek API 连通性...")
    results.append(test_deepseek_api_connection())
    
    # 测试 3: 负面股票审计
    print("\n[3/3] 测试负面股票审计（压力点探测）...")
    results.append(test_negative_stock_audit())
    
    # 打印报告
    print_report(results)
    
    # 返回测试结果
    all_passed = all(r.success for r in results)
    if all_passed:
        print("\n🎉 所有测试通过！系统已准备好进行真实数据集成测试。")
    else:
        print("\n⚠️ 部分测试失败，请检查 API Key 配置和网络连接。")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
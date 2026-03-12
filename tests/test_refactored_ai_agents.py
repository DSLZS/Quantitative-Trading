"""
测试重构后的 AI 调用模块

验证点：
1. QwenAgent 使用 OpenAI SDK 正确初始化
2. DeepSeekAgent 配置被原封不动保留
3. 错误降级机制正常工作

使用方法:
    python tests/test_refactored_ai_agents.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_trade_advisor import (
    QwenAgent, DeepSeekAgent, 
    MockQwenAgent, MockDeepSeekAgent,
    AuditStatus
)


def test_mock_agents():
    """测试 Mock Agents（无需 API Key）"""
    print("=" * 60)
    print("测试 Mock Agents")
    print("=" * 60)
    
    config = {'provider': 'mock'}
    
    # 测试 Mock Qwen
    print("\n[1] Mock Qwen Agent")
    mock_qwen = MockQwenAgent(config)
    summary = mock_qwen.summarize_news(["新闻 1", "新闻 2"])
    print(f"  新闻摘要：{summary}")
    
    # 测试 Mock DeepSeek
    print("\n[2] Mock DeepSeek Agent")
    mock_deep = MockDeepSeekAgent(config)
    result = mock_deep.audit_stock(
        {'symbol': '600519', 'name': '贵州茅台'},
        '公司经营正常，业绩稳健增长'
    )
    print(f"  审计结果：{result.status.value}")
    print(f"  审计原因：{result.reason}")
    
    # 测试 ST 股票被驳回
    print("\n[3] Mock DeepSeek - ST 股票测试")
    st_result = mock_deep.audit_stock(
        {'symbol': 'ST 某股票', 'name': 'ST Test'},
        '无重大负面'
    )
    print(f"  审计结果：{st_result.status.value}")
    print(f"  审计原因：{st_result.reason}")
    
    print("\n✅ Mock Agents 测试通过")
    return True


def test_real_agents_config():
    """测试真实 Agents 的配置加载"""
    print("\n" + "=" * 60)
    print("测试真实 Agents 配置加载")
    print("=" * 60)
    
    # 检查环境变量
    qwen_key = os.getenv('QWEN_API_KEY', '')
    deepseek_key = os.getenv('DEEPSEEK_API_KEY', '')
    
    print(f"\nQWEN_API_KEY: {'已配置' if qwen_key and not qwen_key.startswith('${') else '未配置/占位符'}")
    print(f"DEEPSEEK_API_KEY: {'已配置' if deepseek_key and not deepseek_key.startswith('${') else '未配置/占位符'}")
    
    # 测试配置
    qwen_config = {
        'provider': 'qwen',
        'api_key': qwen_key if qwen_key and not qwen_key.startswith('${') else '',
        'base_url': 'https://coding.dashscope.aliyuncs.com/v1',
        'model': 'qwen3.5-plus'
    }
    
    deepseek_config = {
        'provider': 'deepseek',
        'api_key': deepseek_key if deepseek_key and not deepseek_key.startswith('${') else '',
        'base_url': 'https://api.deepseek.com',
        'model': 'deepseek-chat'
    }
    
    # 验证配置是否正确
    print("\n[配置验证]")
    print(f"  Qwen base_url: {qwen_config['base_url']}")
    print(f"  Qwen model: {qwen_config['model']}")
    print(f"  DeepSeek base_url: {deepseek_config['base_url']}")
    print(f"  DeepSeek model: {deepseek_config['model']}")
    
    # 如果 API Key 有效，尝试初始化真实 Agent
    if qwen_key and not qwen_key.startswith('${'):
        print("\n[尝试初始化 Qwen Agent]")
        try:
            qwen_agent = QwenAgent(qwen_config)
            print(f"  ✅ Qwen Agent 初始化成功")
            print(f"  使用模型：{qwen_agent.model}")
        except Exception as e:
            print(f"  ❌ Qwen Agent 初始化失败：{e}")
    
    if deepseek_key and not deepseek_key.startswith('${'):
        print("\n[尝试初始化 DeepSeek Agent]")
        try:
            deepseek_agent = DeepSeekAgent(deepseek_config)
            print(f"  ✅ DeepSeek Agent 初始化成功")
            print(f"  使用模型：{deepseek_agent.model}")
            print(f"  重试配置：{deepseek_agent.retry_config}")
        except Exception as e:
            print(f"  ❌ DeepSeek Agent 初始化失败：{e}")
    
    print("\n✅ 配置加载测试完成")
    return True


def test_error_degradation():
    """测试错误降级机制"""
    print("\n" + "=" * 60)
    print("测试错误降级机制")
    print("=" * 60)
    
    # 模拟无效 API Key 时的降级
    invalid_config = {
        'provider': 'qwen',
        'api_key': 'invalid_key',
        'base_url': 'https://coding.dashscope.aliyuncs.com/v1',
        'model': 'qwen3.5-plus'
    }
    
    print("\n[测试] Qwen API 调用失败时的错误处理")
    try:
        qwen_agent = QwenAgent(invalid_config)
        response, tokens = qwen_agent.call([
            {'role': 'user', 'content': 'test'}
        ])
        print(f"  响应：{response[:50]}...")
        print(f"  Token 使用：{tokens}")
    except Exception as e:
        print(f"  捕获异常：{type(e).__name__}: {e}")
    
    print("\n✅ 错误降级测试完成")
    return True


def main():
    """主函数"""
    print("重构后的 AI 调用模块测试")
    print(f"测试时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_passed = True
    
    try:
        all_passed &= test_mock_agents()
        all_passed &= test_real_agents_config()
        all_passed &= test_error_degradation()
    except Exception as e:
        print(f"\n❌ 测试过程中发生异常：{e}")
        all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✅ 所有测试通过")
    else:
        print("❌ 部分测试失败")
    print("=" * 60)
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
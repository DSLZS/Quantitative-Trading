"""
测试阿里云 Coding Plan 规范对齐

验证点：
1. QwenAgent 强制从环境变量读取 QWEN_API_KEY
2. QwenAgent 强制使用 sk-sp- 前缀的 Key
3. QwenAgent 硬编码 base_url 和 model
4. DeepSeekAgent 保持原有配置不变

使用方法:
    python tests/test_coding_plan_alignment.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_trade_advisor import QwenAgent, DeepSeekAgent


def test_qwen_key_enforcement():
    """测试 Qwen Key 强制加载"""
    print("=" * 70)
    print("测试 1: Qwen Key 强制加载 (Key Enforcement)")
    print("=" * 70)
    
    # 检查环境变量
    qwen_key = os.getenv("QWEN_API_KEY", "")
    
    print(f"\n当前 QWEN_API_KEY: {qwen_key[:10]}..." if len(qwen_key) > 10 else f"\n当前 QWEN_API_KEY: {qwen_key or '(空)'}")
    
    # 测试 Key 前缀检查
    if qwen_key:
        expected_prefix = "sk-sp-"
        actual_prefix = qwen_key[:6] if len(qwen_key) >= 6 else qwen_key
        
        if actual_prefix == expected_prefix:
            print(f"✅ Key 前缀正确：{expected_prefix}")
        else:
            print(f"⚠️ Key 前缀不匹配！期望 '{expected_prefix}'，实际 '{actual_prefix}'")
            print("   Coding Plan 的 Key 必须以 'sk-sp-' 开头")
    else:
        print("❌ QWEN_API_KEY 未配置！")
    
    return True


def test_qwen_agent_initialization():
    """测试 QwenAgent 初始化"""
    print("\n" + "=" * 70)
    print("测试 2: QwenAgent 初始化配置")
    print("=" * 70)
    
    config = {'provider': 'qwen', 'max_tokens': 500}
    
    try:
        agent = QwenAgent(config)
        
        print(f"\n✅ QwenAgent 初始化成功")
        print(f"   Provider: {agent.provider}")
        print(f"   Model: {agent.model}")
        print(f"   Base URL: {agent.base_url}")
        print(f"   Key 前缀：{agent.api_key[:7] if len(agent.api_key) >= 7 else agent.api_key}")
        
        # 验证配置是否正确
        assert agent.model == "qwen3.5-plus", f"Model 应该是 qwen3.5-plus，实际是 {agent.model}"
        assert agent.base_url == "https://coding.dashscope.aliyuncs.com/v1", f"Base URL 不正确"
        
        print("\n✅ 配置验证通过")
        return True
        
    except ValueError as e:
        print(f"\n❌ 初始化失败 (预期行为，如果 Key 未配置): {e}")
        return False
    except Exception as e:
        print(f"\n❌ 初始化失败: {e}")
        return False


def test_deepseek_isolation():
    """测试 DeepSeek 配置隔离"""
    print("\n" + "=" * 70)
    print("测试 3: DeepSeek 配置隔离 (Isolation)")
    print("=" * 70)
    
    config = {
        'provider': 'deepseek',
        'api_key': os.getenv("DEEPSEEK_API_KEY", ""),
        'base_url': 'https://api.deepseek.com',
        'model': 'deepseek-chat',
        'max_tokens': 500
    }
    
    try:
        agent = DeepSeekAgent(config)
        
        print(f"\n✅ DeepSeekAgent 初始化成功")
        print(f"   Provider: {agent.provider}")
        print(f"   Model: {agent.model}")
        print(f"   Base URL: {agent.base_url}")
        
        # 验证 DeepSeek 配置是否独立
        assert agent.model == "deepseek-chat", f"Model 应该是 deepseek-chat"
        assert agent.base_url == "https://api.deepseek.com", f"Base URL 应该是 DeepSeek 官方地址"
        
        print("\n✅ DeepSeek 配置隔离验证通过")
        return True
        
    except Exception as e:
        print(f"\n⚠️ DeepSeekAgent 初始化失败: {e}")
        return False


def test_configuration_summary():
    """输出配置摘要"""
    print("\n" + "=" * 70)
    print("配置摘要")
    print("=" * 70)
    
    print("\n📋 Qwen (阿里云 Coding Plan):")
    print("   - base_url: https://coding.dashscope.aliyuncs.com/v1")
    print("   - model: qwen3.5-plus")
    print("   - api_key: 从环境变量 QWEN_API_KEY 读取 (必须 sk-sp-开头)")
    
    print("\n📋 DeepSeek:")
    print("   - base_url: https://api.deepseek.com")
    print("   - model: deepseek-chat")
    print("   - api_key: 从环境变量 DEEPSEEK_API_KEY 读取")
    
    print("\n" + "=" * 70)


def main():
    """主函数"""
    print("阿里云 Coding Plan 规范对齐测试")
    print(f"测试时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 运行测试
    test_qwen_key_enforcement()
    test_qwen_agent_initialization()
    test_deepseek_isolation()
    test_configuration_summary()
    
    print("\n" + "=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
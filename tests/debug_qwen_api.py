"""
Qwen API 诊断脚本

用于诊断 Qwen API Key 配置问题
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv
import httpx

# Load environment variables
load_dotenv()

def diagnose_qwen_api():
    """诊断 Qwen API 配置"""
    print("=" * 70)
    print("Qwen API 诊断报告")
    print(f"诊断时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    api_key = os.getenv("QWEN_API_KEY")
    
    print(f"\n1. API Key 检查:")
    print(f"   - 是否设置：{'是' if api_key else '否'}")
    if api_key:
        print(f"   - 长度：{len(api_key)}")
        print(f"   - 前缀：{api_key[:10]}...")
        print(f"   - 后缀：...{api_key[-5:]}")
        print(f"   - 是否占位符：{'是' if api_key.startswith('${') or 'YOUR' in api_key.upper() else '否'}")
        print(f"   - 是否 sk-sp 格式：{'是' if api_key.startswith('sk-sp-') else '否'}")
    
    # 测试不同的 API 端点
    endpoints = [
        ("DashScope 兼容模式", "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"),
        ("DashScope 原生模式", "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"),
    ]
    
    if not api_key:
        print("\n❌ API Key 未设置，请检查 .env 文件")
        return False
    
    # 测试不同模型名称
    models_to_test = [
        "qwen-turbo",
        "qwen-plus", 
        "qwen-max",
        "qwen2.5-72b-instruct",
    ]
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    messages = [
        {"role": "system", "content": "你是一个助手。"},
        {"role": "user", "content": "你好"}
    ]
    
    print("\n2. 测试不同模型名称:")
    print("-" * 50)
    
    for model in models_to_test:
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": 50
        }
        
        try:
            start_time = time.time()
            with httpx.Client(timeout=15.0) as client:
                response = client.post(
                    "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
                    json=payload,
                    headers=headers
                )
                elapsed = time.time() - start_time
                
                if response.status_code == 200:
                    print(f"   ✅ {model}: 成功 ({elapsed:.2f}s)")
                    data = response.json()
                    content = data.get('choices', [{}])[0].get('message', {}).get('content', '')
                    print(f"      响应：{content[:50]}...")
                    return True
                else:
                    error_msg = response.text[:100]
                    print(f"   ❌ {model}: {response.status_code} - {error_msg}")
        except Exception as e:
            print(f"   ❌ {model}: 异常 - {e}")
    
    print("\n3. 建议操作:")
    print("-" * 50)
    print("   1. 检查 API Key 是否正确复制（无多余空格）")
    print("   2. 确认 API Key 来自 DashScope 控制台")
    print("   3. 验证账户是否有可用额度")
    print("   4. 尝试在 DashScope 控制台重新生成 API Key")
    
    return False


if __name__ == "__main__":
    diagnose_qwen_api()
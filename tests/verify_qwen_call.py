"""
Qwen 真实 API 调用验证脚本

验证 QwenAgent 的 call 方法和 summarize_news 方法是否能正常工作。

使用方法:
    python tests/verify_qwen_call.py
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from daily_trade_advisor import QwenAgent


def test_qwen_summarize_news():
    """测试 Qwen 新闻总结功能"""
    print("=" * 70)
    print("Qwen 真实 API 调用验证")
    print("=" * 70)
    
    # 模拟股票新闻
    mock_news = [
        "[2026-03-12] 贵州茅台：2025 年营收同比增长 15%，净利润再创新高",
        "[2026-03-12] 贵州茅台：拟每 10 股派发现金红利 280 元",
        "[2026-03-11] 白酒行业持续回暖，高端酒企表现亮眼",
        "[2026-03-10] 贵州茅台：暂无应披露而未披露的重大信息",
        "[2026-03-09] 机构研报：维持贵州茅台买入评级，目标价 2200 元"
    ]
    
    print(f"\n测试新闻 ({len(mock_news)} 条):")
    for news in mock_news:
        print(f"  - {news[:50]}...")
    
    # 实例化 QwenAgent
    print("\n初始化 QwenAgent...")
    config = {'provider': 'qwen', 'max_tokens': 500}
    
    try:
        agent = QwenAgent(config)
        print(f"✅ QwenAgent 初始化成功")
        print(f"   Model: {agent.model}")
        print(f"   Base URL: {agent.base_url}")
    except Exception as e:
        print(f"❌ QwenAgent 初始化失败：{e}")
        return False
    
    # 调用 summarize_news
    print("\n调用 summarize_news 方法...")
    try:
        summary = agent.summarize_news(mock_news)
        
        print("\n" + "=" * 70)
        print("✅ Qwen 生成的新闻摘要:")
        print("=" * 70)
        print(summary)
        print("=" * 70)
        
        # 验证返回结果
        if summary and len(summary) > 10:
            print("\n✅ 验证通过：Qwen 成功生成新闻摘要")
            return True
        else:
            print("\n⚠️ 验证警告：返回的摘要过短")
            return False
            
    except Exception as e:
        print(f"\n❌ API 调用失败：{e}")
        return False


def main():
    """主函数"""
    print("Qwen 真实 API 调用验证脚本")
    print(f"测试时间：{__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"QWEN_API_KEY: {os.getenv('QWEN_API_KEY', '')[:10]}...\n")
    
    success = test_qwen_summarize_news()
    
    print("\n" + "=" * 70)
    if success:
        print("✅ Qwen 链路验证成功！")
    else:
        print("❌ Qwen 链路验证失败！")
    print("=" * 70)
    
    return success


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
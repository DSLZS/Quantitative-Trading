import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

try:

    client = OpenAI(
        # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为: api_key="sk-xxx",
        api_key=os.getenv("QWEN_API_KEY"),
        base_url="https://coding.dashscope.aliyuncs.com/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3.5-plus",  # 模型列表: https://help.aliyun.com/model-studio/getting-started/models
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '你是谁？'}
        ]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"错误信息：{e}")
    print("请参考文档：https://help.aliyun.com/model-studio/developer-reference/error-code")
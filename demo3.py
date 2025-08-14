import requests
import logging
import time

API_URL = f"https://llm-365ai.openai.azure.com/openai/deployments/gpt-5/chat/completions?api-version=2025-01-01-preview"
"""处理用户查询请求"""
# 获取历史记录

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text",
             "text": "你是谁"}
        ],
    }
]

data = {
    "messages": messages,
    "temperature": 1,
    "max_completion_tokens": 1024,
    "reasoning_effort": "low"
}
HEADERS = {
    "Content-Type": "application/json",
    "api-key": "856174a8453543389bcfb57142b1076f"
}
try:
    _start = time.time()
    response = requests.post(API_URL, headers=HEADERS, json=data, timeout=None)
    response.raise_for_status()  # 确保 HTTP 请求成功
    result = response.json()['choices'][0]['message']['content']  # 返回 API 响应
    _end = time.time()
    print(result)
    print(f"耗时：{_end - _start}")
except requests.exceptions.RequestException as e:
    print('---------')

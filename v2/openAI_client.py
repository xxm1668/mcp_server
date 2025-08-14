import requests
import json
import os

api_key = "sk-da4b6cb4a41e4cascascasc9508deb556942"
headers = {
    'Authorization': f'Bearer {api_key}',
    'Content-Type': 'application/json'
}
url = "http://192.168.210.237:8000/v1/chat/completions"


def parse_sse_contents(raw_sse: str):
    """
    解析SSE流，提取并拼接所有content字段
    :param raw_sse: 多行SSE原始字符串
    :return: 拼接后的完整content字符串
    """
    contents = []
    for line in raw_sse.splitlines():
        line = line.strip()
        if not line.startswith("data: "):
            continue
        data_str = line[len("data: "):]
        if data_str == "[DONE]":
            break
        try:
            data_json = json.loads(data_str)
            delta = data_json.get("choices", [{}])[0].get("delta", {})
            if "content" in delta:
                contents.append(delta["content"])
        except json.JSONDecodeError:
            continue
    return "".join(contents)


payload = {
    'model': 'gpt-5',
    'messages': [
        {"role": "user", "content": '句子：江悦润府和万科朗时雨核想比如何？\n提取句子中的主语'}
    ],
    'temperature': '1',
    'max_tokens': '1024'
}

response = requests.post(url, json=payload, headers=headers)
response.raise_for_status()
results = response.text
result = parse_sse_contents(results)
result = result.encode("latin1").decode("utf-8")
print("原始提取:", result)

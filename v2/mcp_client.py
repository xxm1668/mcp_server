import os
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from colorama import Fore, Style
import asyncio
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "856174a8453543389bcfb57142b1076f"
AZURE_OPENAI_ENDPOINT = 'https://llm-365ai.openai.azure.com/'
AZURE_OPENAI_DEPLOYMENT = 'gpt-5'

model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview",
    temperature=1
)


async def main():
    client = MultiServerMCPClient(
        {
            "math-query": {
                "url": "http://127.0.0.1:6030/sse",
                "transport": "sse"
            }
        }
    )
    tools = await client.get_tools()
    checkpointer = InMemorySaver()
    agent = create_react_agent(
        model,
        tools,
        checkpointer=checkpointer,
        prompt="You are an AI assistant that helps people find information."
    )
    config = {
        "configurable": {
            "thread_id": "1"
        }
    }
    while True:
        question = input("请输入：")
        if not question:
            continue
        if question == "q":
            break
        async for chunk in agent.astream({"messages": [{"role": "user", "content": question}]},
                                         stream_mode="updates", config=config):
            if "agent" in chunk:
                content = chunk["agent"]["messages"][0].content
                tool_calls = chunk["agent"]["messages"][0].tool_calls
                if tool_calls:
                    for tool in tool_calls:
                        print(Fore.YELLOW, Style.BRIGHT, f">>> Call MCP Server: {tool['name']} , args: {tool['args']}")
                else:
                    print(Fore.BLACK, Style.BRIGHT, f"LLM: {content}")
            elif "tools" in chunk:
                content = chunk["tools"]["messages"][0].content
                name = chunk["tools"]["messages"][0].name
                print(Fore.GREEN, Style.BRIGHT, f"<<< {name} : {content}")


if __name__ == '__main__':
    asyncio.run(main())

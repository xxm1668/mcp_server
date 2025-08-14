# Create server parameters for stdio connection
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import asyncio
import os
from langchain.schema import HumanMessage, SystemMessage

# 替换为 DeepSeek 的 API Key 和 Base URL
os.environ["OPENAI_API_KEY"] = "8a72a5d2-002b-4649-86eb-03957430ae3c"
os.environ["OPENAI_API_BASE"] = "https://ark.cn-beijing.volces.com/api/v3"  # 假设 DeepSeek 兼容 OpenAI
model = ChatOpenAI(
    model="ep-20250328104025-xnl2f",  # DeepSeek 模型名称，比如 deepseek-chat
    temperature=0,
)

# 单轮对话
# response = model.invoke([
#     SystemMessage(content="你是一个乐于助人的助手"),
#     HumanMessage(content="帮我写一句关于春天的诗")
# ])
#
# print(response.content)

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your server.py file
    args=["server.py"],
)


async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()

            # Get tools
            tools = await load_mcp_tools(session)
            tool = next(t for t in tools if t.name == "extract_all_entity")
            # Create and run the agent
            agent = create_react_agent(model, [tool])
            agent_response = await agent.ainvoke({"messages": "识别句子中的所有楼盘名，句子：江悦润府位于哪里"})
            return agent_response


# Run the async function
if __name__ == "__main__":
    result = asyncio.run(run_agent())
    res = result['messages'][-1].content
    print(res)

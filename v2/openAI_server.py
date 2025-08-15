# openAI形势的封装
import os
from fastapi import FastAPI, HTTPException, Header, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional, Generator
from langgraph.prebuilt import create_react_agent
from langchain_mcp_adapters.client import MultiServerMCPClient
from datetime import datetime
import time, uuid
from langchain_openai import AzureChatOpenAI

os.environ["AZURE_OPENAI_API_KEY"] = "856174a8453543389bcfb57142b1076"
AZURE_OPENAI_ENDPOINT = 'https://llm-365ai.openai.azure.com/'
AZURE_OPENAI_DEPLOYMENT = 'gpt-5'

model = AzureChatOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    azure_deployment=AZURE_OPENAI_DEPLOYMENT,
    api_version="2025-01-01-preview",
    temperature=1
)

app = FastAPI(title="OpenAI Compatible Chat API")
api_key = "sk-da4b6cb4a41e4cascascasc9508deb556942"
agent = None


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[Dict[str, Any]]
    temperature: Optional[float] = 1.0
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False


class ChatCompletionChoice(BaseModel):
    index: int
    message: List[Dict[str, Any]]
    finish_reason: str


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Dict[str, int]


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


async def create_agent():
    '''创建MCP Agent'''
    client = MultiServerMCPClient(
        {
            "math-query": {
                "url": "http://127.0.0.1:6030/sse",
                "transport": "sse"
            }
        }
    )
    tools = await client.get_tools()
    return create_react_agent(
        model,
        tools,
        prompt="You are an AI assistant that helps people find information.\n针对用户的问题，你的执行流程是1.获取表结构。2.生成SQL查询数据。3.生成可视化图表。4.总结结果回复用户"
    )


async def invoke_agent(messages: []):
    global agent
    if not agent:
        agent = await create_agent()
    async for chunk in agent.astream({"messages": messages}, stream_mode="updates"):
        if "agent" in chunk:
            content = chunk["agent"]["messages"][0].content
            tool_calls = chunk["agent"]["messages"][0].tool_calls
            if tool_calls:
                for tool in tool_calls:
                    yield f"> ```Call MCP Server: {tool['name']}```\n\n"
            else:
                yield content


async def handle_stream_response(messages: [], model: str, request_id: str) -> Generator[str, None, None]:
    # 执行 agent
    async for msg in invoke_agent(messages):
        chunk_data = ChatCompletionChunk(
            id=request_id,
            created=int(time.time()),
            model=model,
            choices=[{
                "index": 0,
                "delta": {
                    "content": msg
                },
                "finish_reason": None
            }]
        )
        yield f"data: {chunk_data.model_dump_json()}\n\n"

    final_chunk = ChatCompletionChunk(
        id=request_id,
        created=int(time.time()),
        model=model,
        choices=[{
            "index": 0,
            "delta": {},
            "finish_reason": "stop"
        }]
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def verify_auth(authorization: Optional[str] = Header(None)) -> bool:
    '''验证token'''
    if not authorization:
        return False
    if authorization.startswith("Bearer "):
        token = authorization[7:]
    else:
        token = authorization
    return token == api_key


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest, auth_result: bool = Depends(verify_auth)):
    # 检查身份验证结果
    if not auth_result:
        raise HTTPException(
            status_code=401,
            detail={
                "error": {
                    "message": "Invalid authentication credentials",
                    "type": "invalid_request_error",
                    "param": None,
                    "code": "invalid_api_key"
                }
            },
            headers={"WWW-Authenticate": "Bearer"}
        )
    ## 暂不支持非流式返回
    if request.stream:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "Streaming responses are not implemented in this mock API",
                    "type": "invalid_request_error",
                    "param": "stream",
                    "code": "invalid_parameter"
                }
            }
        )
    try:
        # 触发 agent 并流式返回
        request_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
        return StreamingResponse(
            handle_stream_response(request.messages, request.model, request_id),
            media_type="text/plain",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Content-Type": "text/event-stream"
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/v1/models")
async def list_models():
    """列出可用模型"""
    return {
        "object": "list",
        "data": [
            {
                "id": "agent_model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "agent"
            }
        ]
    }


@app.get("/health")
async def health_check():
    """健康检查"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

# MCP + LangChain 调用示例

本项目演示了如何使用 **MCP（Model Context Protocol）** 与 LangChain 集成，实现本地自定义工具的调用。

### 项目结构
服务启动命令
> python3 v2/mcp_server.py[mcp_server.py](v2/mcp_server.py)
> 
> python3 v2/[openAI_server.py](v2/openAI_server.py)

供外部调用命令服务
> python3 v2/[openAI_client.py](v2/openAI_client.py)
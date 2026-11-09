from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_rag_self_tools():
    return await get_stdio_tools("self_rag_tools", "rag/self_rag.py")

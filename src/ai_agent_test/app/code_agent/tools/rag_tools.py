from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_rag_tools():
    return await get_stdio_tools("rag_tools", "rag/rag.py")

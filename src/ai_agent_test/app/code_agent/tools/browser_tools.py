from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_browser_tools():
    return await get_stdio_tools("browser_tools", "mcp/browser_tools.py")

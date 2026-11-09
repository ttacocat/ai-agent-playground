from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_terminal_tools():
    return await get_stdio_tools("terminal_tools", "mcp/terminal_tools.py")

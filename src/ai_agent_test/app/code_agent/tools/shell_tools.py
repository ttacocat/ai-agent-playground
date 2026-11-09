from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_shell_tools():
    return await get_stdio_tools("shell_tools", "mcp/shell_tools.py")

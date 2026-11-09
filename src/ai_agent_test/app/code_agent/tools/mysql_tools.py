from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_mysql_tools():
    return await get_stdio_tools("mysql_tools", "mcp/mysql_tools.py")

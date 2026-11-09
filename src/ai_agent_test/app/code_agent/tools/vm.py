from ai_agent_test.app.code_agent.utils.mcp import get_stdio_tools


async def get_stdio_vm_tools():
    return await get_stdio_tools("vm_tools", "mcp/vm.py")

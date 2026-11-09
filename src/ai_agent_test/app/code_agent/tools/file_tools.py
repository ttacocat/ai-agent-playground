from langchain_community.agent_toolkits.file_management import FileManagementToolkit

from ai_agent_test.config import settings

# 根目录锁在沙箱工作目录内。不给 agent 整个 home 的读写权限，
# 越界路径会被 Toolkit 直接拒绝，这是最省事的一道边界。
file_tools = FileManagementToolkit(root_dir=str(settings.paths.work_dir)).get_tools()

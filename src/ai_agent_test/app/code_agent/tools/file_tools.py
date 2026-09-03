from langchain_community.agent_toolkits.file_management import FileManagementToolkit

file_tools = FileManagementToolkit(root_dir="/Users/wxy/ai-agent-test/.temp").get_tools()
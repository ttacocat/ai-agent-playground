import os

from langchain_core.prompt_values import StringPromptValue
from langgraph.constants import START, END
from langgraph.graph import StateGraph, MessagesState

from ai_agent_test.app.code_agent.mcp.browser_tools import search_in_duckduckgo
from ai_agent_test.app.code_agent.model.model import qwen_llm

key_extract_query_keyword = "key_extract_query_keyword"
key_search_duck = "key_search_duck"
key_reply_user = "key_reply_user"


class WorkflowState(MessagesState):
    """继承 MessagesState，在 messages 之外扩展自己的字段"""
    keyword: str          # 提取出的搜索关键词
    search_result: str    # duckduckgo 的搜索结果


def node_extract_query_keyword(state: WorkflowState):
    question = state["messages"][-1].content
    print(question)
    prompt = StringPromptValue(text=f"请从如下信息中提取需要在duckduckgo中搜索的关键词，直接返回最终结果：{question}")
    print(prompt)
    message = qwen_llm.invoke(input=prompt)
    # 只返回需要更新的字段，LangGraph 会自动合并进 state
    return {"messages": [message], "keyword": message.content.strip()}


def node_search_duck(state: WorkflowState):
    keyword = state["keyword"]
    html = search_in_duckduckgo(keyword)
    return {"search_result": html or ""}


def node_reply_user(state: WorkflowState):
    question = state["messages"][0].content
    search_result = state["search_result"]
    result = qwen_llm.invoke(input=f"""
# 要求
请结合duckduckgo搜索的结果，回答用户的问题：{question}

搜索结果：
{search_result}
""")
    return {"messages": [result]}


def output_graph_image(graph, filename):
    try:
        png_data = graph.get_graph().draw_mermaid_png()

        output_file_dir = os.path.dirname(__file__)
        output_file_path = os.path.join(output_file_dir, filename + '.png')

        with open(output_file_path, 'wb') as f:
            f.write(png_data)

        print(f"文件写入成功：{output_file_path}")
    except Exception as e:
        print(e)


state_graph = StateGraph(WorkflowState)
state_graph.add_node(key_extract_query_keyword, node_extract_query_keyword)
state_graph.add_node(key_search_duck, node_search_duck)
state_graph.add_node(key_reply_user, node_reply_user)

state_graph.add_edge(START, key_extract_query_keyword)
state_graph.add_edge(key_extract_query_keyword, key_search_duck)
state_graph.add_edge(key_search_duck, key_reply_user)
state_graph.add_edge(key_reply_user, END)

graph = state_graph.compile()

if __name__ == "__main__":
    # output_graph_image(graph, "langgraph_workflow")

    results = graph.stream({
        "messages": [("user", "北京今天的天气如何？")]
    })

    for s in results:
        node_name = list(s)[0]
        update = s[node_name]
        print(f"[{node_name}]")
        if "keyword" in update:
            print("keyword:", update["keyword"])
        if "search_result" in update:
            print("search_result:", update["search_result"][:200], "...")
        if "messages" in update:
            print(update["messages"][-1].content)
        print("-" * 60)
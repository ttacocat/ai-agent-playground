from typing import Annotated

from mcp.server import FastMCP
from pydantic import Field

from ai_agent_test.app.code_agent.rag.rag import (
    add_document_to_index,
    bailian,
    create_client,
    get_index_job_status,
    retrieve_index,
    upload_rag_file_to_bailian,
)

mcp = FastMCP(name="self_rag")


@mcp.tool(name="query_rag", description="从百炼平台查询知识库信息")
def query_rag_from_bailian(
    query: Annotated[str, Field(description="访问知识库查询的内容", examples=["终端的操作规范"])]
) -> str:
    cfg = bailian()
    client = create_client()
    rag = retrieve_index(client, cfg.workspace_id, cfg.query_index_id, query)

    nodes = rag.body.data.nodes or []
    result = "".join(node.text for node in nodes)

    print(f"🔍 查询: {query!r} → 命中 {len(nodes)} 个节点")
    return result


@mcp.tool(name="upload_local_file_to_bailian_rag", description="将本地的知识文件上传到百炼平台知识库")
def upload_rag_from_bailian(
    file_path: Annotated[
        str,
        Field(
            description="本地知识文件的路径，推荐传绝对路径",
            examples=["<仓库根目录>/.temp/knowledge.md"],
        ),
    ]
) -> str:
    cfg = bailian()
    client = create_client()
    file_id = upload_rag_file_to_bailian(client, cfg.workspace_id, cfg.upload_category_id, file_path)
    return add_document_to_index(client, cfg.workspace_id, cfg.query_index_id, file_id)


@mcp.tool(name="query_bailian_rag_job_status", description="查询上传到百炼知识库中的知识文件的处理状态")
def query_bailian_rag_job_status(job_id: str) -> str:
    cfg = bailian()
    client = create_client()
    job_status = get_index_job_status(client, cfg.workspace_id, cfg.query_index_id, job_id)

    return job_status.body.data


if __name__ == "__main__":
    mcp.run(transport="stdio")

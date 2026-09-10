import os
from typing import Annotated

from dotenv import load_dotenv
import alibabacloud_bailian20231229.client as bailian_client
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_tea_util import models as util_models
from mcp.server.fastmcp import FastMCP
from pydantic import Field

mcp = FastMCP()

load_dotenv()

def create_client() -> bailian_client.Client:
    config = openapi_models.Config(
        access_key_id=os.getenv("ALI_ACCESS_KEY_ID"),
        access_key_secret=os.getenv("ALI_ACCESS_KEY_SECRET"),
    )
    config.endpoint = 'bailian.cn-beijing.aliyuncs.com'
    return bailian_client.Client(config)

def retrieve_index(client, workspace_id, index_id, query):
    retrieve_request = bailian_models.RetrieveRequest(
        index_id=index_id,
        query=query,
    )
    runtime = util_models.RuntimeOptions()
    return client.retrieve_with_options(
        workspace_id,
        retrieve_request,
        {},
        runtime,
    )

@mcp.tool(name="query_rag", description="从百炼平台查询知识库信息")
def query_rag_from_bailian(query: Annotated[str, Field(description="访问知识库查询的内容", examples="终端的操作规范")]) -> str:
     balian_client = create_client()
     workspace_id = "ws-kf6h0res0gvjmvt6"
     index_id = "bt2zn763dg"
     rag = retrieve_index(balian_client, workspace_id, index_id, query)

     result = ""

     for data in rag.body.data.nodes:
         result += f"""{data.text}"""

     print("[query from bailian]", query)
     print(result)
     return result

if __name__ == '__main__':
     mcp.run(transport="stdio")
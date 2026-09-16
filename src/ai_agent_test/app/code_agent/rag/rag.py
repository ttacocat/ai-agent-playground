import hashlib
import os
import time
from typing import Annotated

import alibabacloud_bailian20231229.client as bailian_client
import requests
from alibabacloud_bailian20231229 import models as bailian_models
from alibabacloud_tea_openapi import models as openapi_models
from alibabacloud_tea_util import models as util_models
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from pydantic import Field

load_dotenv()

def log_step(title: str) -> None:
    print("─" * 50)
    print(f"▶ {title}")

WORKSPACE_ID = os.getenv("WORKSPACE_ID")
QUERY_INDEX_ID = os.getenv("QUERY_INDEX_ID")
UPLOAD_CATEGORY_ID = os.getenv("UPLOAD_CATEGORY_ID")
PARSER = "DASHSCOPE_DOCMIND"

mcp = FastMCP()

def create_client() -> bailian_client.Client:
    config = openapi_models.Config(
        access_key_id=os.getenv("ALI_ACCESS_KEY_ID"),
        access_key_secret=os.getenv("ALI_ACCESS_KEY_SECRET"),
    )
    config.endpoint = "bailian.cn-beijing.aliyuncs.com"
    return bailian_client.Client(config)


def retrieve_index(client, workspace_id, index_id, query):
    request = bailian_models.RetrieveRequest(index_id=index_id, query=query)
    runtime = util_models.RuntimeOptions()
    return client.retrieve_with_options(workspace_id, request, {}, runtime)


@mcp.tool(name="query_rag", description="从百炼平台查询知识库信息")
def query_rag_from_bailian(
    query: Annotated[str, Field(description="访问知识库查询的内容", examples="终端的操作规范")]
) -> str:
    client = create_client()
    rag = retrieve_index(client, WORKSPACE_ID, QUERY_INDEX_ID, query)

    nodes = rag.body.data.nodes or []
    result = "".join(node.text for node in nodes)

    print(f"🔍 查询: {query!r} → 命中 {len(nodes)} 个节点")
    return result


def calculate_md5(file_path: str) -> str:
    """计算文件的 MD5 哈希值。"""
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_file_info(file_path):
    file_name = os.path.basename(file_path)
    file_size = str(os.path.getsize(file_path))
    file_md5 = calculate_md5(file_path)
    return file_name, file_size, file_md5


def apply_lease(client, category_id, file_name, file_md5, file_size, workspace_id):
    request = bailian_models.ApplyFileUploadLeaseRequest(
        file_name=file_name,
        md_5=file_md5,
        size_in_bytes=file_size,
    )
    print(f"  • 文件名: {file_name} | 大小: {file_size} bytes | MD5: {file_md5}")
    return client.apply_file_upload_lease_with_options(
        category_id, workspace_id, request, {}, util_models.RuntimeOptions()
    )


def apply_lease_by_file_path(client, category_id, workspace_id, file_path):
    file_name, file_size, file_md5 = get_file_info(file_path)
    return apply_lease(client, category_id, file_name, file_md5, file_size, workspace_id)


def upload_file_to_bailian(upload_url, headers, file_path):
    with open(file_path, "rb") as f:
        file_content = f.read()

    upload_headers = {
        "Content-Type": headers["Content-Type"],
        "X-bailian-extra": headers["X-bailian-extra"],
    }
    session = requests.Session()
    session.trust_env = False
    response = session.put(upload_url, data=file_content, headers=upload_headers)
    response.raise_for_status()
    print(f"  • HTTP 状态: {response.status_code}")


def add_file_to_bailian_category(client, lease_id, parser, category_id, workspace_id):
    request = bailian_models.AddFileRequest(
        lease_id=lease_id,
        parser=parser,
        category_id=category_id,
    )
    return client.add_file_with_options(workspace_id, request, {}, util_models.RuntimeOptions())


def describe_file(client, workspace_id, file_id):
    return client.describe_file_with_options(
        workspace_id, file_id, {}, util_models.RuntimeOptions()
    )

def upload_rag_file_to_bailian(client, workspace_id, category_id, file_path):
    """
    上传文件到百炼数据中心，并添加到指定分类：
    申请租约 -> 上传文件 -> 添加到分类 -> 查询状态
    """
    start = time.monotonic()

    log_step("① 申请文件租约")
    lease = apply_lease_by_file_path(client, category_id, workspace_id, file_path)
    headers = lease.body.data.param.headers
    lease_id = lease.body.data.file_upload_lease_id
    upload_url = lease.body.data.param.url

    log_step("② 上传文件到百炼")
    upload_file_to_bailian(upload_url, headers, file_path)

    log_step("③ 将文件添加到指定分类")
    add_file_result = add_file_to_bailian_category(client, lease_id, PARSER, category_id, workspace_id)
    file_id = add_file_result.body.data.file_id
    print(f"  • 文件ID: {file_id}")

    log_step("④ 查询文件处理状态")
    describe_file_response = describe_file(client, workspace_id, file_id)
    print(f"  • 状态: {describe_file_response.body.data.status}")

    print(f"✅ 上传完成，共耗时 {time.monotonic() - start:.1f}s")
    return file_id

def create_index(client, workspace_id, name, file_id, structure_type="unstructured", source_type="DATA_CENTER_FILE", sink_type="BUILT_IN"):
    headers = {}
    runtime = util_models.RuntimeOptions()
    request = bailian_models.CreateIndexRequest(
        structure_type=structure_type,
        source_type=source_type,
        sink_type=sink_type,
        name=name,
        document_ids=[file_id],
    )
    return client.create_index_with_options(workspace_id, request, headers, runtime)

def submit_index(client, workspace_id, index_id):
    headers = {}
    runtime = util_models.RuntimeOptions()
    submit_request = bailian_models.SubmitIndexJobRequest(index_id=index_id)
    return client.submit_index_job_with_options(workspace_id, submit_request, headers, runtime)

def get_index_job_status(client, workspace_id, index_id, job_id):
    headers = {}
    runtime = util_models.RuntimeOptions()

    get_index_job_status_request = bailian_models.GetIndexJobStatusRequest(
        index_id=index_id,
        job_id=job_id,
    )
    return client.get_index_job_status_with_options(workspace_id, get_index_job_status_request, headers, runtime)

def list_indices(client, workspace_id):
    headers = {}
    runtime = util_models.RuntimeOptions()
    list_indices_request = bailian_models.ListIndicesRequest()

    return client.list_indices_with_options(workspace_id, list_indices_request, headers, runtime)

def submit_index_add_documents_job(client, workspace_id, index_id, file_id, source_type="DATA_CENTER_FILE"):
    headers = {}
    runtime = util_models.RuntimeOptions()
    submit_index_add_documents_job_request = bailian_models.SubmitIndexAddDocumentsJobRequest(
        index_id=index_id,
        document_ids=[file_id],
        source_type=source_type,
    )
    return client.submit_index_add_documents_job_with_options(workspace_id, submit_index_add_documents_job_request, headers, runtime)

def add_document_to_index(client, workspace_id, category_id, file_id):
    job_result = submit_index_add_documents_job(client, workspace_id, category_id, file_id)
    job_id = job_result.body.data.id

    job_status = get_index_job_status(client, workspace_id, category_id, job_id)
    print(job_status)
if __name__ == "__main__":
    # mcp.run(transport="stdio")
    bailian_client = create_client()
    rag_file_path = "/Users/wxy/ai-agent-test/src/ai_agent_test/app/code_agent/rag/rag_test.txt"

    # try:
    #     upload_rag_file_to_bailian(bailian_client, WORKSPACE_ID, UPLOAD_CATEGORY_ID, rag_file_path)
    # except Exception as e:
    #     print(f"❌ 上传流程失败: {e}")
    #
    # res = create_index(bailian_client, WORKSPACE_ID, "智能体控制知识库", "file_04705b1cc3ab4c3bb123af0a87eadd8b_17836056")
    # print(res)
    rag_index_id = "m93lutppa4"
    # job_result = submit_index(bailian_client, WORKSPACE_ID, rag_index_id)
    # job_id = job_result.body.data.id
    # print(job_id)
    # job_id = "63cdb3072fb44926bf6de113e86f92b6"
    # job_status = get_index_job_status(bailian_client, WORKSPACE_ID, rag_index_id, job_id)
    # print(job_status.body.data)
    # list_indices_res = list_indices(bailian_client, WORKSPACE_ID)
    # print(list_indices_res.body.data)

    rag_file_id = "file_29d6dc69017c4f34b05b2bc61cd5f93f_17836056"

    res = submit_index_add_documents_job(bailian_client, WORKSPACE_ID, rag_index_id, rag_file_id)
    print(res)

import os
import httpx
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

load_dotenv()

api_key = os.environ.get("MODEL_API_KEY")
if not api_key:
    raise ValueError("请设置 MODEL_API_KEY 环境变量")

qwen_llm = ChatOpenAI(
    model="qwen3-vl-235b-a22b-thinking",
    base_url="https://ws-kf6h0res0gvjmvt6.cn-beijing.maas.aliyuncs.com/compatible-mode/v1",
    api_key=SecretStr(api_key),
    streaming=True,
    http_client=httpx.Client(trust_env=False),
    http_async_client=httpx.AsyncClient(trust_env=False),
)
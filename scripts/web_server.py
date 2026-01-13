import sys
from typing import List, Dict

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# 兼容包内/脚本运行导入
try:
    from . import rag_chat  # type: ignore
except ImportError:  # pragma: no cover
    import rag_chat  # type: ignore


class HistoryMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    history: List[HistoryMessage] = []
    query: str
    top_k: int = 5
    max_new_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9


class ChatResponse(BaseModel):
    answer: str


app = FastAPI(title="Local RAG Qwen3 Chat")

# 允许前端从 file:// 或任意域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _build_query_with_history(history: List[HistoryMessage], query: str) -> str:
    """
    简单把历史对话串进当前问题里，让模型“看到”上下文。
    RAG 检索仍然基于这个拼接后的文本。
    """
    if not history:
        return query

    lines: List[str] = ["这是一次多轮对话，请结合历史对话理解当前问题。", "", "【历史对话】"]
    for msg in history:
        role = "用户" if msg.role == "user" else "助手"
        lines.append(f"{role}：{msg.content}")

    lines.append("")
    lines.append("【当前用户问题】")
    lines.append(query)
    return "\n".join(lines)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    full_query = _build_query_with_history(req.history, req.query)
    answer = rag_chat.run_rag_chat(
        full_query,
        top_k=req.top_k,
        max_new_tokens=req.max_new_tokens,
        temperature=req.temperature,
        top_p=req.top_p,
    )
    return ChatResponse(answer=answer)


if __name__ == "__main__":
    print(
        "请使用 uvicorn 启动服务器，例如：\n"
        "  uvicorn scripts.web_server:app --host 127.0.0.1 --port 8000\n",
        file=sys.stderr,
    )



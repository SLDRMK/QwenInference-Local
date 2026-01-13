import sys
from typing import List, Dict, Any

from pathlib import Path

from llama_cpp import Llama

# 兼容包导入和脚本直接运行
try:
    from .config_llm_rag import (  # type: ignore
        LLAMA_MODEL_PATH,
        LLAMA_CTX_SIZE,
        LLAMA_N_GPU_LAYERS,
    )
except ImportError:  # pragma: no cover
    from config_llm_rag import (  # type: ignore
        LLAMA_MODEL_PATH,
        LLAMA_CTX_SIZE,
        LLAMA_N_GPU_LAYERS,
    )


_LLAMA_INSTANCE: Llama | None = None


def _load_llama() -> Llama:
    """
    加载本地 GGUF 模型（通过 llama.cpp-python），全局只初始化一次。

    注意：
    - 需要先安装：pip install llama-cpp-python
    - 需要手动下载 GGUF 文件到 LLAMA_MODEL_PATH 指定的位置。
    - 在 Mac 上若想使用 Metal/MPS 加速，通常需要本机编译启用 Metal 支持，
      这一点可参考 llama.cpp 官方文档。
    """
    global _LLAMA_INSTANCE
    if _LLAMA_INSTANCE is not None:
        return _LLAMA_INSTANCE

    model_path: Path = Path(LLAMA_MODEL_PATH).expanduser().resolve()
    if not model_path.exists():
        print(
            f"[llama_cpp] 模型文件不存在：{model_path}\n"
            "请先下载 Qwen3 GGUF（如 Qwen3-0.6B-Instruct-Q4_K_M.gguf），"
            "然后更新 config_llm_rag.LLAMA_MODEL_PATH。",
            file=sys.stderr,
        )
        raise FileNotFoundError(str(model_path))

    print(f"[llama_cpp] 加载 GGUF 模型：{model_path}", file=sys.stderr)
    _LLAMA_INSTANCE = Llama(
        model_path=str(model_path),
        n_ctx=LLAMA_CTX_SIZE,
        n_gpu_layers=LLAMA_N_GPU_LAYERS,
        # chat_models 一般需要这个
        chat_format="chatml",
    )
    return _LLAMA_INSTANCE


def chat_completion_llama(
    user_message: str,
    system_prompt: str = "你是一个专业的中文助教，请用简体中文回答问题。",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    使用 llama.cpp (GGUF) 做一次对话推理。

    这里假设使用的是 Qwen3 系列的 chat/instruct GGUF 权重，使用 ChatML 模板。
    """
    llm = _load_llama()

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    resp = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    try:
        content = resp["choices"][0]["message"]["content"]
    except Exception:
        return str(resp)
    return (content or "").strip()



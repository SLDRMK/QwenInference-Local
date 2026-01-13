import argparse
import json
from pathlib import Path
from typing import List, Dict, Tuple

import numpy as np

# 兼容包导入和脚本直接运行
try:
    from .config_llm_rag import (  # type: ignore
        RAG_CHUNKS_PATH,
        RAG_EMBEDDINGS_PATH,
    )
    from .llm_utils import (  # type: ignore
        load_qwen_llm,
        chat_completion,
    )
except ImportError:
    from config_llm_rag import (
        RAG_CHUNKS_PATH,
        RAG_EMBEDDINGS_PATH,
    )
    from llm_utils import (
        load_qwen_llm,
        chat_completion,
    )


def _load_chunks() -> List[Dict]:
    chunks: List[Dict] = []
    path = Path(RAG_CHUNKS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"未找到分块文本文件：{path}，请先运行 build_rag_index.py")
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            chunks.append(json.loads(line))
    return chunks


def _load_embeddings() -> np.ndarray:
    path = Path(RAG_EMBEDDINGS_PATH)
    if not path.exists():
        raise FileNotFoundError(f"未找到嵌入向量文件：{path}，请先运行 build_rag_index.py")
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"嵌入向量文件形状异常：{arr.shape}")
    return arr


_GLOBAL_LLM = None


def _get_llm():
    """
    简单的全局缓存，避免每次调用都重新加载 Qwen 模型。
    """
    global _GLOBAL_LLM
    if _GLOBAL_LLM is None:
        _GLOBAL_LLM = load_qwen_llm()
    return _GLOBAL_LLM


def _load_embedder():
    # 为了避免和 build_rag_index 重复依赖，这里做一个轻量级导入
    from sentence_transformers import SentenceTransformer

    try:
        from .config_llm_rag import EMBEDDING_MODEL_NAME  # type: ignore
    except ImportError:
        from config_llm_rag import EMBEDDING_MODEL_NAME

    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def retrieve(
    query: str,
    top_k: int = 5,
) -> Tuple[List[Dict], List[float]]:
    """
    从已构建的索引中检索与 query 最相关的 top_k 个文本块。
    """
    chunks = _load_chunks()
    embeddings = _load_embeddings()
    embedder = _load_embedder()

    q_emb = (
        embedder.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        .astype("float32")[0]
    )  # (dim,)

    # 余弦相似度：由于已归一化，直接点积即可
    scores_all = embeddings @ q_emb  # (N,)

    if top_k <= 0:
        top_k = 5
    top_k = min(top_k, scores_all.shape[0])

    # 取分数最高的 top_k 个索引
    top_indices = np.argpartition(-scores_all, top_k - 1)[:top_k]
    # 再按得分排序
    top_indices = top_indices[np.argsort(-scores_all[top_indices])]

    results: List[Dict] = []
    result_scores: List[float] = []
    for idx in top_indices:
        score = scores_all[idx]
        if idx < 0 or idx >= len(chunks):
            continue
        ch = chunks[int(idx)]
        ch["score"] = float(score)
        results.append(ch)
        result_scores.append(float(score))

    return results, result_scores


def run_rag_chat(
    query: str,
    top_k: int = 5,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    retrieved, _ = retrieve(query, top_k=top_k)

    if not retrieved:
        system_prompt = (
            "你是一个专业的中文助教。现在没有任何参考资料，请根据你自己的常识尽量简要回答，"
            "如果无法确定，请坦诚地说你不知道。"
        )
        llm = _get_llm()
        return chat_completion(
            llm,
            query,
            system_prompt=system_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )

    context_blocks = []
    for i, ch in enumerate(retrieved, start=1):
        src = ch.get("source", "")
        score = ch.get("score", 0.0)
        context_blocks.append(
            f"[片段 {i} | score={score:.3f} | 来源={src}]\n{ch['text']}"
        )

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "你是一个专业的中文助教，会基于给定的参考资料回答问题。\n"
        "要求：\n"
        "1. 回答时尽量引用参考资料中的关键信息做摘要，而不是胡编。\n"
        "2. 如果参考资料中没有足够信息，请明确说明“根据现有资料无法确定”。\n"
        "3. 回答使用简体中文，条理清晰、尽量简洁。"
    )

    user_message = (
        "下面是若干与问题相关的参考资料片段，请先阅读，再回答问题。\n\n"
        "【参考资料】\n"
        f"{context}\n\n"
        "【问题】\n"
        f"{query}\n\n"
        "请基于参考资料进行作答。"
    )

    llm = _get_llm()
    answer = chat_completion(
        llm,
        user_message,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
    )
    return answer


def main() -> None:
    """
    使用已构建的 RAG 索引进行一次问答。

    示例：
        conda activate vllm
        python scripts/rag_chat.py --query "请概括一下本书第一章主要讲什么？"
    """
    parser = argparse.ArgumentParser(description="基于 RAG 的本地 Qwen3 问答。")
    parser.add_argument(
        "--query",
        type=str,
        required=True,
        help="要提的问题（中文）。",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=5,
        help="每次检索的文本块数量。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="生成时的最大新 token 数。",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="采样温度，越高回答越发散。",
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.9,
        help=" nucleus 采样的 top_p 截断。",
    )

    args = parser.parse_args()

    answer = run_rag_chat(
        args.query,
        top_k=args.top_k,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    print("====== 模型回答 ======")
    print(answer)


if __name__ == "__main__":
    main()



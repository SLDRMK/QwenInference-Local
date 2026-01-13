import json
from pathlib import Path
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

# 同时兼容包内相对导入和脚本直接运行
try:
    from .config_llm_rag import (  # type: ignore
        RAG_SOURCE_FILES,
        RAG_SOURCE_DIRS,
        EMBEDDING_MODEL_NAME,
        RAG_DATA_DIR,
        RAG_INDEX_PATH,
        RAG_CHUNKS_PATH,
        RAG_EMBEDDINGS_PATH,
        ensure_directories,
    )
except ImportError:
    from config_llm_rag import (  # type: ignore
        RAG_SOURCE_FILES,
        RAG_SOURCE_DIRS,
        EMBEDDING_MODEL_NAME,
        RAG_DATA_DIR,
        RAG_INDEX_PATH,
        RAG_CHUNKS_PATH,
        RAG_EMBEDDINGS_PATH,
        ensure_directories,
    )


def _read_file(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")


def _split_into_chunks(text: str, chunk_words: int = 200, overlap_words: int = 40) -> List[str]:
    """
    简单按词数切分文本，得到适合 RAG 的小段落。
    """
    words = text.split()
    if not words:
        return []

    chunks: List[str] = []
    start = 0
    n = len(words)

    while start < n:
        end = min(n, start + chunk_words)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == n:
            break
        start = max(end - overlap_words, start + 1)

    return chunks


def _collect_all_source_files() -> List[Path]:
    """
    汇总所有需要做 RAG 的源文件：
    - config 中显式列出的 RAG_SOURCE_FILES
    - RAG_SOURCE_DIRS 下递归查找的 .txt / .md 文件
    """
    files: List[Path] = []

    # 显式文件
    files.extend(RAG_SOURCE_FILES)

    # 目录里的 .txt / .md 文件
    for directory in RAG_SOURCE_DIRS:
        if not directory.exists():
            continue
        for path in directory.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".txt", ".md"}:
                continue
            files.append(path)

    # 去重（按绝对路径）
    unique: List[Path] = []
    seen = set()
    for p in files:
        key = str(p.resolve())
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)

    return unique


def build_index() -> None:
    ensure_directories()

    all_chunks: List[Dict] = []

    print(f"[RAG] 将在目录中写入索引和分块信息：{RAG_DATA_DIR}")

    source_files = _collect_all_source_files()
    if not source_files:
        raise RuntimeError(
            "没有任何可用于构建 RAG 索引的文本块，请检查 RAG_SOURCE_FILES / RAG_SOURCE_DIRS 配置。"
        )

    for src in source_files:
        if not src.exists():
            print(f"[RAG] 跳过文件（不存在）：{src}")
            continue

        print(f"[RAG] 读取并分块：{src}")
        text = _read_file(src)
        chunks = _split_into_chunks(text)

        for c in chunks:
            all_chunks.append(
                {
                    "source": str(src),
                    "text": c,
                }
            )

    if not all_chunks:
        raise RuntimeError("没有任何可用于构建 RAG 索引的文本块，请检查 RAG_SOURCE_FILES 配置。")

    print(f"[RAG] 共得到文本块数量：{len(all_chunks)}")

    # 生成向量
    print(f"[RAG] 加载向量化模型：{EMBEDDING_MODEL_NAME}")
    embedder = SentenceTransformer(EMBEDDING_MODEL_NAME)

    texts = [c["text"] for c in all_chunks]
    embeddings = embedder.encode(
        texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    ).astype("float32")

    dim = embeddings.shape[1]
    print(f"[RAG] 向量维度：{dim}")

    # 保存向量矩阵（后续检索时直接用 NumPy 点积，无需 faiss）
    np.save(RAG_EMBEDDINGS_PATH, embeddings)
    print(f"[RAG] 已保存嵌入向量到：{RAG_EMBEDDINGS_PATH}")

    # 保存文本块
    with RAG_CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for item in all_chunks:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"[RAG] 已保存分块文本到：{RAG_CHUNKS_PATH}")


def main() -> None:
    """
    构建/更新 RAG 索引。

    使用方式：
        conda activate vllm
        python scripts/build_rag_index.py
    """
    build_index()


if __name__ == "__main__":
    main()



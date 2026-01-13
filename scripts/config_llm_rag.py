from pathlib import Path

"""
本文件集中管理本地 LLM 与 RAG 相关的路径和配置。

你只需要根据需要修改这里的常量，例如：
- 切换使用的 Qwen3 模型（0.6B / 4B）
- 增加需要做 RAG 的文本文件路径
"""


# 项目根目录（此脚本位于 project_root/scripts/ 下）
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]


#
# 大语言模型相关配置
#

# 后端类型：
# - "transformers"：使用 ModelScope + transformers 加载 Qwen/Qwen3-0.6B（当前默认）
# - "llama_cpp"：使用 llama.cpp（GGUF，本地量化，如 Qwen3-0.6B-Q4_K_M.gguf），适合 Mac + 低内存
# LLM_BACKEND: str = "transformers"
LLM_BACKEND = "llama_cpp"

#
# transformers / ModelScope 后端（Qwen3 系列）
#

# 默认使用更轻量的 Qwen3-0.6B，适合 16GB 内存的 MacBook Air M4
# 如果后续确认 4B 也能稳定运行，可将下面的 ID 改为 Qwen3-4B 对应的 ModelScope 模型名。
# 例：QWEN_MODEL_ID = "Qwen/Qwen3-4B-Instruct"  （具体名字以 ModelScope 页面为准）
QWEN_MODEL_ID: str = "Qwen/Qwen3-0.6B"

# 使用 ModelScope 下载 Qwen 模型时的缓存目录
QWEN_MODEL_CACHE_DIR: Path = PROJECT_ROOT / "models" / "qwen3"

#
# llama.cpp / GGUF 后端（推荐在 Mac 上使用量化 Qwen / Qwen2.5）
#

# 举例：你可以下载 HuggingFace 上的
#   qwen2.5-3b-instruct-q4_k_m.gguf  （来自 Qwen2.5-3B-Instruct-GGUF 仓库）
# 放在 models/gguf/ 目录下，或根据实际文件名修改下面路径。
LLAMA_MODEL_PATH: Path = (
    PROJECT_ROOT
    / "models"
    / "gguf"
    / "qwen2.5-3b-instruct-q4_k_m.gguf"
)

# 上下文长度（视 GGUF 模型和显存情况调整）
LLAMA_CTX_SIZE: int = 4096

# Mac 上使用 Metal 时，n_gpu_layers = -1 通常表示尽量把所有层放到 GPU；
# 也可以根据稳定性酌情改为 0（全部 CPU）或较小的正整数。
LLAMA_N_GPU_LAYERS: int = -1


#
# RAG 检索相关配置
#

# 使用的文本向量化模型（Sentence-Transformers）
# 这里选用多语言 MiniLM，体积较小、支持中文，后续可以自行替换为更强的模型。
EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# RAG 索引与分块文本存储目录
RAG_DATA_DIR: Path = PROJECT_ROOT / "rag_data"
RAG_INDEX_PATH: Path = RAG_DATA_DIR / "faiss.index"
RAG_CHUNKS_PATH: Path = RAG_DATA_DIR / "chunks.jsonl"
RAG_EMBEDDINGS_PATH: Path = RAG_DATA_DIR / "embeddings.npy"


# 需要做 RAG 的**单个**原始文本文件列表
# （适合少量大文件，如完整教材等）
RAG_SOURCE_FILES = [
    PROJECT_ROOT / "PMPP-3rd-Edition.txt",
    PROJECT_ROOT
    / "General-purpose graphics processor architectures (Aamodt, Tor M. Fung, Wilson Wai Lun Rogers etc.) (Z-Library).txt",
    # 以后在这里追加新的 .txt 文件路径，例如：
    # PROJECT_ROOT / "datasets" / "section_1.txt",
]


# 需要做 RAG 的**目录**列表
# 典型用途：
# - 下载的 Triton 文档（转成 .txt / .md 后放在 docs/triton/）
# - 下载的 tilelang 文档（docs/tilelang/）
# - 以后其他本地技术文档
#
# build_rag_index.py 会自动递归遍历这些目录下的所有 .txt / .md 文件。
RAG_SOURCE_DIRS = [
    PROJECT_ROOT / "docs",
]


def ensure_directories() -> None:
    """确保相关目录存在。"""
    QWEN_MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    RAG_DATA_DIR.mkdir(parents=True, exist_ok=True)



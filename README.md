## 项目说明：本地 Qwen3 推理与 RAG 框架（MacBook Air M4）

本目录下提供了一套在 **MacBook Air M4 16GB** 上运行的本地大模型（Qwen3 系列）推理与 RAG（检索增强生成）基础框架，代码集中放在 `scripts` 目录中，并以 `PMPP-3rd-Edition.txt` 等学习资料为例做检索问答。

---

### 一、目录结构（与本框架相关部分）

- **`scripts/`**
  - `extract_docx_text.py`：将 `.docx` 转为 `.txt` 的小工具（纯 Python，无第三方依赖），可用于把课件/教材转成纯文本。
  - `config_llm_rag.py`：统一配置文件，管理模型 ID、缓存路径、RAG 索引路径、需要做 RAG 的源文件列表等。
  - `llm_utils.py`：本地 Qwen3 大模型的加载与推理工具：
    - 通过 ModelScope 下载/缓存 Qwen3 模型；
    - 自动选择设备（优先 `mps`，其次 `cuda`，最后 `cpu`）；
    - 提供 `chat_completion` 接口和一个简单 CLI 对话入口。
  - `build_rag_index.py`：从若干 `.txt` 材料中切分段落、生成向量，并使用 FAISS 构建 RAG 检索索引。
  - `rag_chat.py`：基于已构建的 RAG 索引与本地 Qwen3 模型进行问答。
  - `download_docs.py`：用 Python 把 **Triton 文档**（`https://triton-lang.org/main/index.html`）和 **tilelang 文档**（GitHub `tile-ai/tilelang` 仓库下的 `docs/`）抓到本地 `docs/` 目录，作为 RAG 语料。

- **`PMPP-3rd-Edition.txt`**
  - 从教材/文档提取出的纯文本示例，用作 RAG 的初始语料。

- 运行过程中会自动生成的目录与文件（首次运行后出现）：
  - **`models/qwen3/`**：通过 ModelScope 下载的 Qwen3 模型缓存目录。
  - **`rag_data/`**
    - `faiss.index`：RAG 的向量索引文件。
    - `chunks.jsonl`：RAG 切分后的文本块（每行一个 JSON，包含 `source` 和 `text` 字段）。
  - **`docs/`**
    - `triton/`：通过 `download_docs.py` 抓取、并抽取文本后的 Triton 文档（若已执行）。
    - `tilelang/`：通过 `download_docs.py` 抓取的 tilelang 文档（若已执行）。

---

### 二、基础环境配置（Conda 环境：`vllm`）

> 以下命令均在项目根目录（即本 `README.md` 所在目录）下执行。

1. **创建并激活 Conda 环境**

```bash
conda create -n vllm python=3.10
conda activate vllm
```

2. **安装依赖**

```bash
# PyTorch（Mac 上会自动使用 MPS，在 M 系列 GPU 上运行）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 其余 Python 依赖（推荐直接使用 requirements.txt）
pip install -r requirements.txt
```

> 说明：vLLM 服务器暂未在本框架中使用，目前是直接用 Transformers + ModelScope 在本地加载 Qwen3 进行推理。

---

### 三、配置说明：`scripts/config_llm_rag.py`

主要配置项说明（只列关键的）：

- **项目根目录**

```python
PROJECT_ROOT = Path(__file__).resolve().parents[1]
```

- **Qwen 模型相关**

```python
# 默认使用的模型（ModelScope 的模型 ID），适合 16GB 内存
QWEN_MODEL_ID = "Qwen/Qwen3-0.6B"

# 模型缓存目录，会通过 ModelScope 下载到这里
QWEN_MODEL_CACHE_DIR = PROJECT_ROOT / "models" / "qwen3"
```

如需尝试更大的 Qwen3-4B 模型，只需要将 `QWEN_MODEL_ID` 修改为对应的 ModelScope 模型名（例如 `"Qwen/Qwen3-4B-Instruct"`，以官网为准），其余代码不变。但在 16GB 内存的 MacBook Air 上可能会因显存/内存不足而失败或运行缓慢。

- **RAG 相关配置**

```python
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

RAG_DATA_DIR = PROJECT_ROOT / "rag_data"
RAG_INDEX_PATH = RAG_DATA_DIR / "faiss.index"
RAG_CHUNKS_PATH = RAG_DATA_DIR / "chunks.jsonl"

RAG_SOURCE_FILES = [
    PROJECT_ROOT / "PMPP-3rd-Edition.txt",
    PROJECT_ROOT / "General-purpose graphics processor architectures (Aamodt, Tor M. Fung, Wilson Wai Lun Rogers etc.) (Z-Library).txt",
    # 以后可在此追加更多单独的 .txt 资料
]

RAG_SOURCE_DIRS = [
    PROJECT_ROOT / "docs",
]
```

如果你新增了其它教材/讲义/技术文档，有两种方式接入 RAG：

1. **单文件方式**：  
   - 用 `extract_docx_text.py` 或其他方式转成 `.txt`；  
   - 把新 `.txt` 文件路径加到 `RAG_SOURCE_FILES` 列表中；  
   - 重新运行 `build_rag_index.py`。
2. **目录方式（推荐用于大量文档）**：  
   - 将 `.txt` / `.md` 文件放到 `docs/` 下任意子目录（如 `docs/triton/`、`docs/tilelang/`）；  
   - 确保 `RAG_SOURCE_DIRS` 中包含 `PROJECT_ROOT / "docs"`；  
   - 重新运行 `build_rag_index.py`。

---

### 四、设备选择与 GPU 加速（M4 / MPS）

在 `llm_utils.py` 中，设备选择逻辑为：

1. 若 `torch.backends.mps.is_available()` 为 `True` → 使用 `mps`（Apple GPU，加速推理）；  
2. 否则若 `torch.cuda.is_available()` → 使用 `cuda`；  
3. 否则回退到 `cpu`。

在本机 MacBook Air M4 上，只要 PyTorch 安装正确、MPS 可用，则本地 Qwen3 推理会自动跑在 GPU 上。

可用以下命令简单检查：

```bash
conda activate vllm
python -c "import torch; print(torch.backends.mps.is_available())"
```

若输出为 `True`，说明 MPS 可用。

---

### 五、常用脚本与指令

以下所有命令均假设你已经：

- 在项目根目录下；
- 已执行 `conda activate vllm` 并安装完依赖。

#### 1. 测试本地 Qwen3 推理：`scripts/llm_utils.py`

- **方式一：命令行传入 prompt**

```bash
python scripts/llm_utils.py --prompt "你好，简单介绍一下 GPU 与 CPU 的主要区别。"
```

- **方式二：从标准输入读取（可输入多行，Ctrl+D 结束）**

```bash
python scripts/llm_utils.py
# 然后在终端中输入问题，回车，最后 Ctrl+D 结束输入
```

脚本会自动：

1. 使用 ModelScope 下载/加载 `Qwen/Qwen3-0.6B` 到 `models/qwen3`；
2. 自动选择设备（优先 MPS）；
3. 按预设的系统提示词，以简体中文输出回答。

#### 2. 构建/更新 RAG 索引：`scripts/build_rag_index.py`

当你新增或修改了 RAG 语料（例如修改了 `PMPP-3rd-Edition.txt`，或添加了新的 `.txt` 文件并更新了 `RAG_SOURCE_FILES`），需要重新构建索引：

```bash
python scripts/build_rag_index.py
```

作用：

1. 读取 `RAG_SOURCE_FILES` 中列出的所有 `.txt` 文件；
2. 递归遍历 `RAG_SOURCE_DIRS`（默认是 `docs/`）下所有 `.txt` / `.md` 文件；
2. 将每个文件按词数切分成多个小段（支持一定的重叠）；
3. 用 `SentenceTransformer` 生成向量，并存入 FAISS 索引；
4. 将索引写入 `rag_data/faiss.index`，并将文本块写入 `rag_data/chunks.jsonl`。

#### 3. 基于 RAG 的问答：`scripts/rag_chat.py`

在完成索引构建后，可用下列命令进行基于资料的问答：

```bash
python scripts/rag_chat.py --query "简单概括一下本书第一章主要讲的内容。"
```

可选参数：

- `--top_k`：检索的最相关文本块数量（默认 5）

示例：

```bash
python scripts/rag_chat.py --query "GPU 与 CPU 在并行处理上的区别是什么？" --top_k 8
```

内部流程：

1. 从 `rag_data/faiss.index` 与 `rag_data/chunks.jsonl` 中加载索引与文本块；
2. 对用户问题向量化，检索最相关的若干段落；
3. 将这些段落作为「参考资料」，组合成 Prompt 传给本地 Qwen3 模型；
4. 模型基于资料进行中文回答，且被要求不要胡编（缺少信息时要明确说明）。

---

### 六、将新材料接入 RAG 的步骤总结

1. 准备原始资料：
   - 若是 `.docx`，可使用：
     ```bash
     python scripts/extract_docx_text.py 你的文件.docx -o 你的文件.txt
     ```
   - 若是 PDF，可先手动或用工具转成 `.txt`。
2. **如果是少量大文件**（比如一本书）：  
   - 将新 `.txt` 路径加入 `scripts/config_llm_rag.py` 中的 `RAG_SOURCE_FILES` 列表。
3. **如果是一堆碎片化文档/笔记/教程**（如 Triton、tilelang 文档）：  
   - 统一放到 `docs/` 下（可以分子目录）；  
   - 或运行：
     ```bash
     python scripts/download_docs.py          # 自动抓取 Triton + tilelang 在线文档
     ```
4. 重新构建索引：
   ```bash
   conda activate vllm
   python scripts/build_rag_index.py
   ```
5. 使用 `rag_chat.py` 提问，系统即可同时利用旧资料和新资料进行检索增强问答。

---

### 七、可能的后续扩展方向（留作以后改进）

- 在当前基础上增加：
  - Web 界面（如 Gradio/FastAPI + 前端）实现浏览器内问答；
  - 多轮对话状态管理（记忆历史对话）；
  - 更细粒度/结构化的教材章节索引（按章节/小节组织 RAG 语料）。
- 替换或增加更强的向量模型（如 bge-m3 等）以提高检索质量。

当前版本已经提供了：

- 本地 Qwen3 模型加载与推理（优先 GPU/MPS）；  
- 基于教材文本的 RAG 检索与问答；  
- 清晰的配置与脚本结构，便于后续扩展与维护。


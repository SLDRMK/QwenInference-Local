import os
import sys
from dataclasses import dataclass
from typing import Optional, List, Dict, Any

import torch
from modelscope import snapshot_download
from transformers import AutoTokenizer, AutoModelForCausalLM

# 既支持作为包模块导入（python -m scripts.llm_utils），
# 也支持直接脚本运行（python scripts/llm_utils.py）。
try:
    from .config_llm_rag import (  # type: ignore
        QWEN_MODEL_ID,
        QWEN_MODEL_CACHE_DIR,
        ensure_directories,
    )
except ImportError:
    from config_llm_rag import (  # type: ignore
        QWEN_MODEL_ID,
        QWEN_MODEL_CACHE_DIR,
        ensure_directories,
    )


@dataclass
class LLMResources:
    tokenizer: AutoTokenizer
    model: AutoModelForCausalLM
    device: str


def _select_device() -> str:
    """
    选择推理设备，支持通过环境变量强制指定：

    - LLM_DEVICE=cpu / mps / cuda
    - 若未指定，则优先 mps，其次 cuda，最后 cpu。
    """
    forced = os.environ.get("LLM_DEVICE") or os.environ.get("FORCE_DEVICE")
    if forced:
        forced = forced.lower()
        if forced in {"cpu", "cuda", "mps"}:
            return forced

    # 默认策略：优先 MPS，其次 CUDA，最后 CPU
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_qwen_llm(model_id: Optional[str] = None) -> LLMResources:
    """
    通过 ModelScope 下载并加载 Qwen 模型，返回分词器和模型。

    - 默认使用 config_llm_rag.QWEN_MODEL_ID
    - 会把模型缓存到 config_llm_rag.QWEN_MODEL_CACHE_DIR 下
    """
    ensure_directories()

    mid = model_id or QWEN_MODEL_ID

    print(f"[LLM] 使用 ModelScope 下载/加载模型：{mid}", file=sys.stderr)
    model_dir = snapshot_download(
        mid,
        cache_dir=str(QWEN_MODEL_CACHE_DIR),
    )

    device = _select_device()
    print(f"[LLM] 使用设备：{device}", file=sys.stderr)

    tokenizer = AutoTokenizer.from_pretrained(
        model_dir,
        trust_remote_code=True,
    )

    # 对于 Mac M 系列，优先使用 float16 + MPS；CPU 则用 float32
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.to(device)
    model.eval()

    return LLMResources(tokenizer=tokenizer, model=model, device=device)


def chat_completion(
    llm: LLMResources,
    user_message: str,
    system_prompt: str = "你是一个专业的中文助教，请用简体中文回答问题。",
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
) -> str:
    """
    使用 Qwen 做一次简单的单轮对话推理。

    · 默认按 chat 模板走，如果模型未提供 apply_chat_template，则退回到简单 prompt。
    """
    tokenizer = llm.tokenizer
    model = llm.model
    device = llm.device

    messages: List[Dict[str, Any]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_message})

    try:
        # 优先尝试官方 chat 模板
        prompt_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    except Exception:
        # 兼容一些不带模板的权重
        if system_prompt:
            prompt_text = f"{system_prompt}\n\n用户：{user_message}\n助手："
        else:
            prompt_text = f"用户：{user_message}\n助手："

    inputs = tokenizer(
        prompt_text,
        return_tensors="pt",
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    answer = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )

    # 在 MPS 上适当释放缓存，减缓显存/内存持续上涨的问题
    if device == "mps" and hasattr(torch, "mps"):
        try:  # 防御性调用，避免因不同版本 PyTorch 报错
            torch.mps.empty_cache()
        except Exception:
            pass

    return answer.strip()


def simple_cli_chat() -> None:
    """
    命令行下进行一次简单对话：

    python -m scripts.llm_utils "你好，介绍一下本书的大致内容"
    或：
    python scripts/llm_utils.py --prompt "..."
    """
    import argparse

    parser = argparse.ArgumentParser(description="使用本地 Qwen3 做一次简单问答。")
    parser.add_argument(
        "--prompt",
        type=str,
        help="用户问题，如果不提供则从标准输入读取。",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="最大生成 token 数。",
    )
    args = parser.parse_args()

    if args.prompt:
        question = args.prompt
    else:
        print("请输入问题，然后回车（Ctrl+D 结束）：", file=sys.stderr)
        question = sys.stdin.read().strip()

    if not question:
        print("未提供任何问题。", file=sys.stderr)
        sys.exit(1)

    llm = load_qwen_llm()
    answer = chat_completion(llm, question, max_new_tokens=args.max_new_tokens)

    print("====== 模型回答 ======")
    print(answer)


if __name__ == "__main__":
    simple_cli_chat()



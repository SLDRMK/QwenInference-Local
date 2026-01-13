import argparse
import sys
from dataclasses import dataclass
from typing import Set, List

from pathlib import Path
import shutil
import subprocess

import requests
from bs4 import BeautifulSoup

# 兼容包导入和脚本直接运行
try:
    from .config_llm_rag import PROJECT_ROOT  # type: ignore
except ImportError:  # pragma: no cover
    from config_llm_rag import PROJECT_ROOT  # type: ignore


REQUEST_TIMEOUT = 15


@dataclass
class TritonConfig:
    start_url: str = "https://triton-lang.org/main/index.html"
    max_pages: int = 80
    save_dir: Path = PROJECT_ROOT / "docs" / "triton"


@dataclass
class TileLangConfig:
    repo_url: str = "https://github.com/tile-ai/tilelang.git"
    save_dir: Path = PROJECT_ROOT / "docs" / "tilelang"


def _http_get(url: str) -> str:
    """简单封装：带超时的 GET，请求失败时给出清晰错误。"""
    resp = requests.get(url, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    return resp.text


# ---------------- Triton 文档抓取 ----------------

def download_triton_docs(cfg: TritonConfig) -> None:
    """
    递归抓取 Triton 文档站（主域名 triton-lang.org/main/），
    用 BeautifulSoup 提取纯文本，保存为 .txt 到 cfg.save_dir 下。

    说明：
    - 这是一个「尽力而为」的小爬虫，只抓取有限页数（max_pages），避免跑飞。
    - 只跟进同一域名、路径前缀为 /main/ 的链接。
    """
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    visited: Set[str] = set()
    to_visit: List[str] = [cfg.start_url]

    print(f"[Triton] 文本将保存到目录: {cfg.save_dir}")

    while to_visit and len(visited) < cfg.max_pages:
        url = to_visit.pop(0)
        if url in visited:
            continue
        visited.add(url)

        print(f"[Triton] 抓取 {len(visited)}/{cfg.max_pages}: {url}")
        try:
            html = _http_get(url)
        except Exception as exc:  # pragma: no cover - 运行时网络错误
            print(f"[Triton] 失败: {url} -> {exc}", file=sys.stderr)
            continue

        soup = BeautifulSoup(html, "lxml")

        # 尝试只取主内容区域，减少导航/页脚噪音
        main_node = soup.find(attrs={"role": "main"}) or soup.find("main")
        if main_node is not None:
            text = main_node.get_text(separator="\n")
        else:
            text = soup.get_text(separator="\n")
        text = text.strip()
        if not text:
            continue

        # 把 URL 映射成一个相对安全的文件名
        filename = _triton_url_to_filename(url)
        out_path = cfg.save_dir / filename
        out_path.write_text(text, encoding="utf-8")

        # 继续抓取站内链接
        base = "https://triton-lang.org"
        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            if href.startswith("#") or href.startswith("mailto:"):
                continue
            # 绝对 / 相对
            if href.startswith("/"):
                full = base + href
            elif href.startswith("http://") or href.startswith("https://"):
                full = href
            else:
                # 相对路径
                if url.endswith("/"):
                    full = url + href
                else:
                    full = url.rsplit("/", 1)[0] + "/" + href

            # 只保留 triton-lang.org/main 下的页面
            if not full.startswith("https://triton-lang.org/"):
                continue
            if "/main/" not in full:
                continue
            if full in visited or full in to_visit:
                continue

            to_visit.append(full)


def _triton_url_to_filename(url: str) -> str:
    """
    根据 URL 生成一个 .txt 文件名。
    例如：
        https://triton-lang.org/main/index.html -> index.txt
        https://triton-lang.org/main/intro.html -> intro.txt
        https://triton-lang.org/main/guide/install/ -> guide_install.txt
    """
    # 去掉 query / fragment
    clean = url.split("?", 1)[0].split("#", 1)[0]
    parts = clean.rstrip("/").split("/")
    # 取 main 后面的那部分路径
    try:
        idx = parts.index("main")
        tail = parts[idx + 1 :]
    except ValueError:
        tail = parts[-1:]

    if not tail:
        name = "index"
    else:
        last = tail[-1]
        if "." in last:
            last = last.rsplit(".", 1)[0]
        if not last:
            last = "index"
        if len(tail) > 1:
            prefix = "_".join(tail[:-1])
            name = f"{prefix}_{last}"
        else:
            name = last

    # 保守替换非法字符
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in name)
    return f"{safe}.txt"


# ---------------- tilelang 文档抓取 ----------------

def download_tilelang_docs(cfg: TileLangConfig) -> None:
    """
    使用 git 克隆 tilelang 仓库，然后把仓库里的 docs/ 目录
    拷贝到本项目的 docs/tilelang/ 下，作为 RAG 语料。

    注意：
    - 需要本机安装 git 并且可以访问 GitHub。
    - 若对网络不放心，也可以手动 git clone，再把 docs/ 复制到本项目 docs/tilelang/。
    """
    cfg.save_dir.mkdir(parents=True, exist_ok=True)

    # 简单检测是否有 git
    if shutil.which("git") is None:
        print(
            "[tilelang] 未找到 git，请手动执行：\n"
            "  git clone https://github.com/tile-ai/tilelang.git\n"
            "然后将该仓库下的 docs/ 拷贝到本项目的 docs/tilelang/ 目录。",
            file=sys.stderr,
        )
        return

    tmp_repo = PROJECT_ROOT / ".tmp_tilelang_repo"
    if tmp_repo.exists():
        shutil.rmtree(tmp_repo)

    print(f"[tilelang] 使用 git 克隆仓库到临时目录: {tmp_repo}")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", cfg.repo_url, str(tmp_repo)],
            check=True,
        )
    except Exception as exc:  # pragma: no cover
        print(f"[tilelang] git clone 失败: {exc}", file=sys.stderr)
        return

    src_docs = tmp_repo / "docs"
    if not src_docs.exists():
        print(f"[tilelang] 仓库中未找到 docs/ 目录: {src_docs}", file=sys.stderr)
        return

    print(f"[tilelang] 拷贝 {src_docs} -> {cfg.save_dir}")
    # Python 3.8+ 支持 dirs_exist_ok
    shutil.copytree(src_docs, cfg.save_dir, dirs_exist_ok=True)

    # 可以按需保留 repo 调试，这里默认删除
    shutil.rmtree(tmp_repo, ignore_errors=True)


# ---------------- CLI ----------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "下载 Triton 和 tilelang 的在线文档到本地 docs/ 目录，"
            "用于离线 RAG 检索。需要网络、requests、beautifulsoup4、lxml 等依赖。"
        )
    )
    parser.add_argument(
        "--triton",
        action="store_true",
        help="仅下载 Triton 文档",
    )
    parser.add_argument(
        "--tilelang",
        action="store_true",
        help="仅下载 tilelang 文档",
    )

    args = parser.parse_args()

    # 默认：两个都下
    do_triton = args.triton or (not args.triton and not args.tilelang)
    do_tilelang = args.tilelang or (not args.triton and not args.tilelang)

    if do_triton:
        download_triton_docs(TritonConfig())
    if do_tilelang:
        download_tilelang_docs(TileLangConfig())


if __name__ == "__main__":
    main()



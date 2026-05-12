#!/usr/bin/env python3
"""
合并由 api_server_deepresearch_redis.py 中 RedisCacheManager.sync_to_json 落盘的缓存 JSON。

格式：顶层为一个 JSON object，key 为 Redis 键（如 dr:search:...、dr:visit_raw:...），
value 为字符串（visit 可能为 Z: 开头的压缩串），与 load_from_json 兼容。

示例:
  python merge_deepresearch_cache_json.py a.json b.json -o merged.json
  python merge_deepresearch_cache_json.py dumps/*.json -o warm.json --keep last --prefix dr:
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple


def _load_cache_dict(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: 顶层必须是 JSON object（dict）")
    return data


def merge_deepresearch_caches(
    paths: List[Path],
    *,
    keep: str,
    key_prefix: str | None,
) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """
    keep: 'first' 保留先出现文件的值；'last' 保留后出现文件的值（同 key 冲突时）。
    key_prefix: 若设置，只合并以此开头的键（与 DEEP_RESEARCH_CACHE_PREFIX 一致时常用 dr:）。
    """
    merged: Dict[str, Any] = {}
    stats = {
        "files_read": 0,
        "entries_seen": 0,
        "entries_merged": 0,
        "conflicts_same_key_diff_value": 0,
    }

    for p in paths:
        data = _load_cache_dict(p)
        stats["files_read"] += 1
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            if key_prefix is not None and not k.startswith(key_prefix):
                continue
            stats["entries_seen"] += 1

            if keep == "last":
                if k in merged and merged[k] != v:
                    stats["conflicts_same_key_diff_value"] += 1
                merged[k] = v
            else:
                if k in merged:
                    if merged[k] != v:
                        stats["conflicts_same_key_diff_value"] += 1
                else:
                    merged[k] = v

    stats["entries_merged"] = len(merged)
    return merged, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="合并多个 DeepResearch Redis 缓存落盘 JSON 为一个 dict（可供 WARM_START_FILE 使用）。"
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="输入 JSON 路径（一个或多个）")
    parser.add_argument("-o", "--output", type=Path, required=True, help="输出 JSON 路径")
    parser.add_argument(
        "--keep",
        choices=("first", "last"),
        default="first",
        help="同一 key 多处出现时保留首次或末次（默认 first）",
    )
    parser.add_argument(
        "--prefix",
        default=None,
        metavar="PREFIX",
        help="只合并键名以前缀开头的项（例如与默认 Redis 前缀一致时写 dr:）；不设则合并全部键",
    )
    parser.add_argument(
        "--indent",
        type=int,
        default=None,
        metavar="N",
        help="若指定正整数，输出美化缩进（默认无缩进，体积小、与 sync 风格接近）",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="不打印统计信息",
    )
    args = parser.parse_args()

    paths = [p.expanduser().resolve() for p in args.inputs]
    for p in paths:
        if not p.is_file():
            print(f"错误: 不是文件或不存在: {p}", file=sys.stderr)
            sys.exit(1)

    merged, stats = merge_deepresearch_caches(
        paths,
        keep=args.keep,
        key_prefix=args.prefix,
    )

    out = args.output.expanduser().resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(merged, f, ensure_ascii=False, indent=args.indent)

    if not args.quiet:
        print(
            f"已写入 {out} | 文件数={stats['files_read']} "
            f"扫描条目={stats['entries_seen']} 合并后键数={stats['entries_merged']} "
            f"冲突(同键不同值)={stats['conflicts_same_key_diff_value']} keep={args.keep}"
        )


if __name__ == "__main__":
    main()

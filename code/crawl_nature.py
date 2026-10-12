#weijinzi
# 用途：从 Crossref 抓取近 18 个月 Nature 正刊与 Nature Machine Intelligence 的 AI 论文。
# 前因：用户要求渠道覆盖 Nature 与 Nature 子刊（选定正刊 + NMI），补齐顶刊来源。
# 原因：Nature 官网有付费墙，Crossref 按 ISSN 提供开放元数据，免费稳定无需密钥。
# 内容：按 ISSN + 发表日期窗口分页拉取期刊论文（journal-article 类型），
#       Nature 正刊用 AI 关键词过滤标题与摘要，NMI 本身即 AI 期刊全量保留；
#       其中 agent/harness 与组合优化主题的论文分别挪入专属目录。
# 行为：覆盖式落盘 Nature/nature_window.json 与 Nature/nmi_window.json，
#       agent 论文 → agent_and_harness/nature_agent.json，组合优化 → combinatorial_optimization/nature_co.json。
# 使用：python crawl_nature.py [--window-days 545] [--max-items N] [--dry-run]。
# 依赖：requests；Crossref 429/5xx 时按 Retry-After 退避重试，仍失败则中断（失败要响）。
# ---

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# ============ 配置区：期刊、过滤词表（修改只需改这里） ============

JOURNALS = [
    {"key": "nature", "name": "Nature", "issn": "1476-4687", "ai_filter": True},
    {"key": "nmi", "name": "Nature Machine Intelligence", "issn": "2522-5839", "ai_filter": False},
]
WINDOW_DAYS = 545  # 约 18 个月

CROSSREF_API = "https://api.crossref.org/journals/{issn}/works"
CR_PAGE_SIZE = 200
CR_INTERVAL = 1.0
CR_MAX_RETRIES = 4

# Nature 正刊的 AI 相关词表（标题 + 摘要匹配）
AI_RE = re.compile(
    r"\b(machine learning|deep learning|neural network|artificial intelligence|"
    r"large language model|reinforcement learning|foundation model|generative ai|"
    r"\bgpt\b|\bllm\b|diffusion model|transformer|computer vision|"
    r"language model|ai system)\b",
    re.IGNORECASE,
)
AGENT_RE = re.compile(
    r"\b(agent|harness|tool[- ]?use|function[- ]?call|multi[- ]?agent|"
    r"code[- ]?agent|computer[- ]?use|foundation model)\b",
    re.IGNORECASE,
)
CO_RE = re.compile(
    r"\b(combinatorial optimization|mixed[- ]integer|branch[- ]and[- ]bound|"
    r"vehicle routing|constraint programming|sat solving|"
    r"travelling salesman|traveling salesman|bin packing|job shop)\b",
    re.IGNORECASE,
)

# ================================================================


def strip_jats(text: str) -> str:
    """去掉 Crossref abstract 里的 JATS XML 标签。"""
    return re.sub(r"<[^>]+>", " ", text or "").strip()


def crossref_fetch_journal(issn: str, start: datetime, end: datetime, max_items: int) -> list[dict[str, Any]]:
    """按 ISSN + 日期窗口分页拉取论文元数据。"""
    session = requests.Session()
    items: list[dict[str, Any]] = []
    cursor = "*"
    while True:
        params = {
            "filter": (
                f"from-pub-date:{start.date().isoformat()},"
                f"until-pub-date:{end.date().isoformat()},type:journal-article"
            ),
            "rows": CR_PAGE_SIZE,
            "cursor": cursor,
            "select": "title,abstract,DOI,URL,issued,author,container-title",
        }
        for attempt in range(CR_MAX_RETRIES):
            try:
                resp = session.get(CROSSREF_API.format(issn=issn), params=params, timeout=60)
                if resp.status_code in (429, 500, 502, 503):
                    wait = float(resp.headers.get("Retry-After", 5 * (attempt + 1)))
                    print(f"[warn] Crossref {resp.status_code}，等待 {wait}s 重试", file=sys.stderr)
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                break
            except requests.RequestException as exc:
                if attempt == CR_MAX_RETRIES - 1:
                    raise RuntimeError(f"Crossref 请求失败（已重试 {CR_MAX_RETRIES} 次）: {exc}") from exc
                time.sleep(5 * (attempt + 1))
        message = resp.json().get("message", {})
        batch = message.get("items", [])
        if not batch:
            break
        items.extend(batch)
        print(f"[crossref] issn={issn} 已拉取 {len(items)}", flush=True)
        cursor = message.get("next-cursor")
        if not cursor or len(items) >= max_items:
            break
        time.sleep(CR_INTERVAL)
    return items


def build_paper(item: dict[str, Any], journal_name: str) -> dict[str, Any]:
    """Crossref 条目 → 统一 schema（字段名与其它 crawl 脚本一致）。"""
    titles = item.get("title") or []
    title = re.sub(r"\s+", " ", (titles[0] if titles else "")).strip()
    if not title:
        raise ValueError(f"Crossref 条目缺少标题：{item.get('DOI')}")
    date_parts = (item.get("issued", {}).get("date-parts") or [[0]])[0]
    year = int(date_parts[0] or 0)
    authors = []
    for a in item.get("author") or []:
        name = (a.get("given", "") + " " + a.get("family", "")).strip()
        if name:
            authors.append(name)
    abstract = re.sub(r"\s+", " ", strip_jats(item.get("abstract", "")))
    return {
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "url": item.get("URL") or f"https://doi.org/{item.get('DOI', '')}",
        "year": year,
        "published": "-".join(str(d) for d in date_parts),
        "venue": journal_name,
        "source": "Crossref API",
        "category": "",
        "citation_count": 0,
    }


def write_output(path: Path, papers: list[dict[str, Any]]) -> None:
    if not papers:
        print(f"[skip] {path} 无论文，不落盘")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "Crossref API",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(papers),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {path} ({len(papers)} 篇)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取 Nature/NMI 18 个月窗口 AI 论文")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--max-items", type=int, default=5000, help="每刊拉取上限")
    parser.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.window_days)
    root = Path(__file__).resolve().parents[1]
    all_agent_papers: list[dict[str, Any]] = []
    all_co_papers: list[dict[str, Any]] = []

    for journal in JOURNALS:
        raw = crossref_fetch_journal(journal["issn"], start, now, args.max_items)
        papers = [build_paper(item, journal["name"]) for item in raw]

        ai_papers, agent_papers, co_papers = [], [], []
        for p in papers:
            text = f"{p['title']} {p['abstract']}"
            if AGENT_RE.search(text):
                p["is_agent"] = True
                agent_papers.append(p)
                continue
            if CO_RE.search(text):
                p["is_co"] = True
                co_papers.append(p)
                continue
            if journal["ai_filter"] and not AI_RE.search(text):
                continue  # Nature 正刊：非 AI 论文丢弃
            ai_papers.append(p)
        print(f"[{journal['key']}] 总 {len(papers)}，AI 保留 {len(ai_papers)}，"
              f"agent {len(agent_papers)}，组合优化 {len(co_papers)}")
        all_agent_papers.extend(agent_papers)
        all_co_papers.extend(co_papers)

        if args.dry_run:
            for p in ai_papers[:5]:
                print(f"  ai: {p['title'][:70]}")
            continue

        write_output(root / "Nature" / f"{journal['key']}_window.json", ai_papers)
    if not args.dry_run:
        write_output(root / "agent_and_harness" / "nature_agent.json", all_agent_papers)
        write_output(root / "combinatorial_optimization" / "nature_co.json", all_co_papers)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"抓取失败：{exc}", file=sys.stderr)
        raise

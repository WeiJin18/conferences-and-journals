#weijinzi
# 用途：抓取近 18 个月 arXiv 上 agent/harness 与组合优化相关论文，按重要性预筛落盘。
# 前因：原每日抓取只覆盖当天新论文，引用等质量信号全为 0，永远筛不出"重要论文"。
# 原因：用户改为每 45 天全量刷新 18 个月窗口，窗口内论文才有真实引用与 venue 信号。
# 内容：arXiv API 按月分片拉取窗口内三类目论文（保证 18 个月全覆盖，月内截长尾），
#       本地词表过滤分主题，Semantic Scholar 批量补引用数与收录 venue，
#       按"顶会收录优先 + 引用降序"全局排序。
# 行为：agent/harness 论文落 agent_and_harness/，组合优化论文落 combinatorial_optimization/，
#       各自取 top N 覆盖式写入，控制仓库体积。
# 存储：agent_and_harness/arxiv_agent_window.json 与 combinatorial_optimization/arxiv_co_window.json。
# 使用：python crawl_arxiv_window.py [--window-days 545] [--top-agent 150] [--top-co 50] [--dry-run]。
# 依赖：requests；S2 查询失败重试一次，仍失败整批引用记 0 并打印警告，不中断主流程。
# ---

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# ============ 配置区：窗口、抓取、筛选（修改只需改这里） ============

WINDOW_DAYS = 545          # 约 18 个月
CATEGORIES = ["cs.AI", "cs.CL", "cs.LG"]
ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_PAGE_SIZE = 100
ARXIV_INTERVAL = 3.0       # arXiv 官方建议的请求间隔（秒）
ARXIV_MAX_RETRIES = 5      # 429 限速重试次数
MONTHLY_LIMIT = 1000       # 每个月分片最多拉取条数（窗口全覆盖，月内截长尾）
MAX_FETCH = 20000          # 单次运行拉取条数总上限（保护，超限直接截断并警告）

TOP_AGENT = 150            # agent/harness 主题落盘数
TOP_CO = 50                # 组合优化主题落盘数

S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "title,citationCount,venue,publicationVenue"
S2_BATCH_SIZE = 100
S2_INTERVAL = 4.0          # 无 key 时 S2 共享池约 100 请求/5 分钟，保守间隔

# 与 code/paper_quality_filter.py 的 TOP_VENUES 保持一致
TOP_VENUES = [
    "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl",
    "cvpr", "iccv", "eccv", "aaai", "ijcai", "kdd", "www", "sigir",
    "nature", "science", "tmlr",
]

# arXiv 查询词（宽匹配，本地再用 AGENT_RE/CO_RE 精筛）
QUERY_TERMS = [
    "all:agent", "all:harness", 'all:"tool use"', 'all:"function calling"',
    'all:"multi-agent"', 'all:"swe-bench"', "all:webarena", 'all:"code agent"',
    'all:"combinatorial optimization"', 'all:"mixed-integer"',
    'all:"vehicle routing"', 'all:"traveling salesman"', 'all:"travelling salesman"',
    'all:"constraint programming"', 'all:"SAT solving"', 'all:"branch and bound"',
]

# 主题词表（对 title + abstract 匹配）
AGENT_RE = re.compile(
    r"\b(agent|harness|tool[- ]?use|function[- ]?call|multi[- ]?agent|planner|"
    r"reflection|swe[- ]?bench|webarena|orchestrat|scaffold|code[- ]?agent|"
    r"computer[- ]?use|large language model|llm|foundation model)\b",
    re.IGNORECASE,
)
CO_RE = re.compile(
    r"\b(combinatorial optimization|mixed[- ]integer|branch[- ]and[- ]bound|"
    r"vehicle routing|constraint satisfaction|constraint programming|"
    r"sat solving|travelling salesman|traveling salesman|graph coloring|"
    r"max[- ]cut|bin packing|job shop)\b",
    re.IGNORECASE,
)

# ================================================================


def fetch_arxiv_slice(start: datetime, end: datetime, slice_limit: int) -> list[dict[str, Any]]:
    """拉取 [start, end] 时间片内的 arXiv 论文（按提交日期降序，片内截 slice_limit）。"""
    date_range = (
        f"submittedDate:[{start.strftime('%Y%m%d%H%M')} TO {end.strftime('%Y%m%d%H%M')}]"
    )
    cat_clause = "(" + " OR ".join(f"cat:{c}" for c in CATEGORIES) + ")"
    term_clause = "(" + " OR ".join(QUERY_TERMS) + ")"
    search_query = f"{cat_clause} AND {term_clause} AND {date_range}"

    ns = {"a": "http://www.w3.org/2005/Atom"}
    papers: list[dict[str, Any]] = []
    start_idx = 0
    while start_idx < slice_limit:
        params = {
            "search_query": search_query,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
            "start": start_idx,
            "max_results": ARXIV_PAGE_SIZE,
        }
        resp = None
        for attempt in range(ARXIV_MAX_RETRIES):
            resp = requests.get(ARXIV_API, params=params, timeout=60)
            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", 10 * (attempt + 1)))
                print(f"[warn] arXiv 429，等待 {wait}s 重试（第 {attempt + 1} 次）", file=sys.stderr)
                time.sleep(wait)
                continue
            break
        if resp is None or resp.status_code == 429:
            raise RuntimeError(f"arXiv 持续限速（重试 {ARXIV_MAX_RETRIES} 次仍 429）")
        resp.raise_for_status()
        root = ET.fromstring(resp.text)
        entries = root.findall("a:entry", ns)
        if not entries:
            break
        for entry in entries:
            title = "".join(entry.find("a:title", ns).itertext()).strip()
            abstract = "".join(entry.find("a:summary", ns).itertext()).strip()
            url = entry.find("a:id", ns).text.strip()
            published = entry.find("a:published", ns).text.strip()
            authors = [
                "".join(a.find("a:name", ns).itertext()).strip()
                for a in entry.findall("a:author", ns)
            ]
            papers.append({
                "title": title,
                "authors": authors,
                "abstract": abstract,
                "url": url,
                "published": published,
                "category": "",
            })
        start_idx += len(entries)
        if len(entries) < ARXIV_PAGE_SIZE:
            break
        time.sleep(ARXIV_INTERVAL)
    return papers[:slice_limit]


def month_slices(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    """把 [start, end] 按自然月切片（首尾为不完整月），返回 (片起, 片止) 列表。"""
    slices: list[tuple[datetime, datetime]] = []
    cursor = start
    while cursor < end:
        if cursor.month == 12:
            next_month = datetime(cursor.year + 1, 1, 1, tzinfo=cursor.tzinfo)
        else:
            next_month = datetime(cursor.year, cursor.month + 1, 1, tzinfo=cursor.tzinfo)
        slices.append((cursor, min(next_month, end)))
        cursor = next_month
    return slices


def fetch_arxiv_window(start: datetime, end: datetime, monthly_limit: int = MONTHLY_LIMIT) -> list[dict[str, Any]]:
    """按月分片拉取整个窗口，保证 18 个月全覆盖（单次查询会因量级过大被截断）。"""
    papers: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    for slice_start, slice_end in month_slices(start, end):
        if len(papers) >= MAX_FETCH:
            print(f"[warn] 达到 MAX_FETCH={MAX_FETCH}，剩余分片跳过", file=sys.stderr)
            break
        batch = fetch_arxiv_slice(slice_start, slice_end, monthly_limit)
        fresh = [p for p in batch if p["url"] not in seen_urls]
        seen_urls.update(p["url"] for p in fresh)
        papers.extend(fresh)
        print(f"[arxiv] {slice_start.date()}~{slice_end.date()} 片内 {len(batch)}，累计 {len(papers)}", flush=True)
        time.sleep(ARXIV_INTERVAL)
    return papers


def arxiv_id_from_url(url: str) -> str:
    """从 arXiv 链接提取不含版本号的 id，如 2501.00001。"""
    match = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,6})(v\d+)?$", url)
    if not match:
        raise ValueError(f"无法从链接提取 arXiv id：{url}")
    return match.group(1)


def fetch_citations_s2(papers: list[dict[str, Any]]) -> None:
    """批量查询 Semantic Scholar，把 citation_count / s2_venue 写回每篇论文。"""
    ids: list[str] = []
    for p in papers:
        try:
            ids.append(arxiv_id_from_url(p["url"]))
        except ValueError:
            ids.append("")
    todo = [(i, pid) for i, pid in enumerate(ids) if pid]

    session = requests.Session()
    for batch_start in range(0, len(todo), S2_BATCH_SIZE):
        batch = todo[batch_start:batch_start + S2_BATCH_SIZE]
        payload = {"ids": [f"ARXIV:{pid}" for _, pid in batch]}
        info_map: dict[int, dict[str, Any]] = {}
        for attempt in range(2):  # 失败重试一次
            try:
                resp = session.post(
                    S2_BATCH_URL, params={"fields": S2_FIELDS},
                    json=payload, timeout=30,
                )
                resp.raise_for_status()
                for (idx, _pid), info in zip(batch, resp.json()):
                    info_map[idx] = info or {}
                break
            except (requests.RequestException, ValueError) as exc:
                if attempt == 0:
                    time.sleep(S2_INTERVAL)
                    continue
                print(f"[warn] S2 批量查询失败（这一批引用记 0）: {exc}", file=sys.stderr)
        for idx, info in info_map.items():
            papers[idx]["citation_count"] = info.get("citationCount", 0) or 0
            papers[idx]["s2_venue"] = (info.get("venue") or "").strip()
        for idx, _pid in batch:
            papers[idx].setdefault("citation_count", 0)
            papers[idx].setdefault("s2_venue", "")
        done = min(batch_start + S2_BATCH_SIZE, len(todo))
        print(f"[s2] 引用查询进度 {done}/{len(todo)}", flush=True)
        time.sleep(S2_INTERVAL)


def is_top_venue(s2_venue: str) -> bool:
    return any(v in s2_venue.lower() for v in TOP_VENUES)


def sort_key(paper: dict[str, Any]) -> tuple[int, int]:
    """顶会收录优先（+50 引用的等效加成），其次引用数降序。"""
    citations = paper.get("citation_count", 0)
    bonus = 50 if is_top_venue(paper.get("s2_venue", "")) else 0
    return (0 if is_top_venue(paper.get("s2_venue", "")) else 1, -(citations + bonus))


def write_output(path: Path, papers: list[dict[str, Any]], start: datetime, end: datetime) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": "arXiv API (18-month window)",
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "window": {"from": start.date().isoformat(), "to": end.date().isoformat()},
        "count": len(papers),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {path} ({len(papers)} 篇)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取 18 个月窗口的 arXiv agent/组合优化论文")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--monthly-limit", type=int, default=MONTHLY_LIMIT, help="每个月分片拉取上限")
    parser.add_argument("--top-agent", type=int, default=TOP_AGENT)
    parser.add_argument("--top-co", type=int, default=TOP_CO)
    parser.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc)
    start = now - timedelta(days=args.window_days)

    papers = fetch_arxiv_window(start, now, args.monthly_limit)
    if not papers:
        raise RuntimeError("窗口内未抓到任何论文，请检查 arXiv API 或查询词")

    agent_pool, co_pool = [], []
    for p in papers:
        text = f"{p['title']} {p['abstract']}"
        is_agent = bool(AGENT_RE.search(text))
        is_co = bool(CO_RE.search(text))
        p["is_agent"], p["is_co"] = is_agent, is_co
        if is_agent:
            agent_pool.append(p)
        elif is_co:  # 同属两主题的归 agent 池（agent 配额更大）
            co_pool.append(p)
    print(f"[filter] 总计 {len(papers)}，agent {len(agent_pool)}，组合优化 {len(co_pool)}")

    fetch_citations_s2(papers)

    agent_top = sorted(agent_pool, key=sort_key)[: args.top_agent]
    co_top = sorted(co_pool, key=sort_key)[: args.top_co]
    for p in agent_top + co_top:
        p["year"] = int(p["published"][:4])

    if args.dry_run:
        print(f"[dry-run] agent top{args.top_agent} 与 CO top{args.top_co} 不落盘")
        for p in agent_top[:5]:
            print(f"  agent: [{p['citation_count']:4d} 引] {p['title'][:70]}")
        for p in co_top[:5]:
            print(f"  CO   : [{p['citation_count']:4d} 引] {p['title'][:70]}")
        return

    root = Path(__file__).resolve().parents[1]
    write_output(root / "agent_and_harness" / "arxiv_agent_window.json", agent_top, start, now)
    write_output(root / "combinatorial_optimization" / "arxiv_co_window.json", co_top, start, now)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"抓取失败：{exc}", file=sys.stderr)
        raise

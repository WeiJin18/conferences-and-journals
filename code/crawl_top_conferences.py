#weijinzi
# 用途：从 OpenReview 抓取近 18 个月 NeurIPS/ICML/ICLR 三大顶会论文并取样落盘。
# 前因：用户要求除 arXiv 新论文外，还要覆盖 AI 三大顶会的"近 18 个月重要论文"。
# 原因：顶会论文"收录即重要"（用户决策），需要按领域各取代表性若干篇，
#       代表性以 Semantic Scholar 引用数为代理信号。
# 内容：枚举窗口内的会议届次，OpenReview API v2 分页拉全量接收论文，
#       本地按 agent/harness、组合优化、六大领域三路分类，
#       候选经 S2 标题搜索补引用数后排序取样。
# 行为：agent 论文落 agent_and_harness/，组合优化落 combinatorial_optimization/，
#       其余按会议落 NeurIPS|ICML|ICLR/<会议>_<年份>_papers.json，覆盖式写入。
# 使用：python crawl_top_conferences.py [--window-days 545] [--max-notes N] [--dry-run]
#       [--fast]（--fast 跳过 S2 引用查询，按抓取序取样，仅供快速验证）。
# 依赖：requests；S2 标题匹配带相似度校验，匹配失败引用记 0，不中断。
# ---

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import requests

# ============ 配置区：会议、领域、配额（修改只需改这里） ============

# 会议表：venueid 模板 + 会议月份（用于判断届次是否落入窗口，日期近似取 15 日）
CONFERENCES = [
    {"venue": "NeurIPS", "venueid_tpl": "NeurIPS.cc/{year}/Conference", "month": 12},
    {"venue": "ICML", "venueid_tpl": "ICML.cc/{year}/Conference", "month": 7},
    {"venue": "ICLR", "venueid_tpl": "ICLR.cc/{year}/Conference", "month": 4},
]
WINDOW_DAYS = 545          # 约 18 个月
WINDOW_TOLERANCE_DAYS = 30 # 窗口起点容差，避免边界届次被误排除

OPENREVIEW_API = "https://api2.openreview.net/notes"
OR_PAGE_SIZE = 1000
OR_INTERVAL = 1.0
S2_SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
S2_PAGE_SIZE = 100
S2_INTERVAL = 1.2
S2_MAX_RETRIES = 5

VENUE_ALIASES = {
    "NeurIPS": ("neurips", "advances in neural information processing systems"),
    "ICML": ("icml", "international conference on machine learning"),
    "ICLR": ("iclr", "international conference on learning representations"),
}

# 领域关键词表（与 code/crawl_neurips.py 的 FIELDS 保持一致）
FIELDS = {
    "大语言模型与生成": ["language model", "llm", "generative", "instruction", "transformer"],
    "计算机视觉": ["vision", "image", "visual", "video", "object detection", "segmentation"],
    "强化学习": ["reinforcement learning", "policy", "reward", "multi-agent"],
    "优化与学习理论": ["optimization", "theory", "generalization", "convergence", "learning theory"],
    "图学习": ["graph neural", "graph learning", "molecular", "network"],
    "可信与安全": ["fairness", "privacy", "robust", "alignment", "interpretability"],
}

# 取样配额（每个会议）
PER_FIELD = 3              # 每领域篇数
TOP_AGENT = 30             # agent/harness 篇数（说明.md 放宽配额）
TOP_CO = 30                # 组合优化篇数
# 候选截断：进 S2 引用查询前的上限，控制请求数
CANDIDATE_AGENT = 60
CANDIDATE_CO = 40
CANDIDATE_PER_FIELD = 20

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

S2_FIELDS = "title,authors,abstract,url,year,citationCount,venue,publicationVenue"
TITLE_MATCH_THRESHOLD = 0.85  # S2 返回标题与本地的相似度下限，防撞车
S2_QUERY_TERMS: tuple[str, ...] = ()

# ================================================================


def normalize_title(title: str) -> str:
    text = unicodedata.normalize("NFKC", title or "").lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def conferences_in_window(now: datetime) -> list[dict[str, Any]]:
    """枚举窗口内的会议届次（会议日期近似取该月 15 日）。"""
    start = now - timedelta(days=WINDOW_DAYS + WINDOW_TOLERANCE_DAYS)
    result = []
    for conf in CONFERENCES:
        # Include every calendar year touched by the time window; the date check below filters boundary years.
        for year in range(start.year, now.year + 1):
            conf_date = datetime(year, conf["month"], 15, tzinfo=timezone.utc)
            if start <= conf_date <= now:
                entry = dict(conf)
                entry["year"] = year
                result.append(entry)
    if not result:
        raise RuntimeError(f"窗口 [{start.date()} ~ {now.date()}] 内没有会议届次")
    return result


def _s2_authors(item: dict[str, Any]) -> list[str]:
    return [str(author.get("name", "")).strip() for author in item.get("authors") or [] if author.get("name")]


def _venue_matches(venue: str, item: dict[str, Any]) -> bool:
    item_venue = str(item.get("venue") or "")
    publication_venue = item.get("publicationVenue") or {}
    venue_text = " ".join((item_venue, str(publication_venue.get("name") or ""))).lower()
    return any(alias in venue_text for alias in VENUE_ALIASES[venue])


def _s2_get(session: requests.Session, params: dict[str, Any]) -> dict[str, Any]:
    for attempt in range(S2_MAX_RETRIES):
        try:
            resp = session.get(S2_SEARCH_URL, params=params, timeout=60)
            if resp.status_code == 429:
                retry_after = float(resp.headers.get("Retry-After", 2 ** attempt))
                wait = min(retry_after, 60.0)
                print(f"[warn] S2 429，等待 {wait:.1f}s 后重试", file=sys.stderr)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except (requests.RequestException, ValueError) as exc:
            if attempt == S2_MAX_RETRIES - 1:
                raise RuntimeError(f"Semantic Scholar 请求失败: {exc}") from exc
            wait = min(2 ** attempt, 30)
            print(f"[warn] S2 请求失败，{wait}s 后重试: {exc}", file=sys.stderr)
            time.sleep(wait)
    raise RuntimeError("Semantic Scholar 请求重试耗尽")


def semantic_scholar_fetch(venue: str, year: int, max_notes: int) -> list[dict[str, Any]]:
    """按 venue/year 分页查询 S2，并用会议别名严格校验来源。"""
    session = requests.Session()
    results: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for query in (f"{venue} {year}", *S2_QUERY_TERMS):
        offset = 0
        while len(results) < max_notes:
            payload = _s2_get(session, {
                "query": query,
                "limit": S2_PAGE_SIZE,
                "offset": offset,
                "year": str(year),
                "venue": venue,
                "fields": S2_FIELDS,
            })
            batch = payload.get("data") or []
            for item in batch:
                if item.get("year") != year or not _venue_matches(venue, item):
                    continue
                title = str(item.get("title") or "").strip()
                normalized = normalize_title(title)
                if not title or normalized in seen_titles:
                    continue
                seen_titles.add(normalized)
                results.append({
                    "title": title,
                    "authors": _s2_authors(item),
                    "abstract": str(item.get("abstract") or "").strip(),
                    "url": str(item.get("url") or ""),
                    "year": year,
                    "citation_count": int(item.get("citationCount", 0) or 0),
                    "venue": venue,
                    "source": "Semantic Scholar API (venue/year search; may be incomplete)",
                    "category": "",
                })
                if len(results) >= max_notes:
                    return results
            if len(batch) < S2_PAGE_SIZE:
                break
            offset += len(batch)
            time.sleep(S2_INTERVAL)
    return results


def openreview_fetch_all(venueid: str, max_notes: int) -> list[dict[str, Any]]:
    """分页拉取某届会议全部接收论文，返回统一 schema 的条目。"""
    session = requests.Session()
    notes: list[dict[str, Any]] = []
    offset = 0
    while True:
        resp = session.get(
            OPENREVIEW_API,
            params={"content.venueid": venueid, "limit": OR_PAGE_SIZE, "offset": offset},
            timeout=60,
        )
        if resp.status_code == 403 and "ChallengeRequiredError" in resp.text:
            raise RuntimeError(
                f"OpenReview 被 Cloudflare 挑战拦截（本机 IP 受限）。"
                f"GitHub Actions runner 上通常正常；本地可换网络或手动触发 workflow 验证：{venueid}"
            )
        resp.raise_for_status()
        batch = resp.json().get("notes", [])
        if not batch:
            break
        for note in batch:
            content = note.get("content", {})
            title = (content.get("title", {}) or {}).get("value", "") if isinstance(content.get("title"), dict) else content.get("title", "")
            abstract = (content.get("abstract", {}) or {}).get("value", "") if isinstance(content.get("abstract"), dict) else content.get("abstract", "")
            authors = (content.get("authors", {}) or {}).get("value", []) if isinstance(content.get("authors"), dict) else content.get("authors", [])
            if not title:
                continue
            notes.append({
                "title": str(title).strip(),
                "authors": [str(a) for a in (authors or [])],
                "abstract": str(abstract or "").strip(),
                "url": f"https://openreview.net/forum?id={note.get('id', '')}",
                "year": 0,  # 由调用方按会议年份回填
                "citation_count": 0,
            })
        offset += len(batch)
        print(f"[or] {venueid} 已拉取 {offset}", flush=True)
        if offset >= max_notes:
            print(f"[warn] {venueid} 达到 max_notes={max_notes}，截断", file=sys.stderr)
            break
        if len(batch) < OR_PAGE_SIZE:
            break
        time.sleep(OR_INTERVAL)
    if not notes:
        raise RuntimeError(f"OpenReview 未返回论文，请检查 venueid：{venueid}")
    return notes


def fetch_citation_by_title(title: str) -> int:
    """S2 标题搜索取引用数；返回标题相似度不足时记 0。"""
    try:
        resp = requests.get(
            S2_SEARCH_URL,
            params={"query": title, "limit": 3, "fields": "title,citationCount"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json().get("data") or []
        local = normalize_title(title)
        for item in data:
            if SequenceMatcher(None, local, normalize_title(item.get("title", ""))).ratio() >= TITLE_MATCH_THRESHOLD:
                return int(item.get("citationCount", 0) or 0)
        return 0
    except requests.RequestException as exc:
        print(f"[warn] S2 标题查询失败（引用记 0）: {exc}", file=sys.stderr)
        return 0


def classify(notes: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """把会议论文按 agent / co / 领域三路分类，返回候选池（未排序）。"""
    pools: dict[str, list[dict[str, Any]]] = {"agent": [], "co": [], **{f"field:{k}": [] for k in FIELDS}}
    for note in notes:
        text = f"{note['title']} {note['abstract']}"
        if AGENT_RE.search(text):
            pools["agent"].append(note)
            continue  # 同属多主题的归 agent 池
        if CO_RE.search(text):
            pools["co"].append(note)
            continue
        lowered = text.lower()
        for field, keywords in FIELDS.items():
            if any(k in lowered for k in keywords):
                pools[f"field:{field}"].append(note)
    # 截断候选，控制 S2 请求数
    pools["agent"] = pools["agent"][:CANDIDATE_AGENT]
    pools["co"] = pools["co"][:CANDIDATE_CO]
    for field in FIELDS:
        pools[f"field:{field}"] = pools[f"field:{field}"][:CANDIDATE_PER_FIELD]
    return pools


def take_top(pool: list[dict[str, Any]], n: int) -> list[dict[str, Any]]:
    return sorted(pool, key=lambda p: -p["citation_count"])[:n]


def write_output(path: Path, papers: list[dict[str, Any]], venue: str, year: int) -> None:
    if not papers:
        print(f"[skip] {path} 无论文，不落盘")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "source": sorted({p.get("source", "unknown") for p in papers}),
        "venue": venue,
        "year": year,
        "fetched_at": datetime.now(timezone.utc).isoformat(),
        "count": len(papers),
        "papers": papers,
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[out] {path} ({len(papers)} 篇)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="抓取三大顶会 18 个月窗口论文")
    parser.add_argument("--window-days", type=int, default=WINDOW_DAYS)
    parser.add_argument("--max-notes", type=int, default=6000, help="每届会议拉取上限")
    parser.add_argument("--fast", action="store_true", help="跳过 S2 引用查询（快速验证）")
    parser.add_argument("--dry-run", action="store_true", help="只统计不落盘")
    parser.add_argument("--openreview", action="store_true", help="尝试用 OpenReview 补充；403/失败仅告警，不影响 S2 主流程")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    now = datetime.now(timezone.utc)
    confs = conferences_in_window(now)
    print("窗口内会议：", [(c["venue"], c["year"]) for c in confs])

    root = Path(__file__).resolve().parents[1]
    for conf in confs:
        venue, year = conf["venue"], conf["year"]
        venueid = conf["venueid_tpl"].format(year=year)
        notes = semantic_scholar_fetch(venue, year, args.max_notes)
        if not notes:
            print(f"[warn] S2 未返回 {venue} {year}；主渠道无可用结果，该届跳过", file=sys.stderr)
        if args.openreview:
            try:
                supplement = openreview_fetch_all(venueid, args.max_notes)
                known = {normalize_title(note["title"]) for note in notes}
                notes.extend(note for note in supplement if normalize_title(note["title"]) not in known)
                print(f"[or] {venue} {year} 新增 {len(notes) - len(known)} 篇", flush=True)
            except Exception as exc:
                print(f"[warn] OpenReview 补充失败（不影响 S2 主渠道）{venueid}: {exc}", file=sys.stderr)
        for note in notes:
            note["year"] = year
            note["venue"] = venue
            note.setdefault("source", "OpenReview API")
            note["category"] = ""
        pools = classify(notes)
        print(f"[{venue} {year}] 总 {len(notes)}，agent {len(pools['agent'])}，co {len(pools['co'])}")

        if not args.fast:
            for pool in pools.values():
                for note in pool:
                    if not note.get("citation_count"):
                        note["citation_count"] = fetch_citation_by_title(note["title"])
                        time.sleep(S2_INTERVAL)

        if args.dry_run:
            for note in take_top(pools["agent"], 3):
                print(f"  agent: [{note['citation_count']:4d} 引] {note['title'][:70]}")
            continue

        write_output(root / "agent_and_harness" / f"topconf_agent_{venue.lower()}{year}.json",
                     take_top(pools["agent"], TOP_AGENT), venue, year)
        write_output(root / "combinatorial_optimization" / f"topconf_co_{venue.lower()}{year}.json",
                     take_top(pools["co"], TOP_CO), venue, year)
        field_papers = []
        for field in FIELDS:
            for note in take_top(pools[f"field:{field}"], PER_FIELD):
                note = dict(note)
                note["category"] = field
                field_papers.append(note)
        write_output(root / venue / f"{venue.lower()}_{year}_papers.json", field_papers, venue, year)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"抓取失败：{exc}", file=sys.stderr)
        raise

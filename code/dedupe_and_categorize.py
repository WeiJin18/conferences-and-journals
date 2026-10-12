#weijinzi
# 用途：把会议与期刊目录下抓取到的论文统一去重并按年/会议/主题归档。
# 前因：多个抓取脚本产出的 JSON 之间会重复，且缺少统一索引。
# 原因：手动管理成本高，主题、专属目录和会议需要分开维护。
# 内容：扫描指定目录下的 JSON，按标题、arXiv id、url 去重。
# 行为：输出统一索引 JSON，并按年、会议和 agent_and_harness 主题组织子集。
# 存储：结果写入 _index 目录，并生成按主题分组的 Markdown 简表。
# 使用：python dedupe_and_categorize.py [论文根目录] [输出目录]。
# 依赖：仅使用标准库。
# ---

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


ARXIV_ID_RE = re.compile(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,6})", re.IGNORECASE)
OPENREVIEW_ID_RE = re.compile(r"openreview\.net/forum\?id=([A-Za-z0-9_-]+)", re.IGNORECASE)

AGENT_KEYWORDS = [
    "agent",
    "harness",
    "tool use",
    "function calling",
    "multi-agent",
    "planner",
    "reflection",
    "swe-bench",
    "webarena",
    "scaffold",
    "code agent",
    "computer use",
    "orchestrat",
]


@dataclass
class Paper:
    title: str
    source: str
    year: int
    venue: str
    url: str
    authors: list[str] = field(default_factory=list)
    abstract: str = ""
    category: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def dedup_key(self) -> str:
        arxiv_id = ARXID_FROM_URL(self.url) or ARXID_FROM_URL(self.extra.get("url", ""))
        if arxiv_id:
            return f"arxiv:{arxiv_id.lower()}"
        openreview_id = OPENREVIEW_ID_RE.search(self.url)
        if openreview_id:
            return f"openreview:{openreview_id.group(1).lower()}"
        return f"title:{self.title.lower()}"


def ARXID_FROM_URL(url: str) -> str:
    match = ARXIV_ID_RE.search(url or "")
    return match.group(1) if match else ""


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def load_paper_files(root: Path) -> list[Paper]:
    if not root.exists():
        raise FileNotFoundError(f"找不到论文根目录：{root}")
    papers: list[Paper] = []
    for path in sorted(root.rglob("*.json")):
        if "_index" in path.parts or path.name.startswith("."):
            continue
        try:
            data = json.loads(strip_comment_lines(path.read_text(encoding="utf-8")))
        except json.JSONDecodeError as exc:
            raise ValueError(f"JSON 解析失败：{path} ({exc})") from exc
        papers.extend(extract_papers(data, path))
    return papers


def strip_comment_lines(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def extract_papers(data: Any, path: Path) -> list[Paper]:
    papers: list[Paper] = []
    year_hint = data.get("year") if isinstance(data, dict) else None
    source_hint = data.get("source") if isinstance(data, dict) else None
    venue_hint = path.parent.name

    if isinstance(data, dict) and "fields" in data and isinstance(data["fields"], dict):
        for field_name, items in data["fields"].items():
            for item in items:
                papers.append(build_paper(item, year_hint, source_hint, venue_hint, field_name))
        return papers

    if isinstance(data, dict) and isinstance(data.get("papers"), list):
        for item in data["papers"]:
            papers.append(build_paper(item, year_hint, source_hint, venue_hint, ""))
        return papers

    if isinstance(data, list):
        for item in data:
            papers.append(build_paper(item, year_hint, source_hint, venue_hint, ""))
        return papers

    if isinstance(data, dict):
        papers.append(build_paper(data, year_hint, source_hint, venue_hint, ""))
    return papers


def build_paper(item: Any, year_hint: Any, source_hint: Any, venue_hint: str, field_name: str) -> Paper:
    if not isinstance(item, dict):
        raise ValueError(f"论文条目必须是对象：{item}")
    title = normalize_text(str(item.get("title", "")))
    if not title:
        raise ValueError(f"论文缺少标题：{item}")
    url = normalize_text(str(item.get("url", "")))
    year = parse_year(item.get("year") or item.get("published") or year_hint)
    if not year:
        raise ValueError(f"无法确定论文年份：{title}")
    return Paper(
        title=title,
        source=normalize_text(str(item.get("source") or source_hint or "unknown")),
        year=year,
        venue=normalize_text(str(item.get("venue") or venue_hint)),
        url=url,
        authors=[normalize_text(a) for a in item.get("authors", []) if str(a).strip()],
        abstract=normalize_text(str(item.get("abstract", ""))),
        category=normalize_text(str(item.get("category") or field_name)),
        extra={k: v for k, v in item.items() if k not in {"title", "authors", "abstract", "url", "year", "published", "category", "source", "venue"}},
    )


def parse_year(value: Any) -> int:
    if isinstance(value, int):
        return value
    if not value:
        return 0
    match = re.search(r"(19|20)\d{2}", str(value))
    return int(match.group(0)) if match else 0


def is_agent_paper(paper: Paper) -> bool:
    text = f"{paper.title} {paper.abstract} {paper.category}".lower()
    return any(keyword in text for keyword in AGENT_KEYWORDS)


def deduplicate(papers: list[Paper]) -> list[Paper]:
    seen: dict[str, Paper] = {}
    order: list[str] = []
    for paper in papers:
        key = paper.dedup_key()
        if key not in seen:
            seen[key] = paper
            order.append(key)
            continue
        existing = seen[key]
        if len(paper.abstract) > len(existing.abstract):
            seen[key] = paper
    return [seen[key] for key in order]


def group_by_year(papers: list[Paper]) -> dict[int, list[Paper]]:
    grouped: dict[int, list[Paper]] = defaultdict(list)
    for paper in papers:
        grouped[paper.year].append(paper)
    return dict(sorted(grouped.items(), reverse=True))


def group_by_venue(papers: list[Paper]) -> dict[str, list[Paper]]:
    grouped: dict[str, list[Paper]] = defaultdict(list)
    for paper in papers:
        grouped[paper.venue or "unknown"].append(paper)
    return dict(sorted(grouped.items()))


def write_index(papers: list[Paper], output_dir: Path) -> dict[str, Any]:
    by_year = group_by_year(papers)
    by_venue = group_by_venue(papers)
    by_keyword = {
        "agent_and_harness": [paper for paper in papers if is_agent_paper(paper)],
        "general": [paper for paper in papers if not is_agent_paper(paper)],
    }

    summary = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "total": len(papers),
        "by_year": {str(year): len(items) for year, items in by_year.items()},
        "by_venue": {venue: len(items) for venue, items in by_venue.items()},
        "by_keyword": {key: len(items) for key, items in by_keyword.items()},
    }

    index = {
        "summary": summary,
        "by_year": {str(year): [serialize(paper) for paper in items] for year, items in by_year.items()},
        "by_venue": {venue: [serialize(paper) for paper in items] for venue, items in by_venue.items()},
        "by_keyword": {key: [serialize(paper) for paper in items] for key, items in by_keyword.items()},
        "papers": [serialize(paper) for paper in papers],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "index.json").write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(index, output_dir / "index.md")
    write_keyword_sections(by_keyword, output_dir)
    return summary


def write_keyword_sections(by_keyword: dict[str, list[Paper]], output_dir: Path) -> None:
    for key, items in by_keyword.items():
        target = output_dir / key
        target.mkdir(parents=True, exist_ok=True)
        (target / "papers.json").write_text(
            json.dumps([serialize(paper) for paper in items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def write_markdown(index: dict[str, Any], path: Path) -> None:
    lines = ["# 论文索引", ""]
    summary = index["summary"]
    lines.append(f"- 总数：**{summary['total']}**")
    lines.append(f"- 生成时间：{summary['generated_at']}")
    lines.append("")
    lines.append("## 按年统计")
    for year, count in summary["by_year"].items():
        lines.append(f"- {year}: {count}")
    lines.append("")
    lines.append("## 按会议/来源统计")
    for venue, count in summary["by_venue"].items():
        lines.append(f"- {venue}: {count}")
    lines.append("")
    lines.append("## 按关键词")
    for key, count in summary["by_keyword"].items():
        lines.append(f"- {key}: {count}")
    lines.append("")
    lines.append("## agent / harness 论文（按年）")
    for paper in index["by_keyword"]["agent_and_harness"]:
        lines.append(format_paper_dict(paper))
    path.write_text("\n".join(lines), encoding="utf-8")


def format_paper_dict(paper: dict[str, Any]) -> str:
    url = paper.get("url") or "（无链接）"
    return f"- **{paper.get('title', '')}** — {paper.get('venue', '')} {paper.get('year', '')} — [链接]({url})"


def serialize(paper: Paper) -> dict[str, Any]:
    return {
        "title": paper.title,
        "authors": paper.authors,
        "abstract": paper.abstract,
        "url": paper.url,
        "year": paper.year,
        "venue": paper.venue,
        "source": paper.source,
        "category": paper.category,
        "is_agent": is_agent_paper(paper),
        **paper.extra,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="去重并按年/会议/关键词归档论文")
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output or (args.root / "_index")
    papers = load_paper_files(args.root)
    unique = deduplicate(papers)
    summary = write_index(unique, output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"处理失败：{exc}", file=sys.stderr)
        raise

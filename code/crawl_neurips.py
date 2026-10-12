#weijinzi
# 本脚本用于首次验证 NeurIPS 论文抓取流程。
# 会议与期刊项目需要覆盖较多 AI 方向，而不是只深读单篇论文。
# 因此脚本先从 OpenReview 官方公开接口获取论文元数据。
# 本次仅做探索层实现，重点是可运行、可重放、失败可见。
# 后续若筛选规则稳定，再拆分为更长期复用的模块。
# 内容包括：拉取指定年份论文、按关键词粗分领域、限制每个领域篇数。
# 输出包含论文标题、作者、摘要、链接和来源信息。
# 默认只抓取少量样本，避免一次性下载过多数据。
# 运行参数可通过命令行修改年份、领域数量和每领域篇数。
# 输出目录固定在会议与期刊/NeurIPS，便于统一管理结果。
# ---

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import requests
from bs4 import BeautifulSoup


PROCEEDINGS_URL = "https://papers.nips.cc/paper_files/paper/{year}/vol{volume}-main-conference"

FIELDS = {
    "大语言模型与生成": ["language model", "llm", "generative", "instruction", "transformer"],
    "计算机视觉": ["vision", "image", "visual", "video", "object detection", "segmentation"],
    "强化学习": ["reinforcement learning", "policy", "reward", "multi-agent"],
    "优化与学习理论": ["optimization", "theory", "generalization", "convergence", "learning theory"],
    "图学习": ["graph neural", "graph learning", "molecular", "network"],
    "可信与安全": ["fairness", "privacy", "robust", "安全", "alignment", "interpretability"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从 OpenReview 抓取 NeurIPS 论文元数据")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--per-field", type=int, default=3)
    parser.add_argument("--limit", type=int, default=5000)
    return parser.parse_args()


def fetch_notes(year: int, limit: int) -> list[dict[str, Any]]:
    volume = year - 1987
    response = requests.get(PROCEEDINGS_URL.format(year=year, volume=volume), timeout=60)
    response.raise_for_status()
    soup = BeautifulSoup(response.text, "html.parser")
    links = [(a.get_text(" ", strip=True), a.get("href")) for a in soup.select("a[href]")]
    papers = [
        {"title": title, "url": f"https://papers.nips.cc{href}"}
        for title, href in links
        if href and "Abstract-" in href and title
    ]
    if not papers:
        raise RuntimeError(f"未从 NeurIPS Proceedings 获取到 {year} 年论文，请检查年份或网页结构。")
    return papers[:limit]


def value(content: Any) -> Any:
    if isinstance(content, dict) and "value" in content:
        return content["value"]
    return content


def classify(title: str, abstract: str) -> list[str]:
    text = f"{title} {abstract}".lower()
    return [field for field, keywords in FIELDS.items() if any(keyword in text for keyword in keywords)]


def build_result(notes: list[dict[str, Any]], year: int, per_field: int) -> dict[str, Any]:
    selected: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for note in notes:
        response = requests.get(note["url"], timeout=30)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        detail = soup.select_one("div.paper-detail")
        if detail is None:
            raise ValueError(f"论文页面缺少详情区域：{note['url']}")
        title = detail.select_one("h1.paper-title").get_text(" ", strip=True)
        abstract_label = next((h for h in detail.select("h2.section-label") if h.get_text(strip=True) == "Abstract"), None)
        abstract = abstract_label.find_next_sibling().get_text(" ", strip=True) if abstract_label else ""
        authors = detail.select_one("div.authors").get_text(" ", strip=True) if detail.select_one("div.authors") else ""
        if not title or not abstract:
            raise ValueError(f"论文页面缺少标题或摘要：{note['url']}")
        fields = classify(title, abstract)
        for field in fields:
            if len(selected[field]) >= per_field:
                continue
            selected[field].append({"title": title, "authors": authors, "abstract": abstract, "url": note["url"]})
    return {
        "source": "NeurIPS Proceedings", 
        "year": year,
        "requested_per_field": per_field,
        "fields": dict(selected),
    }


def main() -> None:
    args = parse_args()
    output_dir = Path(__file__).resolve().parents[1] / "NeurIPS"
    output_dir.mkdir(parents=True, exist_ok=True)
    notes = fetch_notes(args.year, args.limit)
    result = build_result(notes, args.year, args.per_field)
    output_path = output_dir / f"neurips_{args.year}_representative_papers.json"
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    counts = {field: len(papers) for field, papers in result["fields"].items()}
    print(json.dumps({"output": str(output_path), "source_notes": len(notes), "field_counts": counts}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"抓取失败：{exc}", file=sys.stderr)
        raise

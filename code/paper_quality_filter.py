#weijinzi
# 用途：对 agent_and_harness 目录下的 arXiv 论文做质量评分与分级归档。
# 前因：arXiv 未经同行评审、灌水多，仅关键词过滤无法保证质量，用户要求完整版质量筛选。
# 原因：需要可自动运行的评分规则（引用/顶会/代码活跃度/榜单），过滤低质量论文。
# 内容：读入每日抓取的 arXiv JSON，逐篇调 Semantic Scholar 与 GitHub API，
#       爬取 WebArena / SWE-bench 榜单，按规则打分并分为 high_quality / to_review 两档。
# 行为：评分 >= HIGH_QUALITY_THRESHOLD 的归入 high_quality/，其余归入 to_review/，
#       同时生成 quality_report.json 供人工复核。
# 存储：结果写入 agent_and_harness/high_quality/ 与 agent_and_harness/to_review/。
# 使用：python paper_quality_filter.py [--agent-dir DIR] [--dry-run] [--no-fetch]。
# 依赖：requests；榜单爬取失败或 API 失败时自动降级，不中断主流程。
# 注意：API 有速率限制，默认 1 篇/次间隔 1.2 秒；本地缓存避免重复请求。
# ---

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import unicodedata
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import requests

# ============ 配置区：评分规则与阈值（修改只需改这里） ============

# 顶会/顶刊名单：匹配 Semantic Scholar venue 字段（不区分大小写，子串匹配）
TOP_VENUES = [
    "neurips", "nips", "icml", "iclr", "acl", "emnlp", "naacl",
    "cvpr", "iccv", "eccv", "aaai", "ijcai", "kdd", "www", "sigir",
    "nature", "science", "tmlr",
]

# 引用数分档：(下限, 分值)
CITATION_TIERS = [(0, 0), (5, 1), (50, 2), (200, 3)]

# GitHub 活跃度标准
GITHUB_MIN_STARS = 50
GITHUB_ACTIVE_MONTHS = 6

# 榜单页面（静态 HTML 可解析的才有效，JS 渲染页面自动跳过）
LEADERBOARD_URLS = [
    "https://webarena.dev/leaderboard.html",
    "https://www.swebench.com/",
]

# 高质量阈值：总分 >= 该值进入 high_quality
HIGH_QUALITY_THRESHOLD = 3

# 请求间隔（秒），遵守 API 速率限制
REQUEST_INTERVAL = 1.2

# Semantic Scholar 批量查询端点（一次最多 500 篇，比逐篇查询快且省配额）
S2_BATCH_URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
S2_FIELDS = "title,citationCount,venue,publicationVenue,externalIds,openAccessPdf"

# GitHub API 端点
GITHUB_API = "https://api.github.com/repos/{owner}/{repo}"

# 从 abstract 中提取 GitHub 仓库链接的正则
GITHUB_LINK_RE = re.compile(r"github\.com/([A-Za-z0-9_.-]+)/([A-Za-z0-9_.-]+)", re.IGNORECASE)

# ================================================================


def normalize_title(title: str) -> str:
    """标题归一化：用于榜单模糊匹配（去标点、空白、统一小写）。"""
    text = unicodedata.normalize("NFKC", title).lower()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


class QualityFilter:
    def __init__(self, agent_dir: Path, dry_run: bool = False, no_fetch: bool = False) -> None:
        self.agent_dir = agent_dir
        self.dry_run = dry_run
        self.no_fetch = no_fetch
        self.session = requests.Session()
        self.session.headers["User-Agent"] = "paper-quality-filter/1.0"
        # S2 结果缓存（进程内）：arxiv_id -> paper info
        self.s2_cache: dict[str, dict[str, Any]] = {}
        # 榜单论文标题集合（归一化）
        self.leaderboard_titles: set[str] = set()

    # ---------- 数据加载 ----------

    def load_arxiv_papers(self) -> list[dict[str, Any]]:
        """读取 agent 目录下所有 arxiv_*.json 的论文条目，附带来源文件信息。"""
        papers: list[dict[str, Any]] = []
        for path in sorted(self.agent_dir.glob("arxiv_*.json")):
            raw = path.read_text(encoding="utf-8")
            # 跳过以 # 开头的注释行（如 #weijinzi 头部），避免 JSON 解析失败
            cleaned = "\n".join(
                line for line in raw.splitlines() if not line.lstrip().startswith("#")
            )
            data = json.loads(cleaned)
            for item in data.get("papers", []):
                item["_source_file"] = path.name
                papers.append(item)
        return papers

    # ---------- 信号 1：Semantic Scholar 引用数与 venue ----------

    def fetch_s2_batch(self, arxiv_ids: list[str]) -> dict[str, dict[str, Any]]:
        """批量查询 Semantic Scholar，返回 arxiv_id -> {citationCount, venue, ...}。"""
        result: dict[str, dict[str, Any]] = {}
        todo = [pid for pid in arxiv_ids if pid not in self.s2_cache]
        if not todo:
            return {pid: self.s2_cache[pid] for pid in arxiv_ids}

        for i in range(0, len(todo), 100):  # 批量接口单次上限 500，保守取 100
            batch = todo[i : i + 100]
            try:
                resp = self.session.post(
                    S2_BATCH_URL,
                    params={"fields": S2_FIELDS},
                    json={"ids": [f"arXiv:{pid}" for pid in batch]},
                    timeout=30,
                )
                resp.raise_for_status()
                for pid, info in zip(batch, resp.json()):
                    if info is not None:
                        self.s2_cache[pid] = {
                            "citationCount": info.get("citationCount", 0),
                            "venue": info.get("venue") or "",
                            "publicationVenue": (info.get("publicationVenue") or {}).get("name", ""),
                        }
                    else:
                        self.s2_cache[pid] = {"citationCount": 0, "venue": "", "publicationVenue": ""}
            except requests.RequestException as exc:
                print(f"[warn] S2 批量查询失败（这一批记 0 分）: {exc}", file=sys.stderr)
                for pid in batch:
                    self.s2_cache[pid] = {"citationCount": 0, "venue": "", "publicationVenue": ""}
            time.sleep(REQUEST_INTERVAL)

        return {pid: self.s2_cache[pid] for pid in arxiv_ids}

    # ---------- 信号 2：GitHub 仓库活跃度 ----------

    def extract_github_repos(self, abstract: str) -> list[tuple[str, str]]:
        """从摘要提取 (owner, repo)，排除常见非仓库路径。"""
        repos = []
        for match in GITHUB_LINK_RE.finditer(abstract or ""):
            owner, repo = match.group(1), match.group(2).rstrip(".")
            if repo.lower() in {"io", "pages", "com"}:
                continue
            repos.append((owner, repo))
        return repos

    def check_github_active(self, repos: list[tuple[str, str]]) -> dict[str, Any]:
        """查询仓库 star 数与最近 push 时间，返回活跃度信息。"""
        best: dict[str, Any] = {"found": False, "stars": 0, "pushed_at": "", "repo": ""}
        cutoff = datetime.now(timezone.utc) - timedelta(days=GITHUB_ACTIVE_MONTHS * 30)
        for owner, repo in repos[:3]:  # 最多查 3 个仓库，避免耗时
            try:
                resp = self.session.get(GITHUB_API.format(owner=owner, repo=repo), timeout=15)
                if resp.status_code != 200:
                    continue
                info = resp.json()
                stars = info.get("stargazers_count", 0)
                pushed = info.get("pushed_at", "")
                if stars > best["stars"]:
                    best = {"found": True, "stars": stars, "pushed_at": pushed, "repo": f"{owner}/{repo}"}
                if stars >= GITHUB_MIN_STARS and pushed:
                    pushed_dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
                    if pushed_dt >= cutoff:
                        return {"found": True, "stars": stars, "pushed_at": pushed, "repo": f"{owner}/{repo}"}
            except requests.RequestException:
                continue
            time.sleep(0.5)
        return best

    # ---------- 信号 3：benchmark 榜单匹配 ----------

    def fetch_leaderboards(self) -> None:
        """爬取榜单页面，提取论文标题。失败自动跳过，不中断流程。"""
        for url in LEADERBOARD_URLS:
            try:
                resp = self.session.get(url, timeout=20)
                if resp.status_code != 200:
                    print(f"[warn] 榜单页不可访问，跳过: {url}", file=sys.stderr)
                    continue
                # 从 HTML 中提取 <strong>/<td>/<a> 内疑似论文标题的文本
                text_titles = re.findall(r"<(?:strong|td|a)[^>]*>([^<>]{10,200})</(?:strong|td|a)>", resp.text)
                for t in text_titles:
                    clean = re.sub(r"<[^>]+>", "", t).strip()
                    if len(clean) > 10 and not clean.startswith("http"):
                        self.leaderboard_titles.add(normalize_title(clean))
                print(f"[info] {url} 提取到 {len(self.leaderboard_titles)} 个累计标题")
            except requests.RequestException as exc:
                print(f"[warn] 榜单爬取失败，跳过: {url} ({exc})", file=sys.stderr)

    # ---------- 评分 ----------

    @staticmethod
    def score_venue(s2_info: dict[str, Any]) -> int:
        venues = f"{s2_info.get('venue', '')} {s2_info.get('publicationVenue', '')}".lower()
        return 3 if any(v in venues for v in TOP_VENUES) else 0

    @staticmethod
    def score_citations(citation_count: int) -> int:
        score = 0
        for lower_bound, tier_score in CITATION_TIERS:
            if citation_count >= lower_bound:
                score = tier_score
        return score

    @staticmethod
    def score_github(gh_info: dict[str, Any]) -> int:
        if not gh_info.get("found"):
            return 0
        if gh_info.get("stars", 0) < GITHUB_MIN_STARS:
            return 0
        pushed = gh_info.get("pushed_at", "")
        if not pushed:
            return 0
        cutoff = datetime.now(timezone.utc) - timedelta(days=GITHUB_ACTIVE_MONTHS * 30)
        pushed_dt = datetime.fromisoformat(pushed.replace("Z", "+00:00"))
        return 1 if pushed_dt >= cutoff else 0

    def score_leaderboard(self, title: str) -> int:
        return 2 if normalize_title(title) in self.leaderboard_titles else 0

    def evaluate_paper(self, paper: dict[str, Any]) -> dict[str, Any]:
        """对单篇论文打分，返回带 score 与各信号明细的记录。"""
        arxiv_id = paper.get("arxiv_id") or self.extract_arxiv_id(paper.get("url", ""))

        s2_info = self.s2_cache.get(arxiv_id, {"citationCount": 0, "venue": "", "publicationVenue": ""})
        gh_info = self.check_github_active(self.extract_github_repos(paper.get("abstract", ""))) \
            if not self.no_fetch else {"found": False, "stars": 0, "pushed_at": "", "repo": ""}

        venue_score = self.score_venue(s2_info)
        citation_score = self.score_citations(s2_info.get("citationCount", 0))
        github_score = self.score_github(gh_info)
        leaderboard_score = self.score_leaderboard(paper.get("title", ""))
        total = venue_score + citation_score + github_score + leaderboard_score

        return {
            "title": paper.get("title", ""),
            "arxiv_id": arxiv_id,
            "url": paper.get("url", ""),
            "authors": paper.get("authors", []),
            "abstract": paper.get("abstract", ""),
            "published": paper.get("published", ""),
            "_source_file": paper.get("_source_file", ""),
            "scores": {
                "venue": venue_score,
                "citations": citation_score,
                "github": github_score,
                "leaderboard": leaderboard_score,
                "total": total,
            },
            "signals": {
                "s2_citationCount": s2_info.get("citationCount", 0),
                "s2_venue": s2_info.get("venue", "") or s2_info.get("publicationVenue", ""),
                "github": gh_info,
            },
        }

    @staticmethod
    def extract_arxiv_id(url: str) -> str:
        match = re.search(r"arxiv\.org/abs/([0-9]{4}\.[0-9]{4,6})", url, re.IGNORECASE)
        return match.group(1) if match else ""

    # ---------- 归档 ----------

    def archive(self, records: list[dict[str, Any]]) -> dict[str, Any]:
        """按分数归档到 high_quality / to_review，并生成报告。"""
        hq_dir = self.agent_dir / "high_quality"
        tr_dir = self.agent_dir / "to_review"
        if not self.dry_run:
            hq_dir.mkdir(exist_ok=True)
            tr_dir.mkdir(exist_ok=True)

        hq, tr = [], []
        for rec in records:
            if rec["scores"]["total"] >= HIGH_QUALITY_THRESHOLD:
                hq.append(rec)
            else:
                tr.append(rec)

        report = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "total": len(records),
            "high_quality_count": len(hq),
            "to_review_count": len(tr),
            "threshold": HIGH_QUALITY_THRESHOLD,
            "high_quality": [self.strip_meta(r) for r in hq],
            "to_review": [self.strip_meta(r) for r in tr],
        }

        if not self.dry_run:
            stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
            (hq_dir / f"high_quality_{stamp}.json").write_text(
                json.dumps([self.strip_meta(r) for r in hq], ensure_ascii=False, indent=2), encoding="utf-8")
            (tr_dir / f"to_review_{stamp}.json").write_text(
                json.dumps([self.strip_meta(r) for r in tr], ensure_ascii=False, indent=2), encoding="utf-8")
            (self.agent_dir / "quality_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    @staticmethod
    def strip_meta(rec: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in rec.items() if not k.startswith("_")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="agent/harness 论文质量筛选")
    parser.add_argument("--agent-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "agent_and_harness")
    parser.add_argument("--dry-run", action="store_true", help="只评分不写文件")
    parser.add_argument("--no-fetch", action="store_true", help="跳过 GitHub/榜单网络请求（离线调试用）")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.agent_dir.exists():
        raise FileNotFoundError(f"agent 目录不存在: {args.agent_dir}")

    qf = QualityFilter(args.agent_dir, dry_run=args.dry_run, no_fetch=args.no_fetch)
    papers = qf.load_arxiv_papers()
    if not papers:
        print("没有待评分论文（arxiv_*.json）")
        return

    arxiv_ids = [qf.extract_arxiv_id(p.get("url", "")) for p in papers]
    valid_ids = [pid for pid in arxiv_ids if pid]
    if not args.no_fetch and valid_ids:
        qf.fetch_s2_batch(valid_ids)
        qf.fetch_leaderboards()

    records = []
    for paper in papers:
        rec = qf.evaluate_paper(paper)
        records.append(rec)
        print(f"  [{rec['scores']['total']:2d}] {rec['title'][:70]}")
        if not args.no_fetch:
            time.sleep(REQUEST_INTERVAL)

    report = qf.archive(records)
    print(json.dumps({k: report[k] for k in ("total", "high_quality_count", "to_review_count", "threshold")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"质量筛选失败: {exc}", file=sys.stderr)
        raise

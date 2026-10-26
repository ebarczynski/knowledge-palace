"""Retrieval-quality evaluation harness for Knowledge Palace.

Runs the eval cases in ``evals/cases/*.jsonl`` against a live Knowledge Palace
API, scores the replies (title-level relevance judgments), and reports
Hit@k / MRR@k / Recall@k per domain and mode, negative-probe false positives,
and latency. Designed as a regression gate: ``--gate`` fails on threshold or
baseline regressions so quality is maintained across changes to chunking,
embedding models, search weights, or the min_similarity calibration.

Case schema (one JSON object per line):
    {"id": "...", "domain": "cpp", "query": "...",
     "accept": ["Title Fragment", ...]}            # positive case
    {"id": "...", "domain": "negatives", "query": "...",
     "negative": true, "accept": []}               # negative probe (must return nothing)

A result is relevant when its document title contains any ``accept`` fragment
(case-insensitive). Negative probes must return zero results — they guard the
min_similarity quality gate against weakly-related junk for topics absent from
the library.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import httpx

DEFAULT_API = "http://127.0.0.1:8080"
DEFAULT_K = 5
FETCH_DEPTH = 10  # results fetched per query (>= k, gives recall a denominator)
MODES = ("semantic", "hybrid", "context")

# Regression-gate defaults (--gate). Calibrated 2026-10-26 against the nomic
# baseline; tighten as retrieval improves. Negative probes must stay at zero.
GATE_MIN_OVERALL_HIT = 0.75          # overall hit@k, context mode
GATE_MIN_DOMAIN_HIT = 0.50           # every domain's hit@k, context mode
GATE_MAX_NEG_FP = 0.0                # negative probes returning any result
GATE_REGRESSION = 0.05               # max allowed hit@k drop vs baseline


@dataclass
class Case:
    id: str
    domain: str
    query: str
    accept: list[str]
    negative: bool = False
    note: str = ""


@dataclass
class CaseResult:
    case: Case
    mode: str
    n_results: int
    first_relevant_rank: int | None  # 1-based; None = no relevant hit in fetch depth
    fragments_matched_topk: int
    fragments_matched_any: int
    top_score: float | None          # semantic mode only (cosine sim, for calibration)
    latency_ms: float


@dataclass
class Report:
    api: str
    k: int
    cases: int
    results: list[CaseResult] = field(default_factory=list)
    negative_scores: dict[str, list[float]] = field(default_factory=dict)


def load_cases(cases_dir: Path) -> list[Case]:
    cases: list[Case] = []
    for path in sorted(cases_dir.glob("*.jsonl")):
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise SystemExit(f"{path}:{lineno}: invalid JSON: {e}")
            cases.append(
                Case(
                    id=raw["id"],
                    domain=raw.get("domain", path.stem),
                    query=raw["query"],
                    accept=raw.get("accept", []),
                    negative=bool(raw.get("negative", False)),
                    note=raw.get("note", ""),
                )
            )
    if not cases:
        raise SystemExit(f"no eval cases found in {cases_dir}")
    return cases


def default_api_from_config() -> str:
    for candidate in (Path("config.toml"),):
        if candidate.exists():
            try:
                cfg = tomllib.loads(candidate.read_text())
                api = cfg.get("api", {})
                return f"http://{api.get('host', '127.0.0.1')}:{api.get('port', 8080)}"
            except tomllib.TOMLDecodeError:
                pass
    return DEFAULT_API


def _match_fragments(title: str, accept: list[str]) -> set[str]:
    t = title.casefold()
    return {frag for frag in accept if frag.casefold() in t}


def run_case(client: httpx.Client, api: str, case: Case, mode: str, k: int) -> CaseResult:
    start = time.perf_counter()
    if mode == "context":
        resp = client.get(f"{api}/api/v1/context", params={"q": case.query, "limit": FETCH_DEPTH})
    else:
        resp = client.get(f"{api}/api/v1/search", params={"q": case.query, "mode": mode, "limit": FETCH_DEPTH})
    latency_ms = (time.perf_counter() - start) * 1000
    if resp.status_code != 200:
        raise SystemExit(
            f"API error {resp.status_code} for {case.id} ({mode}): {resp.text[:200]}\n"
            f"Is the server running? Start it with: kp serve"
        )
    results = resp.json()["results"]

    matched_topk: set[str] = set()
    matched_any: set[str] = set()
    first_rank: int | None = None
    for rank, item in enumerate(results, 1):
        frags = _match_fragments(item.get("title", ""), case.accept)
        if not frags:
            continue
        matched_any |= frags
        if rank <= k:
            matched_topk |= frags
            if first_rank is None:
                first_rank = rank

    top_score = results[0]["score"] if results and mode == "semantic" else None
    return CaseResult(
        case=case,
        mode=mode,
        n_results=len(results),
        first_relevant_rank=first_rank,
        fragments_matched_topk=len(matched_topk),
        fragments_matched_any=len(matched_any),
        top_score=top_score,
        latency_ms=latency_ms,
    )


def evaluate(api: str, cases: list[Case], modes: list[str], k: int, concurrency: int) -> Report:
    report = Report(api=api, k=k, cases=len(cases))
    jobs = [(case, mode) for case in cases for mode in modes]
    with httpx.Client(timeout=httpx.Timeout(300.0)) as client:
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_case, client, api, case, mode, k) for case, mode in jobs]
            for fut in futures:
                res = fut.result()
                report.results.append(res)
                if res.case.negative and res.top_score is not None:
                    report.negative_scores.setdefault(res.mode, []).append(res.top_score)
    report.results.sort(key=lambda r: (r.case.domain, r.case.id, MODES.index(r.mode)))
    return report


def _agg(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def compute_metrics(report: Report) -> dict:
    """Nest: metric -> mode -> domain -> value (plus overall)."""
    k = report.k
    domains = sorted({r.case.domain for r in report.results if not r.case.negative})
    negatives = [r for r in report.results if r.case.negative]
    metrics: dict = {"hit@k": {}, "mrr@k": {}, "recall@k": {}, "latency_p50_ms": {}, "negative_fp": {}, "negative_top_score_max": {}}

    for mode in MODES:
        rows = [r for r in report.results if r.mode == mode]
        pos = [r for r in rows if not r.case.negative]
        metrics["hit@k"][mode] = {
            **{d: _agg([1.0 if r.first_relevant_rank and r.first_relevant_rank <= k else 0.0 for r in pos if r.case.domain == d]) for d in domains},
            "overall": _agg([1.0 if r.first_relevant_rank and r.first_relevant_rank <= k else 0.0 for r in pos]),
        }
        metrics["mrr@k"][mode] = {
            **{d: _agg([1.0 / r.first_relevant_rank if r.first_relevant_rank and r.first_relevant_rank <= k else 0.0 for r in pos if r.case.domain == d]) for d in domains},
            "overall": _agg([1.0 / r.first_relevant_rank if r.first_relevant_rank and r.first_relevant_rank <= k else 0.0 for r in pos]),
        }
        metrics["recall@k"][mode] = {
            **{d: _agg([r.fragments_matched_topk / r.fragments_matched_any if r.fragments_matched_any else 0.0 for r in pos if r.case.domain == d]) for d in domains},
            "overall": _agg([r.fragments_matched_topk / r.fragments_matched_any if r.fragments_matched_any else 0.0 for r in pos]),
        }
        lat = sorted(r.latency_ms for r in rows)
        metrics["latency_p50_ms"][mode] = {
            "overall": lat[len(lat) // 2] if lat else 0.0,
        }
        neg = [r for r in negatives if r.mode == mode]
        metrics["negative_fp"][mode] = {
            "overall": _agg([1.0 if r.n_results > 0 else 0.0 for r in neg]),
        }
        scores = report.negative_scores.get(mode, [])
        metrics["negative_top_score_max"][mode] = {"overall": max(scores) if scores else None}

    metrics["domains"] = domains
    metrics["n_cases"] = report.cases
    metrics["k"] = k
    return metrics


def format_report(report: Report, metrics: dict, modes: list[str]) -> str:
    lines: list[str] = []
    k = report.k
    lines.append(f"Knowledge Palace eval — {report.cases} cases vs {report.api} (k={k})")
    lines.append("")
    header = f"{'domain':<20}" + "".join(f"{m + ':hit@' + str(k):>16}{m + ':mrr':>10}" for m in modes)
    lines.append(header)
    lines.append("-" * len(header))
    domains = metrics["domains"]
    for d in domains:
        row = f"{d:<20}"
        for m in modes:
            row += f"{metrics['hit@k'][m][d]:>16.2f}{metrics['mrr@k'][m][d]:>10.2f}"
        lines.append(row)
    row = f"{'OVERALL':<20}"
    for m in modes:
        row += f"{metrics['hit@k'][m]['overall']:>16.2f}{metrics['mrr@k'][m]['overall']:>10.2f}"
    lines.append("-" * len(header))
    lines.append(row)
    lines.append("")
    for m in modes:
        lat = metrics["latency_p50_ms"][m]["overall"]
        fp = metrics["negative_fp"][m]["overall"]
        extra = ""
        ns = metrics["negative_top_score_max"][m].get("overall")
        if ns is not None:
            extra = f"  neg top-score max {ns:.3f}"
        lines.append(f"{m:>9}: p50 latency {lat:7.0f} ms  negative FP {fp:.2f}{extra}")
    return "\n".join(lines)


def format_misses(report: Report, k: int, modes: list[str]) -> str:
    lines = ["", "Missed cases (no acceptable source in top-k):"]
    misses = [r for r in report.results if not r.case.negative and (r.first_relevant_rank is None or r.first_relevant_rank > k) and r.mode in modes]
    if not misses:
        lines.append("  none")
    for r in misses:
        lines.append(f"  [{r.mode:>8}] {r.case.id}: {r.case.query[:70]}")
    return "\n".join(lines)


def compare_baseline(metrics: dict, baseline: dict, k: int, regression: float) -> list[str]:
    """Return list of regression complaints (empty = OK)."""
    problems: list[str] = []
    base_k = baseline.get("k")
    if base_k is not None and base_k != k:
        problems.append(f"baseline used k={base_k}, current run k={k} — not comparable")
        return problems
    base_hit = baseline.get("hit@k", {})
    for mode in MODES:
        cur = metrics["hit@k"].get(mode, {})
        old = base_hit.get(mode, {})
        for scope in (*old.keys(),):
            if scope == "domains" or scope not in cur:
                continue
            delta = cur[scope] - old[scope]
            if delta < -regression:
                problems.append(
                    f"hit@{k}[{mode}][{scope}] regressed {old[scope]:.2f} -> {cur[scope]:.2f} (limit -{regression:.2f})"
                )
    base_fp = baseline.get("negative_fp", {})
    for mode in MODES:
        cur_fp = metrics["negative_fp"].get(mode, {}).get("overall", 0.0)
        old_fp = base_fp.get(mode, {}).get("overall", 0.0)
        if old_fp == 0.0 and cur_fp > 0.0:
            problems.append(f"negative probes now return results in {mode} mode (was clean)")
    return problems


def check_gates(metrics: dict, modes: list[str], k: int, min_overall: float, min_domain: float, max_neg_fp: float) -> list[str]:
    problems: list[str] = []
    if "context" in modes:
        overall = metrics["hit@k"]["context"]["overall"]
        if overall < min_overall:
            problems.append(f"context overall hit@{k} {overall:.2f} < {min_overall:.2f}")
        for d, v in metrics["hit@k"]["context"].items():
            if d == "overall":
                continue
            if v < min_domain:
                problems.append(f"context hit@{k} for domain '{d}' is {v:.2f} < {min_domain:.2f}")
    for mode in ("semantic", "context"):
        if mode in modes:
            fp = metrics["negative_fp"][mode]["overall"]
            if fp > max_neg_fp:
                problems.append(f"negative probes returned results in {mode} mode (FP {fp:.2f})")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Score Knowledge Palace retrieval quality (see evals/README.md)")
    parser.add_argument("--api", default=None, help=f"API base URL (default: from config.toml, else {DEFAULT_API})")
    parser.add_argument("--cases-dir", type=Path, default=None, help="Directory of *.jsonl case files")
    parser.add_argument("--domains", default=None, help="Comma-separated domain filter")
    parser.add_argument("--modes", default=",".join(MODES), help=f"Comma-separated modes ({', '.join(MODES)})")
    parser.add_argument("--k", type=int, default=DEFAULT_K, help="Rank cutoff for hit/mrr/recall")
    parser.add_argument("--concurrency", type=int, default=6, help="Parallel API calls")
    parser.add_argument("--json", type=Path, default=None, help="Write full metrics JSON here")
    parser.add_argument("--save-baseline", type=Path, default=None, help="Save metrics as the new baseline")
    parser.add_argument("--baseline", type=Path, default=None, help="Compare against a saved baseline")
    parser.add_argument("--gate", action="store_true", help="Exit 1 on threshold or baseline regressions")
    parser.add_argument("--show-misses", action="store_true", help="List cases with no acceptable hit in top-k")
    parser.add_argument("--min-overall", type=float, default=GATE_MIN_OVERALL_HIT)
    parser.add_argument("--min-domain", type=float, default=GATE_MIN_DOMAIN_HIT)
    parser.add_argument("--max-neg-fp", type=float, default=GATE_MAX_NEG_FP)
    parser.add_argument("--regression", type=float, default=GATE_REGRESSION)
    args = parser.parse_args(argv)

    api = args.api or default_api_from_config()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in MODES:
            parser.error(f"unknown mode '{m}'; choose from {', '.join(MODES)}")

    cases_dir = args.cases_dir
    if cases_dir is None:
        candidates = [
            Path("evals/cases"),
            Path(__file__).resolve().parent.parent.parent / "evals" / "cases",
        ]
        cases_dir = next((c for c in candidates if c.is_dir()), candidates[0])
    cases = load_cases(cases_dir)
    if args.domains:
        keep = {d.strip() for d in args.domains.split(",")}
        cases = [c for c in cases if c.domain in keep]
        if not cases:
            parser.error(f"no cases for domains: {args.domains}")

    print(f"Evaluating {len(cases)} cases × {len(modes)} modes against {api} ...")
    report = evaluate(api, cases, modes, args.k, args.concurrency)
    metrics = compute_metrics(report)

    print(format_report(report, metrics, modes))
    if args.show_misses:
        print(format_misses(report, args.k, modes))

    if args.json:
        args.json.write_text(json.dumps(metrics, indent=2))
        print(f"\nmetrics JSON -> {args.json}")

    problems: list[str] = []
    if args.baseline:
        baseline = json.loads(Path(args.baseline).read_text())
        problems += compare_baseline(metrics, baseline, args.k, args.regression)
    if args.gate:
        problems += check_gates(metrics, modes, args.k, args.min_overall, args.min_domain, args.max_neg_fp)

    if args.save_baseline:
        args.save_baseline.parent.mkdir(parents=True, exist_ok=True)
        args.save_baseline.write_text(json.dumps(metrics, indent=2))
        print(f"baseline saved -> {args.save_baseline}")

    if problems:
        print("\nGATE FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    if args.gate or args.baseline:
        print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())

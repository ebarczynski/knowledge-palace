"""Eval harness: metrics, regression gate, and the shipped case set."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from knowledge_palace import evalharness as h

EVALS = Path(__file__).resolve().parent.parent / "evals"


def case(case_id: str, domain: str, accept: list[str] | None = None, negative: bool = False) -> h.Case:
    return h.Case(id=case_id, domain=domain, query="q", accept=accept or [], negative=negative)


def outcome(
    c: h.Case, mode: str, *, rank: int | None = None, top: int = 0, any_: int = 0,
    n_results: int | None = None, latency: float = 10.0,
) -> h.CaseResult:
    if n_results is None:
        n_results = 0 if c.negative else 10
    return h.CaseResult(
        case=c, mode=mode, n_results=n_results, first_relevant_rank=rank,
        fragments_matched_topk=top, fragments_matched_any=any_, top_score=None, latency_ms=latency,
    )


CASES = [
    case("cpp-1", "cpp", ["X"]),
    case("cpp-2", "cpp", ["Y"]),
    case("rust-1", "rust", ["Z", "W"]),
    case("neg-1", "negatives", negative=True),
]


def report() -> h.Report:
    results = []
    for mode in h.MODES:
        results += [
            outcome(CASES[0], mode, rank=1, top=1, any_=1),
            outcome(CASES[1], mode, rank=7, top=0, any_=1),  # relevant, but below k=5
            outcome(CASES[2], mode, rank=3, top=1, any_=2),
            # the probe leaks in hybrid (literal hits, allowed) and in context (a gate failure)
            outcome(CASES[3], mode, n_results=0 if mode == "semantic" else 1),
        ]
    return h.Report(api="http://x", k=5, cases=4, results=results)


def test_hit_mrr_and_recall_are_computed_per_domain_and_overall():
    metrics = h.compute_metrics(report())
    for mode in h.MODES:
        assert metrics["hit@k"][mode] == {"cpp": 0.5, "rust": 1.0, "overall": pytest.approx(2 / 3)}
        assert metrics["mrr@k"][mode] == {"cpp": 0.5, "rust": pytest.approx(1 / 3), "overall": pytest.approx(4 / 9)}
        assert metrics["recall@k"][mode] == {"cpp": 0.5, "rust": 0.5, "overall": pytest.approx(0.5)}


def test_negative_probes_and_latency():
    metrics = h.compute_metrics(report())
    assert metrics["negative_fp"]["semantic"] == {"overall": 0.0}
    assert metrics["negative_fp"]["hybrid"] == {"overall": 1.0}
    assert metrics["negative_fp"]["context"] == {"overall": 1.0}
    assert metrics["negative_top_score_max"]["semantic"] == {"overall": None}
    assert metrics["latency_p50_ms"]["context"] == {"overall": 10.0}
    assert metrics["domains"] == ["cpp", "rust"] and metrics["k"] == 5 and metrics["n_cases"] == 4


def test_a_case_with_no_relevant_hit_at_all_counts_as_zero_everywhere():
    only = h.Case(id="x", domain="d", query="q", accept=["T"])
    r = h.Report(api="http://x", k=5, cases=1, results=[outcome(only, "semantic", n_results=10)])
    metrics = h.compute_metrics(r)
    assert metrics["hit@k"]["semantic"] == {"d": 0.0, "overall": 0.0}
    assert metrics["mrr@k"]["semantic"]["overall"] == 0.0
    assert metrics["recall@k"]["semantic"]["overall"] == 0.0


def test_gate_fails_on_low_hit_rates_and_negative_leaks():
    metrics = h.compute_metrics(report())  # context overall 0.67 < 0.75 and a context negative leak
    problems = h.check_gates(metrics, list(h.MODES), 5, 0.75, 0.5, 0.0)
    assert any("context overall hit@5" in p for p in problems)
    assert any("negative probes returned results" in p for p in problems)
    assert not [p for p in problems if "hybrid" in p], "hybrid allows literal keyword hits by design"


def test_gate_passes_a_clean_run():
    clean = []
    for mode in h.MODES:
        clean += [outcome(CASES[0], mode, rank=1), outcome(CASES[2], mode, rank=2), outcome(CASES[3], mode)]
    metrics = h.compute_metrics(h.Report(api="http://x", k=5, cases=3, results=clean))
    assert h.check_gates(metrics, list(h.MODES), 5, 0.75, 0.5, 0.0) == []


def test_baseline_comparison_flags_regressions_and_new_negative_leaks():
    good = h.compute_metrics(report())
    good["negative_fp"]["semantic"] = {"overall": 0.0}
    bad = json.loads(json.dumps(good))
    bad["hit@k"]["semantic"]["cpp"] = 0.0          # dropped 0.5 > 0.05
    bad["hit@k"]["semantic"]["rust"] = 0.96        # within tolerance
    bad["negative_fp"]["semantic"] = {"overall": 0.25}
    problems = h.compare_baseline(bad, good, 5, 0.05)
    assert any("hit@5[semantic][cpp] regressed 0.50 -> 0.00" in p for p in problems)
    assert not any("rust" in p for p in problems)
    assert any("negative probes now return results in semantic mode" in p for p in problems)
    assert h.compare_baseline(good, good, 5, 0.05) == []
    assert any("not comparable" in p for p in h.compare_baseline(good, good, 10, 0.05))


def test_shipped_eval_set_is_well_formed():
    cases = h.load_cases(EVALS / "cases")
    assert len(cases) == 59
    assert len({c.id for c in cases}) == 59, "case ids must be unique"
    positives = [c for c in cases if not c.negative]
    negatives = [c for c in cases if c.negative]
    assert len(positives) == 51 and len(negatives) == 8
    assert all(c.accept for c in positives), "a positive case needs at least one acceptable source"
    assert not any(c.accept for c in negatives)
    assert {c.domain for c in negatives} == {"negatives"}


def test_shipped_baseline_matches_what_the_docs_claim():
    baseline = json.loads((EVALS / "baselines" / "baseline-2026-10-26.json").read_text())
    assert baseline["n_cases"] == 59 and baseline["k"] == 5
    assert round(baseline["hit@k"]["semantic"]["overall"], 2) == 0.86
    assert round(baseline["hit@k"]["hybrid"]["overall"], 2) == 0.86
    assert round(baseline["hit@k"]["context"]["overall"], 2) == 0.84
    assert baseline["negative_fp"]["semantic"]["overall"] == 0.0
    assert baseline["negative_fp"]["context"]["overall"] == 0.0
    # Latency IS recorded: "never instrumented" is not true of the repository.
    assert set(baseline["latency_p50_ms"]) == {"semantic", "hybrid", "context"}


def test_load_cases_rejects_invalid_json(tmp_path: Path):
    (tmp_path / "bad.jsonl").write_text("{not json\n")
    with pytest.raises(SystemExit, match="invalid JSON"):
        h.load_cases(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit, match="no eval cases"):
        h.load_cases(empty)


def test_default_api_comes_from_config_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.chdir(tmp_path)
    assert h.default_api_from_config() == "http://127.0.0.1:8080"
    (tmp_path / "config.toml").write_text('[api]\nhost = "127.0.0.1"\nport = 8099\n')
    assert h.default_api_from_config() == "http://127.0.0.1:8099"

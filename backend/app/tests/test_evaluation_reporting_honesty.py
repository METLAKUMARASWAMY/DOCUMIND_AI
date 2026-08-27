"""A percentage without its denominator is not a measurement.

Directive Phase 21: "Do not report '100%' on a tiny artificial sample as though
it proves universal correctness. Display sample size prominently."

The harness reported bare point estimates — `answer_accuracy_pct 100.0` on five
questions. The number was correctly computed; the presentation invited it to be
read as a claim about the system. Every proportion now carries n, the raw
fraction, and a 95% Wilson interval, and the sample size leads the report.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def harness():
    spec = importlib.util.spec_from_file_location(
        "run_evaluation", PROJECT_ROOT / "scripts/run_evaluation.py"
    )
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves annotations through sys.modules, so the module has to
    # be registered before it executes or its dataclasses fail to build.
    sys.modules["run_evaluation"] = module
    spec.loader.exec_module(module)
    return module


def test_a_perfect_small_sample_does_not_get_a_perfect_interval(harness):
    """5/5 is 100%, and the interval is what stops that reading as certainty."""
    low, high = harness.wilson_interval(5, 5)
    assert high == 100.0
    assert low < 60.0, f"interval too tight for n=5: {low}-{high}"


def test_the_interval_narrows_as_the_sample_grows(harness):
    small_low, _ = harness.wilson_interval(5, 5)
    medium_low, _ = harness.wilson_interval(50, 50)
    large_low, _ = harness.wilson_interval(500, 500)

    assert small_low < medium_low < large_low
    assert large_low > 99.0


def test_a_zero_score_also_gets_a_non_degenerate_interval(harness):
    """The normal approximation gives zero width at 0% and 100%; Wilson does not."""
    low, high = harness.wilson_interval(0, 5)
    assert low == 0.0
    assert high > 40.0, high


def test_an_empty_sample_yields_no_interval(harness):
    assert harness.wilson_interval(0, 0) == (0.0, 0.0)


def test_the_interval_always_lies_within_zero_and_one_hundred(harness):
    for successes, total in [(0, 1), (1, 1), (1, 2), (3, 7), (99, 100), (0, 1000)]:
        low, high = harness.wilson_interval(successes, total)
        assert 0.0 <= low <= high <= 100.0, (successes, total, low, high)


def test_the_interval_brackets_the_point_estimate(harness):
    for successes, total in [(1, 4), (3, 7), (17, 20), (450, 500)]:
        low, high = harness.wilson_interval(successes, total)
        point = 100.0 * successes / total
        assert low <= point <= high, (successes, total, point, low, high)


def test_the_report_leads_with_the_sample_size(harness):
    rendered = harness.render(_results(answerable=5, unanswerable=3))
    body = rendered.splitlines()

    heading_index = next(i for i, line in enumerate(body) if "SAMPLE SIZE" in line)
    metric_index = next(i for i, line in enumerate(body) if "answer_accuracy_pct" in line)
    assert heading_index < metric_index, "the sample size must come before the scores"


def test_a_small_corpus_is_labelled_as_one(harness):
    rendered = harness.render(_results(answerable=5, unanswerable=3))

    assert "small controlled corpus" in rendered
    assert "not a measurement of accuracy in production" in rendered


def test_a_large_corpus_drops_the_small_sample_caution(harness):
    rendered = harness.render(_results(answerable=200, unanswerable=100))
    assert "small controlled corpus" not in rendered


def test_every_percentage_is_printed_with_its_denominator(harness):
    rendered = harness.render(_results(answerable=5, unanswerable=3))

    for line in rendered.splitlines():
        if "_pct" in line and "not measured" not in line:
            assert "n=" in line, f"percentage without a denominator: {line!r}"


def test_a_perfect_score_never_appears_without_an_interval(harness):
    rendered = harness.render(_results(answerable=5, unanswerable=3))

    for line in rendered.splitlines():
        if "100.0%" in line:
            assert "95% CI" in line, f"unqualified perfect score: {line!r}"


def test_an_unmeasured_metric_says_so_rather_than_scoring_zero(harness):
    results = _results(answerable=5, unanswerable=3)
    results["metrics"]["citation_correctness_pct"] = {
        "status": "not_measured",
        "reason": "no questions declared expected_documents",
    }
    rendered = harness.render(results)

    assert "citation_correctness_pct" in rendered
    assert "not measured (no questions declared expected_documents)" in rendered
    assert "citation_correctness_pct         0" not in rendered


def _results(*, answerable: int, unanswerable: int) -> dict:
    return {
        "corpus": {
            "documents": 3,
            "ingested": 3,
            "answerable_questions": answerable,
            "unanswerable_questions": unanswerable,
            "ingest_failures": 0,
        },
        "metrics": {
            "answer_accuracy_pct": {
                "status": "measured",
                "value": 100.0,
                "correct": answerable,
                "total": answerable,
            },
            "abstention_correctness_pct": {
                "status": "measured",
                "value": 100.0,
                "correct": unanswerable,
                "total": unanswerable,
            },
            "overclaims": {"status": "measured", "value": 0},
        },
        "runtime": {"vector_backend": "local_hybrid_only"},
    }

"""Offline reward tests — no GPU, no API key, no network.

The point of running these before spending anything: a reward that is subtly
wrong still produces a beautiful training curve. On Prime this was caught by
checking that the gold answer scores 1.0 against itself; if it doesn't, the
task is literally unlearnable and every dollar after that is wasted.

    ./fw-venv/bin/python -m pytest test_reward.py -q

`test_prime_parity_extraction` is the cross-platform check: the extraction rule
here must behave identically to the verifiers Parser in `orena_focus_rl_v2.py`,
which is reproduced verbatim as a reference. The scorer itself needs no such
test — `orena_scoring.py` is byte-identical across the two projects (sha256
bf9ea4f3…), which is asserted below.
"""

from __future__ import annotations

import hashlib
import os
import re

import pytest

import orena_scoring as S
from evaluation import EXCLUDED_CLASSES, extract, score_row

_HERE = os.path.dirname(os.path.abspath(__file__))
PRIME_ENV = os.path.join(os.path.dirname(os.path.dirname(_HERE)), "environments", "orena-focus-rl-v2")


def g(answer, fmt, choices=None):
    return {"answer": answer, "answer_format": fmt, "choices": choices or []}


# --------------------------------------------------------------------------
# Reward integrity: gold must score 1.0 against itself, or nothing is learnable
# --------------------------------------------------------------------------

GOLDS = [
    ("3", "number"),
    ("0", "number"),
    ("12", "number"),
    ("yes", "binary"),
    ("no", "binary"),
    ("clip", "FO_Class"),
    ("clip, sponge", "FO_Class"),
    ("none", "FO_Class"),
    ("specimen bag", "FO_Class"),
]


@pytest.mark.parametrize("answer,fmt", GOLDS)
def test_gold_scores_one_against_itself(answer, fmt):
    r = score_row(answer, g(answer, fmt))
    assert r.score == 1.0, f"gold {answer!r} ({fmt}) scored {r.score} against itself"
    assert r.metrics["exact_match"].score == 1.0


@pytest.mark.parametrize("answer,fmt", GOLDS)
def test_gold_survives_the_answer_tag_wrapper(answer, fmt):
    """Models emit <answer> tags ~43% of the time; both forms must score alike."""
    r = score_row(extract(f"<answer>{answer}</answer>"), g(answer, fmt))
    assert r.score == 1.0


# --------------------------------------------------------------------------
# Wrong answers must actually score zero
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "pred,answer,fmt",
    [
        ("yes", "no", "binary"),
        ("no", "yes", "binary"),
        ("clip", "none", "FO_Class"),       # hallucinating onto an empty gold
        ("none", "clip", "FO_Class"),       # missing a present object
        ("", "3", "number"),                # empty completion
        ("banana", "yes", "binary"),        # unparseable
    ],
)
def test_wrong_answers_score_zero(pred, answer, fmt):
    assert score_row(pred, g(answer, fmt)).score == 0.0


# --------------------------------------------------------------------------
# Dense shaping: partial credit exists, but exactness strictly dominates
# --------------------------------------------------------------------------


def test_near_miss_count_gets_partial_credit_below_exact():
    exact = score_row("4", g("4", "number")).score
    near = score_row("3", g("4", "number")).score
    far = score_row("40", g("4", "number")).score
    assert exact == 1.0
    assert 0.0 < near < exact, f"near-miss got {near}; dense shaping is the whole point"
    assert far < near, "error magnitude must matter"


def test_partial_class_overlap_scores_between_zero_and_one():
    s = score_row("clip", g("clip, sponge", "FO_Class")).score
    assert 0.0 < s < 1.0


def test_partial_credit_never_leaks_into_exact_match():
    """exact_match is the reported number — dense credit must not inflate it."""
    r = score_row("3", g("4", "number"))
    assert r.score > 0.0
    assert r.metrics["exact_match"].score == 0.0


# --------------------------------------------------------------------------
# Zero-weight monitors
# --------------------------------------------------------------------------


def test_excluded_class_monitor_fires_and_does_not_touch_the_score():
    r = score_row("silicone loop", g("clip", "FO_Class"))
    assert r.metrics["no_excluded_class"].score == 0.0
    r_ok = score_row("clip", g("clip", "FO_Class"))
    assert r_ok.metrics["no_excluded_class"].score == 1.0
    # The monitor is diagnostic only: naming an excluded class must not change
    # the gradient, otherwise it is a hidden second objective.
    assert r_ok.score == 1.0


def test_excluded_set_matches_the_exporter():
    import export_dataset as X

    assert EXCLUDED_CLASSES == X.canonical_classes(X.EXCLUDE_CLASSES), (
        "evaluator and exporter disagree on which classes were filtered — "
        "the monitor would report on a different set than was actually removed"
    )


def test_hallucination_monitor_is_structurally_unable_to_fire():
    """Documents a real limitation rather than pretending the metric works.

    `parse_class_set` only ever emits canonical vocabulary entries, so
    `named - known` is always empty and `no_hallucinated_class` is pinned at
    1.0. Same no-op it was on Prime. Kept for ledger-shape parity; do not read
    it as evidence the model isn't inventing classes.
    """
    for pred in ("scalpel", "trocar, grasper", "purple widget", "clip"):
        assert score_row(pred, g("clip", "FO_Class")).metrics["no_hallucinated_class"].score == 1.0


# --------------------------------------------------------------------------
# Cross-platform parity with the Prime environment
# --------------------------------------------------------------------------


def test_scorer_is_byte_identical_to_the_prime_environment():
    mine = hashlib.sha256(open(os.path.join(_HERE, "orena_scoring.py"), "rb").read()).hexdigest()
    assert mine == "bf9ea4f392fca207985919862316c065e8728f3483a40c587388dfbc383b3fd9"
    prime = os.path.join(PRIME_ENV, "orena_scoring.py")
    if os.path.exists(prime):
        assert mine == hashlib.sha256(open(prime, "rb").read()).hexdigest(), (
            "the Prime env's scorer has drifted from this copy — reward numbers "
            "are no longer comparable across the two platforms"
        )


def _prime_extract(text: str) -> str:
    """Verbatim copy of `_extract` from orena_focus_rl_v2.py. Reference only."""
    t = (text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if m:
        return m.group(1).strip()
    t = re.sub(r"</?answer>", "", t, flags=re.I).strip()
    return t.split("\n")[-1].strip() if t else ""


@pytest.mark.parametrize(
    "raw",
    [
        "3",
        "  3  ",
        "<answer>3</answer>",
        "<answer>\n clip, sponge \n</answer>",
        "Let me think.\nThe answer is yes",
        "<answer>yes",
        "",
        "   ",
        "line one\nline two\nnone",
        "ANSWER: 4",
    ],
)
def test_prime_parity_extraction(raw):
    assert extract(raw) == _prime_extract(raw)


# --------------------------------------------------------------------------
# ground_truth normalization
# --------------------------------------------------------------------------


def test_json_string_ground_truth_is_not_silently_rescored_as_open_ended():
    """A stringified ground_truth must still hit the `number` scorer.

    If it fell through to open_ended, "3" vs "4" would score on token overlap
    instead of numeric distance — wrong, and invisible in the curve.
    """
    import json

    from evaluation import gold_of

    class _Row:
        ground_truth = json.dumps({"answer": "3", "answer_format": "number", "choices": []})

    parsed = gold_of(_Row())
    assert parsed["answer_format"] == "number"
    assert score_row("3", parsed).score == 1.0

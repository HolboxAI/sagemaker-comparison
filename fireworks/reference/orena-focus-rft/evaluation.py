"""ORena-FOCUS foreign-object VQA — Fireworks RFT evaluator.

The Eval Protocol equivalent of the `vf.Rubric` in `orena-focus-rl-v2`. Same
scorer file (`orena_scoring.py`, sha256 bf9ea4f3…), same extraction rules, same
weights — so a reward number here means what it meant on Prime.

    weight  metric                 what it is
    ------  ---------------------  ---------------------------------------------
      1.0   correctness            dense, format-aware; the training signal
      0.0   exact_match            strict 0/1; the number you REPORT
      0.0   no_hallucinated_class  named a class outside the ORena vocabulary?
      0.0   no_excluded_class      still naming silicone loop after filtering?

Only `correctness` moves the gradient. The three zero-weight metrics ride along
in `EvaluateResult.metrics` so they surface in the Fireworks ledger without
contaminating the objective — the same split the Prime rubric used.

Reward logic is tested offline by `test_reward.py`, which needs no GPU and no
API key. Run that before spending anything on a job.
"""

# NB: no `from __future__ import annotations` here. Eval Protocol validates the
# decorated function's signature by comparing `row`'s annotation against the
# EvaluationRow class itself; PEP 563 turns it into the string "EvaluationRow"
# and registration fails with a misleading type error.
import json
import os
import re
from typing import Any

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

import orena_scoring as S

# Mirrors EXCLUDE_CLASSES in export_dataset.py. Canonical set, so the monitor
# keeps working if the export-side list later grows aliases.
EXCLUDED_CLASSES = {"silicone loop"}

_HERE = os.path.dirname(os.path.abspath(__file__))
# Overridable so the same evaluator can be pointed at the eval split without
# editing the decorator.
DATASET = os.environ.get("ORENA_RFT_DATASET") or os.path.join(_HERE, "data", "orena_train.jsonl")

# Qwen3-VL-8B is the only vision-capable training shape Fireworks documents.
# NOTE: no VL model is serverless on this account — a live rollout needs a
# dedicated deployment. Reward correctness is covered offline instead.
MODEL = os.environ.get("ORENA_RFT_MODEL", "fireworks_ai/accounts/fireworks/models/qwen3-vl-8b-instruct")


def extract(text: str) -> str:
    """Last-line / <answer> extraction, identical to the verifiers Parser.

    Qwen3-VL routes chain-of-thought into `reasoning_content`, so the visible
    reply IS the answer. <answer> tags appeared in only ~43% of Prime rollouts,
    so accept them when present and take the raw reply when not — demanding them
    produces a noisy signal for no benefit.
    """
    t = (text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if m:
        return m.group(1).strip()
    t = re.sub(r"</?answer>", "", t, flags=re.I).strip()
    return t.split("\n")[-1].strip() if t else ""


def prediction_of(row: EvaluationRow) -> str:
    """Text of the model's final turn, flattened out of whatever shape it arrived in."""
    assistants = [m for m in row.messages if getattr(m, "role", None) == "assistant"]
    if not assistants:
        return ""
    content: Any = assistants[-1].content
    if isinstance(content, list):
        parts = []
        for c in content:
            parts.append(c.get("text", "") if isinstance(c, dict) else (getattr(c, "text", "") or ""))
        content = " ".join(p for p in parts if p)
    return extract(str(content or ""))


def gold_of(row: EvaluationRow) -> dict:
    """ground_truth as a dict, tolerating the JSON-string form.

    The exporter writes a dict, but a round-trip through another tool can
    stringify it. Silently falling back to open_ended would rescore the whole
    dataset with the wrong scorer, so normalize rather than guess.
    """
    gt = row.ground_truth
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (ValueError, TypeError):
            return {"answer": gt, "answer_format": "open_ended", "choices": []}
    if not isinstance(gt, dict):
        return {"answer": str(gt), "answer_format": "open_ended", "choices": []}
    return gt


def score_row(pred: str, gold: dict) -> EvaluateResult:
    """Pure reward computation — no EvaluationRow, no network, so it is testable.

    Everything platform-specific lives in the decorated wrapper below.
    """
    answer = gold.get("answer", "")
    fmt = gold.get("answer_format", "open_ended")
    choices = gold.get("choices") or None

    # Dense and format-aware: partial credit for near-miss counts and for class
    # sets overlapping the gold. This is what makes the task learnable at all.
    correctness = S.score_answer(pred, answer, fmt, choices=choices, soft_number=True, partial_class=True)

    # Strict. Report this one; never train on it.
    strict = S.score_answer(pred, answer, fmt, choices=choices, soft_number=False, partial_class=False)
    exact_match = 1.0 if strict == 1.0 else 0.0

    named = S.parse_class_set(pred)
    known = set(S.FO_CLASSES) | {"none"}
    no_hallucinated = 0.0 if (named - known) else 1.0
    # Holds at 1.0 while exact_match climbs -> the export filter worked. Decays
    # -> the prior survives in the base weights and filtering alone won't remove it.
    no_excluded = 0.0 if (named & EXCLUDED_CLASSES) else 1.0

    return EvaluateResult(
        score=correctness,
        is_score_valid=True,
        reason=f"[{fmt}] pred={pred!r} gold={answer!r} correctness={correctness:.3f} exact={exact_match:.0f}",
        metrics={
            "exact_match": MetricResult(
                score=exact_match, is_score_valid=True, reason="strict 0/1 — the reported number"
            ),
            "no_hallucinated_class": MetricResult(
                score=no_hallucinated,
                is_score_valid=True,
                reason=f"out-of-vocabulary classes: {sorted(named - known) or 'none'}",
            ),
            "no_excluded_class": MetricResult(
                score=no_excluded,
                is_score_valid=True,
                reason=f"excluded classes named: {sorted(named & EXCLUDED_CLASSES) or 'none'}",
            ),
        },
    )


@evaluation_test(
    input_dataset=[DATASET],
    completion_params=[
        {
            "model": MODEL,
            "temperature": 0.7,
            # 512 was measured, not guessed: thinking-off, the median completion
            # on this task is ~2 tokens and 0/24 probe rollouts truncated, while
            # thinking-on ran dense to a 2048 cap with ~50% truncation.
            "max_tokens": 512,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
    # Local smoke test only — RFT sweeps the whole registered dataset.
    max_dataset_rows=int(os.environ.get("ORENA_RFT_MAX_ROWS", "8")),
    passed_threshold=0.0,
)
def orena_focus_reward(row: EvaluationRow, **kwargs) -> EvaluationRow:
    row.evaluation_result = score_row(prediction_of(row), gold_of(row))
    return row


if __name__ == "__main__":
    import pytest

    raise SystemExit(pytest.main([__file__, "-vs"]))

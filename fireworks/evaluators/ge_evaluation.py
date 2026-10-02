"""GoEmotions multi-label classification — Fireworks GRPO reward evaluator.

Reward = Jaccard similarity between the predicted label set and the ground-truth
label set. micro-F1 rides along as a tracked metric. Both are computed on the
model's final assistant turn, parsed as a comma-separated list of label names.
"""
# NB: no `from __future__ import annotations` — Eval Protocol validates the
# decorated fn signature against EvaluationRow itself.
import json
import re
from typing import Any

from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]
_KNOWN = set(LABELS)

MODEL = "fireworks_ai/accounts/fireworks/models/qwen3-0p6b"


def parse_labels(text: str) -> set:
    """Extract a set of canonical labels from a model completion."""
    t = (text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"</?answer>", "", t, flags=re.I).strip()
    t = t.split("\n")[-1].strip()
    out = set()
    for part in re.split(r"[,;|]", t):
        p = part.strip().lower().strip(".").strip()
        if p in _KNOWN:
            out.add(p)
    return out


def prediction_of(row: EvaluationRow) -> str:
    assistants = [m for m in row.messages if getattr(m, "role", None) == "assistant"]
    if not assistants:
        return ""
    content: Any = assistants[-1].content
    if isinstance(content, list):
        parts = []
        for c in content:
            parts.append(c.get("text", "") if isinstance(c, dict) else (getattr(c, "text", "") or ""))
        content = " ".join(p for p in parts if p)
    return str(content or "")


def gold_of(row: EvaluationRow) -> dict:
    gt = row.ground_truth
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except (ValueError, TypeError):
            return {"labels": []}
    if not isinstance(gt, dict):
        return {"labels": []}
    return gt


def score_row(pred_text: str, gold: dict) -> EvaluateResult:
    pred = parse_labels(pred_text)
    gold_set = set(gold.get("labels") or [])
    inter = pred & gold_set
    union = pred | gold_set
    jaccard = len(inter) / len(union) if union else 1.0
    f1 = 2 * len(inter) / (len(pred) + len(gold_set)) if (pred or gold_set) else 1.0
    return EvaluateResult(
        score=jaccard,
        is_score_valid=True,
        reason=f"pred={sorted(pred)} gold={sorted(gold_set)} jaccard={jaccard:.4f} f1={f1:.4f}",
        metrics={
            "micro_f1": MetricResult(score=f1, is_score_valid=True, reason="micro-F1 over predicted vs gold label set"),
            "jaccard": MetricResult(score=jaccard, is_score_valid=True, reason="Jaccard = |intersection|/|union|"),
        },
    )


@evaluation_test(
    input_dataset=["data/train.jsonl"],
    completion_params=[
        {
            "model": MODEL,
            "temperature": 0.7,
            "max_tokens": 64,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
    ],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
    max_dataset_rows=8,
    passed_threshold=0.0,
)
def goemotions_reward(row: EvaluationRow, **kwargs) -> EvaluationRow:
    row.evaluation_result = score_row(prediction_of(row), gold_of(row))
    return row

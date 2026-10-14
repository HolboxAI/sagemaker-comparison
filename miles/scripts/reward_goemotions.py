"""GoEmotions multi-label reward for Miles RL.

Self-contained port of `fireworks/evaluators/ge_evaluation.py` (same label set,
same parse rules, same Jaccard objective). The entrypoint is the Miles custom
reward contract instead of the Fireworks Eval Protocol decorator.

Miles custom reward contract (`--custom-rm-path <this module>.reward_fn`):

    def reward_fn(args, sample, **kwargs) -> dict

Reads `sample.response` (the completion) and `sample.label` (a JSON list of
ground-truth labels from `--label-key`). Returns a dict; `--reward-key
reward_value` pulls the Jaccard score out.

Reward = Jaccard of predicted vs ground-truth label set. micro-F1 rides along
as a tracked metric, exactly as in the Fireworks reference.
"""
from __future__ import annotations

import json
import re
from typing import Any

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization", "relief",
    "remorse", "sadness", "surprise", "neutral",
]
_KNOWN = set(LABELS)


def parse_labels(text: str) -> set:
    t = (text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if m:
        t = m.group(1).strip()
    t = re.sub(r"</?answer>", "", t, flags=re.I).strip()
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S | re.I).strip()
    t = t.split("\n")[-1].strip()
    out = set()
    for part in re.split(r"[,;|]", t):
        p = part.strip().lower().strip(".").strip()
        if p in _KNOWN:
            out.add(p)
    return out


def gold_of(label: str | None) -> list[str]:
    if label is None:
        return []
    if isinstance(label, str):
        try:
            gt = json.loads(label)
        except (ValueError, TypeError):
            return []
        return gt if isinstance(gt, list) else []
    if isinstance(label, list):
        return label
    return []


def score_row(pred_text: str, gold_labels: list[str]) -> dict:
    pred = parse_labels(pred_text)
    gold_set = set(gold_labels or [])
    inter = pred & gold_set
    union = pred | gold_set
    jaccard = len(inter) / len(union) if union else 1.0
    f1 = 2 * len(inter) / (len(pred) + len(gold_set)) if (pred or gold_set) else 1.0
    return {
        "reward_value": jaccard,
        "reward_cat": "success" if jaccard > 0.0 else "no_overlap",
        "micro_f1": f1,
        "jaccard": jaccard,
    }


def reward_fn(args, sample, **kwargs) -> dict:
    """Miles custom reward contract. `sample` is miles.utils.types.Sample."""
    pred = getattr(sample, "response", "")
    gold = gold_of(getattr(sample, "label", None))
    return score_row(pred, gold)


if __name__ == "__main__":
    from types import SimpleNamespace

    cases = [
        ("joy, gratitude", ["joy", "gratitude"], 1.0),
        ("joy", ["joy", "gratitude"], 0.5),          # Jaccard {joy}/{joy,gratitude}
        ("anger", ["joy"], 0.0),
        ("<answer>sadness</answer>", ["sadness"], 1.0),
    ]
    ok = True
    for response, gold, expected in cases:
        s = SimpleNamespace(response=response, label=json.dumps(gold))
        out = reward_fn(None, s)
        got = out["reward_value"]
        status = "OK" if abs(got - expected) < 1e-6 else "FAIL"
        ok = ok and status == "OK"
        print(f"{status}: response={response!r} gold={gold} -> reward={got:.4f} f1={out['micro_f1']:.4f}")
    raise SystemExit(0 if ok else 1)

"""Verifiable per-format reward for ORena surgical-frame VQA (RLVR).

Scores match the config §8.1 / §8.2 contract. Qwen3.5 generates in "thinking"
mode, so we strip any <think>...</think> block before extracting the answer.
"""
from __future__ import annotations

import re
from collections import Counter

FO_CLASSES = [
    "Clip", "Silicone loop", "Needle", "Sponge",
    "Specimen bag", "Specimen", "External drain", "Gallstone",
]

_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def extract_answer(text: str) -> str:
    if not text:
        return ""
    text = _THINK_RE.sub("", text)          # drop thinking block
    m = _ANSWER_RE.search(text)
    return m.group(1).strip() if m else text.strip()


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip().lower()).strip(" .")


def _first_int(s: str):
    m = re.search(r"-?\d+", s)
    return int(m.group(0)) if m else None


def _parse_classes(s: str) -> set[str]:
    if _normalize(s) in ("none", ""):
        return set()
    out = set()
    for part in re.split(r"[,;]|\band\b", s):
        part = _normalize(part)
        if part:
            out.add(part)
    return out


def _micro_f1(pred: set, gold: set) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    prec = tp / len(pred)
    rec = tp / len(gold)
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def _token_f1(pred: str, gold: str) -> float:
    p = Counter(pred.lower().split())
    g = Counter(gold.lower().split())
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    tp = sum((p & g).values())
    prec = tp / sum(p.values())
    rec = tp / sum(g.values())
    return 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0


def score(generated: str, gold: str, fmt: str) -> float:
    pred = extract_answer(generated)
    gold = (gold or "").strip()

    if fmt == "number":
        pi, gi = _first_int(pred), _first_int(gold)
        return 1.0 if (pi is not None and pi == gi) else 0.0
    if fmt in ("binary", "multiple_choice"):
        return 1.0 if _normalize(pred) == _normalize(gold) else 0.0
    if fmt == "fo_class":
        return _micro_f1(_parse_classes(pred), _parse_classes(gold))
    if fmt == "open_ended":
        return _token_f1(pred, gold)
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _completion_text(c):
    """trl 1.14 passes completions as [{"role": "assistant", "content": "..."}]."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "\n".join(_completion_text(x) for x in c)
    if isinstance(c, dict):
        content = c.get("content", c.get("text", ""))
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text")
        return str(content)
    return str(c)


def reward_func(completions, answer, answer_format, **kwargs):
    """TRL 1.14 reward entry: completions, answer, answer_format are all flat lists, one entry per completion."""
    rewards = []
    for comp, gold, fmt in zip(completions, answer, answer_format):
        rewards.append(score(_completion_text(comp), gold, fmt))
    return rewards

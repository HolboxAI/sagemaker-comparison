"""ORena-FOCUS surgical VQA reward for Miles RL.

Self-contained port of `fireworks/reference/orena-focus-rft/orena_scoring.py`
(same normalizer, same aliases, same weights). One change: the entrypoint is the
Miles custom reward contract instead of the Fireworks Eval Protocol decorator.

Miles custom reward contract (`--custom-rm-path <this module>.reward_fn`):

    def reward_fn(args, sample, **kwargs) -> dict

`sample` is a `miles.utils.types.Sample`. The fields we read:

    sample.response  -> the model's final completion (str)
    sample.label     -> ground truth from `--label-key`, a JSON string here

The reward function returns a dict. `--reward-key reward_value` pulls the float
out of that dict. `--log-reward-category reward_cat` logs the answer format.

Semantics match the Fireworks reference exactly:
    reward_value  = dense format-aware correctness  (the training signal)
    exact_match   = strict 0/1                     (the number we report)
"""
from __future__ import annotations

import json
import re
from typing import Any

# --------------------------------------------------------------------------
# Answer formats
# --------------------------------------------------------------------------

ANSWER_FORMATS = ["open_ended", "number", "binary", "FO_Class", "multiple choice"]

FORMAT_ALIASES = {
    "open_ended": "open_ended", "open-ended": "open_ended", "openended": "open_ended",
    "number": "number", "count": "number",
    "binary": "binary", "yes_no": "binary",
    "fo_class": "FO_Class", "foclass": "FO_Class", "class": "FO_Class",
    "multiple choice": "multiple choice", "multiple_choice": "multiple choice",
    "multiple-choice": "multiple choice", "mc": "multiple choice",
}


def canonical_format(fmt: str) -> str:
    if fmt in ANSWER_FORMATS:
        return fmt
    return FORMAT_ALIASES.get(str(fmt).strip().lower(), "open_ended")


# --------------------------------------------------------------------------
# Foreign-object class vocabulary
# --------------------------------------------------------------------------

FO_CLASSES = [
    "clip", "sponge", "silicone loop", "external drain", "specimen",
    "specimen bag", "needle", "gallstone", "unknown foreign object",
]

CLASS_ALIASES: dict[str, str] = {}
for _c in FO_CLASSES:
    CLASS_ALIASES[_c] = _c
    CLASS_ALIASES[_c + "s"] = _c
CLASS_ALIASES.update({
    "silicon loop": "silicone loop", "silicon loops": "silicone loop",
    "silicone-loop": "silicone loop",
    "drain": "external drain", "drains": "external drain",
    "surgical sponge": "sponge", "surgical sponges": "sponge",
    "surgical clip": "clip", "surgical clips": "clip",
    "gall stone": "gallstone", "gall stones": "gallstone",
    "unknown": "unknown foreign object", "unknown object": "unknown foreign object",
    "none": "none", "no foreign object": "none",
    "no foreign objects": "none", "nothing": "none",
})

QUADRANTS = ["top/left", "top/right", "bottom/left", "bottom/right"]
QUADRANT_ALIASES = {
    "top/left": "top/left", "top left": "top/left", "upper left": "top/left",
    "top-left": "top/left", "topleft": "top/left",
    "top/right": "top/right", "top right": "top/right", "upper right": "top/right",
    "top-right": "top/right", "topright": "top/right",
    "bottom/left": "bottom/left", "bottom left": "bottom/left", "lower left": "bottom/left",
    "bottom-left": "bottom/left", "bottomleft": "bottom/left",
    "bottom/right": "bottom/right", "bottom right": "bottom/right", "lower right": "bottom/right",
    "bottom-right": "bottom/right", "bottomright": "bottom/right",
}

_PUNCT_TAIL = " \t\n.;:,!\"'`*"


def _clean(s: Any) -> str:
    s = str(s).strip().strip(_PUNCT_TAIL).lower()
    return re.sub(r"\s+", " ", s)


# --------------------------------------------------------------------------
# Choice extraction
# --------------------------------------------------------------------------

_CHOICE_MARKERS = [
    r"select one answer\s*:", r"please select one of\s*:", r"choose one\s*:",
    r"choose one of\s*:", r"answer with one of\s*:", r"one of the following\s*:",
]


def extract_choices(question: str) -> list[str]:
    q = str(question).replace("<image>", "").strip()
    tail = None
    for marker in _CHOICE_MARKERS:
        m = re.search(marker, q, flags=re.I)
        if m:
            tail = q[m.end():]
            break
    if tail is None:
        m = re.search(r"\(\s*choose one\s*:([^)]*)\)", q, flags=re.I)
        if not m:
            return []
        tail = m.group(1)
    tail = tail.strip().rstrip(_PUNCT_TAIL)
    tail = re.split(r"(?<=[a-z/])\.\s+[A-Z]", tail)[0]
    parts = re.split(r";|\bor\b|,", tail)
    choices = [_clean(p) for p in parts]
    choices = [c for c in choices if c and len(c) < 60]
    return choices if len(choices) >= 2 else []


def _snap_to_choice(text: str, choices: list[str]) -> str | None:
    t = _clean(text)
    if not t or not choices:
        return None
    if all(_clean(c) in QUADRANT_ALIASES for c in choices):
        return parse_quadrant(t)
    for c in choices:
        if t == c:
            return c
    hits = [c for c in choices if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", t)]
    if hits:
        return max(hits, key=len)
    tt = set(t.split())
    scored = [(len(tt & set(c.split())) / max(len(set(c.split())), 1), c) for c in choices]
    best_score, best = max(scored, key=lambda x: x[0])
    return best if best_score >= 0.5 else None


# --------------------------------------------------------------------------
# Parsers
# --------------------------------------------------------------------------

def parse_number(text: str) -> int | None:
    words = {
        "zero": 0, "none": 0, "no": 0, "one": 1, "two": 2, "three": 3,
        "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
        "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    }
    t = _clean(text)
    if t in words:
        return words[t]
    m = re.findall(r"-?\d+", t)
    if m:
        try:
            return int(m[0])
        except ValueError:
            return None
    for w, v in words.items():
        if re.search(rf"\b{w}\b", t):
            return v
    return None


def parse_binary(text: str) -> str | None:
    t = _clean(text)
    if re.match(r"^(yes|true|correct|affirmative)\b", t):
        return "yes"
    if re.match(r"^(no|false|incorrect|negative)\b", t):
        return "no"
    if re.search(r"\byes\b", t):
        return "yes"
    if re.search(r"\bno\b", t):
        return "no"
    return None


def parse_quadrant(text: str) -> str | None:
    t = re.sub(r"\s*/\s*", "/", _clean(text).replace("_", " "))
    if t in QUADRANT_ALIASES:
        return QUADRANT_ALIASES[t]
    for surface, canon in QUADRANT_ALIASES.items():
        if re.search(rf"\b{re.escape(surface)}\b", t):
            return canon
    return None


def parse_class_set(text: str) -> set[str]:
    t = _clean(text)
    if not t:
        return set()
    if t in {"none", "no", "nothing", "no foreign object", "no foreign objects"}:
        return {"none"}
    out: set[str] = set()
    for surface in sorted(CLASS_ALIASES, key=len, reverse=True):
        canon = CLASS_ALIASES[surface]
        if canon == "none":
            continue
        if re.search(rf"(?<![a-z]){re.escape(surface)}(?![a-z])", t):
            out.add(canon)
            t = re.sub(rf"(?<![a-z]){re.escape(surface)}(?![a-z])", " ", t)
    if not out and re.search(r"\bnone\b", _clean(text)):
        return {"none"}
    return out


def parse_enumerated(text: str) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for m in re.finditer(
        r"([A-Za-z][A-Za-z ]*?)\s*:\s*(top|bottom)\s*/\s*(left|right)", str(text), re.I
    ):
        cls = re.sub(r"^\d+\s*[\.\)]\s*", "", _clean(m.group(1))).strip()
        cls = CLASS_ALIASES.get(cls, cls)
        pairs.append((cls, f"{m.group(2).lower()}/{m.group(3).lower()}"))
    return pairs


def _set_f1(pred: set, gold: set) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(pred & gold)
    if tp == 0:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


# --------------------------------------------------------------------------
# Scorers -> float in [0, 1]
# --------------------------------------------------------------------------

def score_number(pred: str, gold: str, soft: bool = True) -> float:
    g, p = parse_number(gold), parse_number(pred)
    if g is None or p is None:
        return 0.0
    if p == g:
        return 1.0
    if not soft:
        return 0.0
    return max(0.0, 0.5 * (1.0 - abs(p - g) / max(g, 1)))


def score_binary(pred: str, gold: str) -> float:
    g, p = parse_binary(gold), parse_binary(pred)
    return 1.0 if (g is not None and g == p) else 0.0


def score_multiple_choice(pred: str, gold: str, choices: list[str] | None = None) -> float:
    if choices:
        g, p = _snap_to_choice(gold, choices), _snap_to_choice(pred, choices)
        if g is not None:
            return 1.0 if g == p else 0.0
    g, p = parse_quadrant(gold), parse_quadrant(pred)
    if g is not None:
        return 1.0 if g == p else 0.0
    return 1.0 if _clean(pred) == _clean(gold) else 0.0


def score_fo_class(pred: str, gold: str, partial: bool = True) -> float:
    g, p = parse_class_set(gold), parse_class_set(pred)
    if not g:
        return 0.0
    if p == g:
        return 1.0
    if not partial:
        return 0.0
    if g == {"none"} or p == {"none"}:
        return 0.0
    return 0.7 * _set_f1(p, g)


def score_open_ended(pred: str, gold: str) -> float:
    gp = parse_enumerated(gold)
    if gp:
        pp = parse_enumerated(pred)
        if set(pp) == set(gp) and len(pp) == len(gp):
            return 1.0
        return 0.7 * _set_f1(set(pp), set(gp))
    g, p = _clean(gold), _clean(pred)
    if not g:
        return 0.0
    if p == g:
        return 1.0
    return 0.5 * _set_f1(set(p.split()), set(g.split()))


def score_answer(
    pred: str,
    gold: str,
    answer_format: str,
    choices: list[str] | None = None,
    soft_number: bool = True,
    partial_class: bool = True,
) -> float:
    fmt = canonical_format(answer_format)
    try:
        if fmt == "number":
            v = score_number(pred, gold, soft=soft_number)
        elif fmt == "binary":
            v = score_binary(pred, gold)
        elif fmt == "multiple choice":
            v = score_multiple_choice(pred, gold, choices)
        elif fmt == "FO_Class":
            v = score_fo_class(pred, gold, partial=partial_class)
        else:
            v = score_open_ended(pred, gold)
        return float(max(0.0, min(1.0, v)))
    except Exception:
        return 0.0


# --------------------------------------------------------------------------
# Prediction / gold extraction (Miles sample -> strings)
# --------------------------------------------------------------------------

EXCLUDED_CLASSES = {"silicone loop"}


def extract(text: str) -> str:
    """Last-line / <answer> extraction, identical to the reference verifier."""
    t = (text or "").strip()
    m = re.search(r"<answer>(.*?)</answer>", t, re.S | re.I)
    if m:
        return m.group(1).strip()
    t = re.sub(r"</?answer>", "", t, flags=re.I).strip()
    # Qwen3-VL routes chain-of-thought into reasoning_content, so the visible
    # reply IS the answer. Still drop a <think> block if one leaked through.
    t = re.sub(r"<think>.*?</think>", "", t, flags=re.S | re.I)
    return t.split("\n")[-1].strip() if t else ""


def prediction_of(response: str) -> str:
    return extract(str(response or ""))


def gold_of(label: str | None) -> dict:
    if label is None:
        return {"answer": "", "answer_format": "open_ended", "choices": []}
    if isinstance(label, dict):
        return label
    if isinstance(label, str):
        try:
            gt = json.loads(label)
        except (ValueError, TypeError):
            return {"answer": label, "answer_format": "open_ended", "choices": []}
        if isinstance(gt, dict):
            return gt
        return {"answer": str(gt), "answer_format": "open_ended", "choices": []}
    return {"answer": str(label), "answer_format": "open_ended", "choices": []}


def score_row(pred: str, gold: dict) -> dict:
    answer = gold.get("answer", "")
    fmt = gold.get("answer_format", "open_ended")
    choices = gold.get("choices") or None

    correctness = score_answer(pred, answer, fmt, choices=choices, soft_number=True, partial_class=True)
    strict = score_answer(pred, answer, fmt, choices=choices, soft_number=False, partial_class=False)
    exact_match = 1.0 if strict == 1.0 else 0.0

    named = parse_class_set(pred)
    known = set(FO_CLASSES) | {"none"}
    no_hallucinated = 0.0 if (named - known) else 1.0
    no_excluded = 0.0 if (named & EXCLUDED_CLASSES) else 1.0

    return {
        "reward_value": correctness,
        "reward_cat": canonical_format(fmt),
        "exact_match": exact_match,
        "no_hallucinated_class": no_hallucinated,
        "no_excluded_class": no_excluded,
    }


# --------------------------------------------------------------------------
# Miles entrypoint
# --------------------------------------------------------------------------

def reward_fn(args, sample, **kwargs) -> dict:
    """Miles custom reward contract. `sample` is miles.utils.types.Sample."""
    pred = prediction_of(getattr(sample, "response", ""))
    gold = gold_of(getattr(sample, "label", None))
    return score_row(pred, gold)


if __name__ == "__main__":
    # Offline smoke test: no GPU, no Miles. Mirrors the reference test_reward.py.
    cases = [
        ("clip", {"answer": "clip", "answer_format": "fo_class", "choices": []}, 1.0),
        ("3", {"answer": "3", "answer_format": "number", "choices": []}, 1.0),
        ("yes", {"answer": "yes", "answer_format": "binary", "choices": []}, 1.0),
        ("<answer>sponge</answer>", {"answer": "sponge", "answer_format": "fo_class", "choices": []}, 1.0),
    ]
    from types import SimpleNamespace

    ok = True
    for response, gold, expected in cases:
        s = SimpleNamespace(response=response, label=json.dumps(gold))
        out = reward_fn(None, s)
        got = out["reward_value"]
        status = "OK" if abs(got - expected) < 1e-6 else "FAIL"
        ok = ok and status == "OK"
        print(f"{status}: response={response!r} gold={gold} -> reward={got:.3f} exact={out['exact_match']}")
    raise SystemExit(0 if ok else 1)

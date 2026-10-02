"""Answer normalization + scoring for the ORena-FOCUS frame VQA task.

Answer formats follow the official ORena-FOCUS taxonomy:

    open_ended | number | binary | FO_Class | multiple choice

This module is shared by `prepare_dataset.py` (offline) and the verifiers
environment (online, during RL rollouts) so gold labels and model predictions go
through *exactly* the same normalizer. Any drift between the two shows up as
unlearnable reward noise, so keep this file as the single source of truth.
"""

from __future__ import annotations

import re
from typing import Any

# --------------------------------------------------------------------------
# Answer formats
# --------------------------------------------------------------------------

ANSWER_FORMATS = ["open_ended", "number", "binary", "FO_Class", "multiple choice"]

# Aliases so configs/CLI can say multiple_choice, fo_class, etc.
FORMAT_ALIASES = {
    "open_ended": "open_ended",
    "open-ended": "open_ended",
    "openended": "open_ended",
    "number": "number",
    "count": "number",
    "binary": "binary",
    "yes_no": "binary",
    "fo_class": "FO_Class",
    "foclass": "FO_Class",
    "class": "FO_Class",
    "multiple choice": "multiple choice",
    "multiple_choice": "multiple choice",
    "multiple-choice": "multiple choice",
    "mc": "multiple choice",
}


def canonical_format(fmt: str) -> str:
    """Map any reasonable spelling onto one of ANSWER_FORMATS."""
    if fmt in ANSWER_FORMATS:
        return fmt
    return FORMAT_ALIASES.get(str(fmt).strip().lower(), "open_ended")


# --------------------------------------------------------------------------
# Foreign-object class vocabulary
# --------------------------------------------------------------------------

FO_CLASSES = [
    "clip",
    "sponge",
    "silicone loop",
    "external drain",
    "specimen",
    "specimen bag",
    "needle",
    "gallstone",
    "unknown foreign object",
]

# Surface form -> canonical class. Covers plurals, the "Silicon loop" typo that
# appears in the raw annotations, and a few loose synonyms.
CLASS_ALIASES: dict[str, str] = {}
for _c in FO_CLASSES:
    CLASS_ALIASES[_c] = _c
    CLASS_ALIASES[_c + "s"] = _c
CLASS_ALIASES.update(
    {
        "silicon loop": "silicone loop",
        "silicon loops": "silicone loop",
        "silicone-loop": "silicone loop",
        "drain": "external drain",
        "drains": "external drain",
        "surgical sponge": "sponge",
        "surgical sponges": "sponge",
        "surgical clip": "clip",
        "surgical clips": "clip",
        "gall stone": "gallstone",
        "gall stones": "gallstone",
        "unknown": "unknown foreign object",
        "unknown object": "unknown foreign object",
        "none": "none",
        "no foreign object": "none",
        "no foreign objects": "none",
        "nothing": "none",
    }
)

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
# Multiple-choice option extraction
# --------------------------------------------------------------------------

_CHOICE_MARKERS = [
    r"select one answer\s*:",
    r"please select one of\s*:",
    r"choose one\s*:",
    r"choose one of\s*:",
    r"answer with one of\s*:",
    r"one of the following\s*:",
]


def extract_choices(question: str) -> list[str]:
    """Pull the explicit option list out of a question, if it has one.

    Returns [] when the question is not multiple choice.
    """
    q = str(question).replace("<image>", "").strip()
    tail = None
    for marker in _CHOICE_MARKERS:
        m = re.search(marker, q, flags=re.I)
        if m:
            tail = q[m.end():]
            break
    if tail is None:
        m = re.search(r"\(\s*choose one\s*:([^)]*)\)", q, flags=re.I)  # inline form
        if not m:
            return []
        tail = m.group(1)

    tail = tail.strip().rstrip(_PUNCT_TAIL)
    tail = re.split(r"(?<=[a-z/])\.\s+[A-Z]", tail)[0]  # stop at a new sentence
    parts = re.split(r";|\bor\b|,", tail)
    choices = [_clean(p) for p in parts]
    choices = [c for c in choices if c and len(c) < 60]
    return choices if len(choices) >= 2 else []


def _snap_to_choice(text: str, choices: list[str]) -> str | None:
    """Map a free-text answer onto one of the offered options."""
    t = _clean(text)
    if not t or not choices:
        return None

    # Use the dedicated quadrant normalizer only when the option set IS the
    # quadrant set ("bottom left" -> bottom/right etc.). Options that merely
    # contain a quadrant phrase (e.g. "upper right abdominal quadrant") must
    # fall through to plain option matching.
    if all(_clean(c) in QUADRANT_ALIASES for c in choices):
        return parse_quadrant(t)

    for c in choices:
        if t == c:
            return c
    # Longest containing option wins ("upper left abdominal quadrant" over "left")
    hits = [c for c in choices if re.search(rf"(?<![a-z]){re.escape(c)}(?![a-z])", t)]
    if hits:
        return max(hits, key=len)
    # Token-overlap fallback
    tt = set(t.split())
    scored = [(len(tt & set(c.split())) / max(len(set(c.split())), 1), c) for c in choices]
    best_score, best = max(scored, key=lambda x: x[0])
    return best if best_score >= 0.5 else None


# --------------------------------------------------------------------------
# Format classification
# --------------------------------------------------------------------------


def classify_answer_format(question: str, answer: str) -> str:
    """Assign one of ANSWER_FORMATS to a (question, answer) pair."""
    q = str(question).replace("<image>", "").strip().lower()
    a = _clean(answer)

    choices = extract_choices(question)
    if choices and _snap_to_choice(a, choices) is not None:
        return "multiple choice"

    if "provide a number" in q or re.fullmatch(r"\d+", a):
        return "number"

    if "yes or no" in q or a in {"yes", "no"}:
        return "binary"

    # Enumerated "1. Sponge: bottom/left" answers are open_ended, not FO_Class.
    if re.search(r"\d\s*\.\s*[a-z ]+\s*:\s*(top|bottom)\s*/\s*(left|right)", a):
        return "open_ended"

    if ("class name" in q or "class names" in q or "foreign object" in q) and parse_class_set(answer):
        return "FO_Class"

    return "open_ended"


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
    """Order-insensitive set of canonical FO classes mentioned in `text`.

    Returns {"none"} when the answer explicitly says none, and an empty set when
    nothing recognizable was found (which scores 0 against any gold).
    """
    t = _clean(text)
    if not t:
        return set()
    if t in {"none", "no", "nothing", "no foreign object", "no foreign objects"}:
        return {"none"}

    out: set[str] = set()
    # Longest-alias-first matching so "specimen bag" wins over "specimen".
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
    """Parse '1. Sponge: bottom/left 2. Clip: top/right' into [(class, quadrant), ...]."""
    pairs: list[tuple[str, str]] = []
    for m in re.finditer(
        r"([A-Za-z][A-Za-z ]*?)\s*:\s*(top|bottom)\s*/\s*(left|right)", str(text), re.I
    ):
        cls = re.sub(r"^\d+\s*[\.\)]\s*", "", _clean(m.group(1))).strip()
        cls = CLASS_ALIASES.get(cls, cls)
        pairs.append((cls, f"{m.group(2).lower()}/{m.group(3).lower()}"))
    return pairs


# --------------------------------------------------------------------------
# Scorers -> float in [0, 1]
# --------------------------------------------------------------------------


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


def score_number(pred: str, gold: str, soft: bool = True) -> float:
    g, p = parse_number(gold), parse_number(pred)
    if g is None or p is None:
        return 0.0
    if p == g:
        return 1.0
    if not soft:
        return 0.0
    # Dense shaping: partial credit decaying with absolute error, capped well
    # below 1.0 so exactness always strictly dominates. Matters for the dense
    # scenes where a pure 0/1 reward gives almost no gradient.
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
    # Never award partial credit for hallucinating classes onto a "none" gold,
    # or for answering "none" when objects are present.
    if g == {"none"} or p == {"none"}:
        return 0.0
    return 0.7 * _set_f1(p, g)


def score_open_ended(pred: str, gold: str) -> float:
    # Enumerated "1. Clip: bottom/left" answers get pair-set scoring.
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
    # Token-overlap fallback for the small tail of prose answers.
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


def normalize_gold(answer: str, answer_format: str, choices: list[str] | None = None) -> str:
    """Canonical string form of a gold answer, used to clean the dataset."""
    fmt = canonical_format(answer_format)
    if fmt == "number":
        v = parse_number(answer)
        return str(v) if v is not None else _clean(answer)
    if fmt == "binary":
        return parse_binary(answer) or _clean(answer)
    if fmt == "multiple choice":
        return _snap_to_choice(answer, choices or []) or parse_quadrant(answer) or _clean(answer)
    if fmt == "FO_Class":
        s = parse_class_set(answer)
        if s == {"none"} or not s:
            return "none"
        return ", ".join(sorted(s))
    pairs = parse_enumerated(answer)
    if pairs:
        return "; ".join(f"{c}: {p}" for c, p in pairs)
    return str(answer).strip()


ANSWER_FORMAT_HINT = {
    "number": "a single integer, e.g. 3",
    "binary": "exactly one word: yes or no",
    "multiple choice": "exactly one of the options offered in the question, copied verbatim",
    "FO_Class": "a comma-separated list of foreign object class names, or none",
    "open_ended": "a short phrase",
}
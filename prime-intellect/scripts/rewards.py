"""Reward functions for the 4 Prime Intellect RL experiments.

Direct port of FIREWORKS_4EXP_CONFIG.md §8. These are plain functions (the
"SageMaker form" in the reference). prime-rl consumes rewards through a live
`verifiers` Environment — see 04_REWARD_EVALUATORS.md for the wrapper. Keep this
file framework-agnostic so the exact same code can be unit-tested and reused
for the held-out eval in 09.

ORena  reward = exact match (per-answer-format normalization); 4-bucket
        meanAccuracy is computed post-run from the per-format acc_* metrics.
GoEmotions reward = Jaccard of the predicted vs ground-truth label set;
        micro-F1 tracked alongside.
"""
import re


# --- ORena (surgical VQA) -------------------------------------------------

def normalize(text: str) -> str:
    """Lowercase, strip surrounding whitespace/punctuation, collapse spaces."""
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = t.strip(".,;:!?\"'` ")
    return t


def _number_tol_ok(pred: str, gold: str, tol: float = 0.01) -> bool:
    try:
        p = float(re.sub(r"[^0-9.\-]", "", pred or "") or "nan")
        g = float(re.sub(r"[^0-9.\-]", "", gold or "") or "nan")
        return abs(p - g) <= tol
    except ValueError:
        return False


def reward_orena(pred_text: str, ground_truth: dict) -> dict:
    """pred_text = model's final answer; ground_truth = {"answer","answer_format","choices"}."""
    fmt = ground_truth.get("answer_format", "open_ended")
    gold = str(ground_truth.get("answer", ""))
    pred = normalize(pred_text)
    gold_n = normalize(gold)
    exact = 1.0
    if fmt == "number":
        exact = 1.0 if _number_tol_ok(pred, gold_n) else 0.0
    else:
        exact = 1.0 if pred == gold_n else 0.0
    return {
        "score": exact,
        "reason": f"exact_match={exact}",
        "metrics": {
            "exact_match": exact,
            f"acc_{fmt}": exact,
        },
    }


# --- GoEmotions (multi-label text) ---------------------------------------

_SYNONYMS = {
    "gratitude": "gratitude", "thankful": "gratitude", "thankfulness": "gratitude",
    "approval": "approval",
    # extend as needed from emotions.txt; unknown tokens are dropped
}


def parse_labels(text: str) -> set:
    out = set()
    for tok in re.split(r"[,\n;|]+", text or ""):
        tok = normalize(tok)
        if not tok:
            continue
        out.add(_SYNONYMS.get(tok, tok))
    return out


def reward_goemotions(pred_text: str, ground_truth: dict) -> dict:
    """ground_truth = {"labels": [...]}."""
    pred = parse_labels(pred_text)
    gold = set(ground_truth.get("labels", []))
    inter = pred & gold
    union = pred | gold
    f1 = 2 * len(inter) / (len(pred) + len(gold)) if (pred or gold) else 1.0
    jac = len(inter) / len(union) if union else 1.0
    return {
        "score": jac,
        "reason": f"f1={f1:.4f} jaccard={jac:.4f}",
        "metrics": {"micro_f1": f1, "jaccard": jac},
    }


if __name__ == "__main__":  # quick self-test
    assert reward_orena("3", {"answer": "3", "answer_format": "number"})["score"] == 1.0
    assert reward_orena("Clip", {"answer": "clip", "answer_format": "fo_class"})["score"] == 1.0
    assert reward_goemotions("joy, gratitude", {"labels": ["joy", "gratitude"]})["score"] == 1.0
    assert reward_goemotions("joy", {"labels": ["joy", "gratitude"]})["score"] < 1.0
    print("rewards self-test OK")

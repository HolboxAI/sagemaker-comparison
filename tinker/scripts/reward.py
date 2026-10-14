#!/usr/bin/env python3
"""Verifiable reward for CholecT50 surgical action triplet recognition (RLVR).

Reward is a dense, per-sample decomposition of triplet multi-label F1:
    reward = 0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)
Component F1 gives partial credit when only part of a triplet is right, so the
reward is non-zero on most parseable completions (avoids the reward==0 trap).
Class imbalance is handled at the data level (frame balancing), not in the reward.

All triplets are canonical strings "instrument-verb-target"; ground truth is
stored the same way in the data manifests.

Compatible with trl GRPOTrainer: `reward(completions, ...)` receives
completions as list[list[{"role","content"}]] and returns a flat list[float].
"""
from typing import Optional, Set

INSTRUMENTS = {"grasper", "bipolar", "hook", "scissors", "clipper", "irrigator"}
VERBS = {"grasp", "retract", "dissect", "coagulate", "clip", "cut",
         "aspirate", "irrigate", "pack", "null_verb"}
TARGETS = {"gallbladder", "cystic_plate", "cystic_duct", "cystic_artery",
           "cystic_pedicle", "blood_vessel", "fluid", "abdominal_wall_cavity",
           "liver", "adhesion", "omentum", "peritoneum", "gut", "specimen_bag",
           "null_target"}

PROMPT = (
    "You are a surgical activity recognition model. List every surgical action "
    "triplet (instrument-verb-target) visible in the endoscopic frame. Reply only "
    "as comma-separated triplets, each formatted instrument-verb-target. If no "
    "action is visible, reply 'none'."
)


def parse_triplets(text: Optional[str]) -> Optional[Set[str]]:
    """Parse a completion into a set of canonical 'instrument-verb-target' strings.

    Returns None for a malformed completion (-> reward 0), an empty set for an
    explicit 'none', else the parsed triplets.
    """
    if text is None:
        return None
    t = text.strip()
    if t.lower() in ("", "none", "no action", "nothing", "no triplets"):
        return set()
    parts = [p.strip().strip(".;()[]").lower()
             for p in t.replace("\n", ",").split(",") if p.strip()]
    trips: Set[str] = set()
    for p in parts:
        comps = p.split("-")
        if len(comps) != 3:
            return None
        i, v, tg = (c.strip() for c in comps)
        if i not in INSTRUMENTS or v not in VERBS or tg not in TARGETS:
            return None
        trips.add(f"{i}-{v}-{tg}")
    return trips


def _f1(pred: Set[str], gt: Set[str]) -> float:
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gt)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def reward_single(pred: Optional[Set[str]], gt: dict) -> float:
    """pred: parsed triplet set (or None); gt: {'triplets','instruments','verbs','targets'}."""
    if pred is None:
        return 0.0
    triplet_f = _f1(pred, set(gt["triplets"]))
    inst_f = _f1({t.split("-")[0] for t in pred}, set(gt["instruments"]))
    verb_f = _f1({t.split("-")[1] for t in pred}, set(gt["verbs"]))
    targ_f = _f1({t.split("-")[2] for t in pred}, set(gt["targets"]))
    return 0.4 * triplet_f + 0.6 * (inst_f + verb_f + targ_f) / 3.0


def _completion_text(comp) -> str:
    """Extract assistant text from a trl completion (str / dict / message list)."""
    if isinstance(comp, str):
        return comp
    if isinstance(comp, list):
        return "\n".join(_completion_text(c) for c in comp)
    if isinstance(comp, dict):
        content = comp.get("content", comp.get("text", ""))
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text")
        return str(content)
    return str(comp)


def reward(completions, **kwargs) -> list:
    """trl GRPO reward entrypoint.

    The GRPO dataset carries ground truth as columns `triplets`, `instruments`,
    `verbs`, `targets` (one list per sample); trl passes them here as flat lists
    aligned one-to-one with `completions`.
    """
    trips = kwargs.get("triplets") or []
    insts = kwargs.get("instruments") or []
    verbs = kwargs.get("verbs") or []
    targs = kwargs.get("targets") or []
    out = []
    for idx, comp in enumerate(completions):
        gt = {
            "triplets": trips[idx] if idx < len(trips) else [],
            "instruments": insts[idx] if idx < len(insts) else [],
            "verbs": verbs[idx] if idx < len(verbs) else [],
            "targets": targs[idx] if idx < len(targs) else [],
        }
        out.append(reward_single(parse_triplets(_completion_text(comp)), gt))
    return out


if __name__ == "__main__":
    gt = {
        "triplets": ["grasper-retract-gallbladder", "hook-dissect-cystic_duct"],
        "instruments": ["grasper", "hook"],
        "verbs": ["retract", "dissect"],
        "targets": ["gallbladder", "cystic_duct"],
    }
    cases = [
        ("grasper-retract-gallbladder, hook-dissect-cystic_duct", "exact match"),
        ("grasper-retract-gallbladder", "one of two"),
        ("grasper-retract-liver", "wrong target only"),
        ("none", "explicit none"),
        ("random garbage text", "malformed"),
        ("grasper-retract-gallbladder, hook-dissect-cystic_duct, hook-dissect-cystic_duct", "dup"),
    ]
    print("reward unit tests:")
    for text, label in cases:
        p = parse_triplets(text)
        r = reward_single(p, gt)
        print(f"  {label:22s} -> parse={p}  reward={r:.3f}")
    assert reward_single(parse_triplets("grasper-retract-gallbladder, hook-dissect-cystic_duct"), gt) == 1.0
    assert reward_single(parse_triplets("random garbage"), gt) == 0.0
    print("OK: exact=1.0, malformed=0.0")

#!/usr/bin/env python3
"""CholecT50 eval: greedy-decode triplets and score multi-label F1 vs ground truth.

Reuses the same dense F1 decomposition as the RL reward so the before/after
numbers are directly comparable to what the reward is optimizing.
"""
from __future__ import annotations

import torch

from cholec_data import build_grpo_dataset
from reward import PROMPT, _f1, parse_triplets, reward_single


@torch.no_grad()
def evaluate(model, processor, split: str = "test",
             cache_dir: str = "/tmp/cholec_frames", max_samples: int = 300,
             max_new_tokens: int = 256) -> dict[str, list[float]]:
    ds = build_grpo_dataset(split, cache_dir, max_samples=max_samples)
    device = next(model.parameters()).device
    m = {"triplet_f1": [], "inst_f1": [], "verb_f1": [], "targ_f1": [],
         "combined": [], "exact": []}
    for ex in ds:
        img = ex["images"][0]
        messages = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt").to(device)
        out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        gen = processor.batch_decode(
            out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]

        gt = {"triplets": ex["triplets"], "instruments": ex["instruments"],
              "verbs": ex["verbs"], "targets": ex["targets"]}
        pred = parse_triplets(gen)
        m["combined"].append(reward_single(pred, gt))
        m["exact"].append(1.0 if (pred is not None and pred == set(gt["triplets"])) else 0.0)
        m["triplet_f1"].append(_f1(pred, set(gt["triplets"])) if pred is not None else 0.0)
        m["inst_f1"].append(_f1({t.split("-")[0] for t in pred}, set(gt["instruments"])) if pred is not None else 0.0)
        m["verb_f1"].append(_f1({t.split("-")[1] for t in pred}, set(gt["verbs"])) if pred is not None else 0.0)
        m["targ_f1"].append(_f1({t.split("-")[2] for t in pred}, set(gt["targets"])) if pred is not None else 0.0)
    return m


def mean_metrics(m: dict[str, list[float]]) -> dict[str, float]:
    return {k: (sum(v) / len(v) if v else 0.0) for k, v in m.items()}


def summarize(tag: str, m: dict[str, list[float]]) -> None:
    n = len(m["combined"])
    if n == 0:
        print(f"EVAL [{tag}]: no examples", flush=True)
        return
    means = mean_metrics(m)
    print("=" * 64, flush=True)
    print(f"EVAL [{tag}]  n={n}", flush=True)
    print(f"  triplet_F1      {means['triplet_f1']:.4f}", flush=True)
    print(f"  instrument_F1   {means['inst_f1']:.4f}", flush=True)
    print(f"  verb_F1         {means['verb_f1']:.4f}", flush=True)
    print(f"  target_F1       {means['targ_f1']:.4f}", flush=True)
    print(f"  COMBINED reward {means['combined']:.4f}", flush=True)
    print(f"  exact_match     {means['exact']:.4f}", flush=True)
    print("=" * 64, flush=True)

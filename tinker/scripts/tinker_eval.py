#!/usr/bin/env python3
"""Evaluate a CholecT50 triplet model on Tinker and report F1 metrics.

Samples one greedy completion per held-out frame and scores it against the
ground-truth triplets with the same component-F1 decomposition used by the RL
reward. Use this to produce the before/after numbers for the demo:

    base model  ->  --base
    SFT ckpt    ->  --ckpt "$(cat /tmp/cholec_tinker_sft_ckpt.txt)"
    RL ckpt     ->  --ckpt "$(cat /tmp/cholec_tinker_rl_ckpt.txt)"

    TINKER_API_KEY=... python tinker_eval.py --split val --base
    TINKER_API_KEY=... python tinker_eval.py --split val --ckpt tinker://.../weights/...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tinker
from PIL import Image

from tinker_cookbook import renderers
from tinker_cookbook.image_processing_utils import get_image_processor
from tinker_cookbook.tokenizer_utils import get_tokenizer

from cholec_data import download_frames, load_manifest
from reward import PROMPT, parse_triplets, reward_single

MODEL = "Qwen/Qwen3.5-4B"
RENDERER_NAME = "qwen3_5_disable_thinking"


def _f1(pred: set, gt: set) -> float:
    tp = len(pred & gt)
    if tp == 0:
        return 0.0
    prec = tp / len(pred)
    rec = tp / len(gt)
    return 2 * prec * rec / (prec + rec) if prec + rec else 0.0


async def run(args):
    tokenizer = get_tokenizer(MODEL)
    imgproc = get_image_processor(MODEL)
    renderer = renderers.get_renderer(RENDERER_NAME, tokenizer, image_processor=imgproc, model_name=MODEL)

    sc = tinker.ServiceClient()
    if args.base:
        sampling_client = sc.create_sampling_client(base_model=MODEL)
    else:
        sampling_client = sc.create_sampling_client(model_path=args.ckpt)

    rows = load_manifest(args.split)
    if args.max_samples:
        rows = rows[: args.max_samples]
    paths = download_frames([r["member"] for r in rows], args.cache_dir)

    tot_triplet = tot_inst = tot_verb = tot_targ = tot_blend = 0.0
    n = 0
    t0 = time.time()
    for i, r in enumerate(rows):
        p = paths.get(r["member"])
        if not p:
            continue
        img = Image.open(p).convert("RGB")
        convo = [{"role": "user", "content": [
            {"type": "image", "image": img},
            {"type": "text", "text": PROMPT},
        ]}]
        prompt = renderer.build_generation_prompt(convo)
        resp = await sampling_client.sample_async(
            prompt=prompt, num_samples=1,
            sampling_params=tinker.SamplingParams(
                max_tokens=args.max_tokens, temperature=0.0,
                stop=renderer.get_stop_sequences()),
        )
        message, _term = renderer.parse_response(resp.sequences[0].tokens)
        text = renderers.get_text_content(message)
        pred = parse_triplets(text)

        gt_triplets = set(r["triplets"])
        gt_inst = set(r["instruments"])
        gt_verb = set(r["verbs"])
        gt_targ = set(r["targets"])
        if pred is None:
            triplet_f = inst_f = verb_f = targ_f = 0.0
        else:
            triplet_f = _f1(pred, gt_triplets)
            inst_f = _f1({t.split("-")[0] for t in pred}, gt_inst)
            verb_f = _f1({t.split("-")[1] for t in pred}, gt_verb)
            targ_f = _f1({t.split("-")[2] for t in pred}, gt_targ)
        blend = reward_single(pred, {
            "triplets": r["triplets"], "instruments": r["instruments"],
            "verbs": r["verbs"], "targets": r["targets"]})

        tot_triplet += triplet_f
        tot_inst += inst_f
        tot_verb += verb_f
        tot_targ += targ_f
        tot_blend += blend
        n += 1
        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(rows)}  tripletF1={tot_triplet/n:.3f} "
                  f"blend={tot_blend/n:.3f} ({time.time()-t0:.0f}s)", flush=True)

    out = {
        "model": "base" if args.base else args.ckpt,
        "split": args.split,
        "n": n,
        "triplet_f1": round(tot_triplet / n, 4),
        "instrument_f1": round(tot_inst / n, 4),
        "verb_f1": round(tot_verb / n, 4),
        "target_f1": round(tot_targ / n, 4),
        "blended_reward": round(tot_blend / n, 4),
    }
    print(json.dumps(out, indent=2))
    if args.out:
        with open(args.out, "w") as f:
            json.dump(out, f, indent=2)
        print(f"wrote {args.out}")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="val")
    ap.add_argument("--base", action="store_true", help="evaluate the base model (no checkpoint)")
    ap.add_argument("--ckpt", type=str, default=None, help="tinker:// checkpoint sampler path")
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all")
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--cache_dir", type=str, default="/tmp/cholec_frames")
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()
    if not args.base and not args.ckpt:
        raise SystemExit("provide --base or --ckpt")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

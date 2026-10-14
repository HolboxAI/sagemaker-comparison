#!/usr/bin/env python3
"""CholecT50 RL: GRPO on top of the SFT checkpoint, with the verifiable
triplet-F1 reward (RLVR).

Loads the merged SFT model from $BT_LOAD_CHECKPOINT_DIR/rank-0 (handed off from
the SFT job), applies a fresh LoRA, and optimizes with trl GRPOTrainer. Runs a
before/after eval (SFT policy vs. RL policy) on a fixed test subset.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from cholec_data import build_grpo_dataset
from eval import evaluate, mean_metrics, summarize
from reward import reward

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def resolve_sft_dir():
    base = os.environ.get("BT_LOAD_CHECKPOINT_DIR", "/tmp/loaded_checkpoints")
    d = os.path.join(base, "rank-0")
    if os.path.exists(os.path.join(d, "config.json")):
        return d
    print(f"WARN: no SFT checkpoint at {d}; falling back to base {MODEL_ID}", flush=True)
    return MODEL_ID


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--num_generations", type=int, default=8)
    ap.add_argument("--max_completion_length", type=int, default=256)
    ap.add_argument("--max_prompt_length", type=int, default=8192)
    ap.add_argument("--max_steps", type=int, default=150)
    ap.add_argument("--max_samples", type=int, default=600)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--eval_samples", type=int, default=300)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    cache_dir = os.environ.get("FRAME_CACHE", "/tmp/cholec_frames")
    out_dir = os.environ.get("BT_CHECKPOINT_DIR", "./grpo_checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    sft_dir = resolve_sft_dir()
    print(f"loading processor + model from {sft_dir}", flush=True)
    processor = AutoProcessor.from_pretrained(sft_dir, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        sft_dir, dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda")

    results = {}

    if not args.skip_eval:
        print("=== BEFORE (SFT policy) ===", flush=True)
        before = evaluate(model, processor, split="test",
                          cache_dir=cache_dir, max_samples=args.eval_samples)
        summarize("sft_policy", before)
        results["sft_policy"] = mean_metrics(before)

    ds = build_grpo_dataset(
        "train", cache_dir, max_samples=args.max_samples)
    print(f"GRPO train examples: {len(ds)}", flush=True)

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = GRPOConfig(
        output_dir=out_dir,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        max_steps=args.max_steps,
        logging_steps=1,
        save_steps=10_000_000,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=peft_config,
    )
    trainer.train()

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"RL_SAVED {out_dir}", flush=True)

    if not args.skip_eval:
        print("=== AFTER (RL policy) ===", flush=True)
        after = evaluate(merged, processor, split="test",
                         cache_dir=cache_dir, max_samples=args.eval_samples)
        summarize("rl", after)
        results["rl"] = mean_metrics(after)

    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("RL_DONE", flush=True)


if __name__ == "__main__":
    main()

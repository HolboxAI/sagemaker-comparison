#!/usr/bin/env python
"""Experiment 2 -- ORena VLM RL (GRPO) via TRL GRPOTrainer.

Fresh LoRA on the base Qwen/Qwen3.5-9B (no warm-start, per config §1). Reward is
the per-format verifiable scorer in orena_reward.py.
"""
from __future__ import annotations

import argparse
import os

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import GRPOConfig, GRPOTrainer

from orena_data import build_grpo_dataset
from orena_reward import reward_func

MODEL_ID = "Qwen/Qwen3.5-9B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-6)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=128)
    ap.add_argument("--max_steps", type=int, default=50)
    ap.add_argument("--max_samples", type=int, default=256)
    args = ap.parse_args()

    data_dir = os.environ.get("SM_CHANNEL_DATA", "/opt/ml/input/data/data")
    frames_root = os.environ.get("SM_CHANNEL_FRAMES", "/opt/ml/input/data/frames")

    print("loading processor + model", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    processor.tokenizer.padding_side = "left"

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)

    fo_defs_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fo_defs.txt")
    fo_defs = open(fo_defs_path, encoding="utf-8").read().strip()

    ds = build_grpo_dataset(
        os.path.join(data_dir, "frames_train_rl.parquet"),
        frames_root, fo_defs, args.max_samples)
    print(f"rl examples (frames resolved): {len(ds)}", flush=True)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = GRPOConfig(
        output_dir="/opt/ml/checkpoints",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
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
        reward_funcs=reward_func,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("/opt/ml/model")
    print("RL_DONE", flush=True)


if __name__ == "__main__":
    main()

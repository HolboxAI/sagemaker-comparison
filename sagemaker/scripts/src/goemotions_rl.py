#!/usr/bin/env python
"""Experiment 4 -- GoEmotions RL (GRPO) via TRL GRPOTrainer.

Reward = Jaccard of the predicted vs ground-truth label set (micro-F1 tracked
in logs). Reads the RL-format JSONL from the `train` SageMaker channel.
"""
from __future__ import annotations

import argparse
import json
import os
import re

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GRPOConfig, GRPOTrainer

MODEL_ID = "Qwen/Qwen3-0.6B"
SYSTEM = ("Classify the text into one or more of the 28 GoEmotions labels. "
          "Reply with a comma-separated list of label names only.")

LABELS = ["admiration", "amusement", "anger", "annoyance", "approval", "caring",
          "confusion", "curiosity", "desire", "disappointment", "disapproval",
          "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
          "joy", "love", "nervousness", "optimism", "pride", "realization",
          "relief", "remorse", "sadness", "surprise", "neutral"]

# synonym map -> canonical label (config §8.2)
ALIASES = {l: l for l in LABELS}
for a, c in [("anger", "anger"), ("annoyance", "annoyance"), ("joy", "joy"),
             ("sadness", "sadness"), ("surprise", "surprise")]:
    ALIASES[a] = c


def parse_labels(text: str) -> set[str]:
    out = set()
    for part in re.split(r"[,;]|\band\b", text or ""):
        p = re.sub(r"\s+", " ", part.strip().lower()).strip(" .!?")
        if p in ALIASES:
            out.add(ALIASES[p])
    return out


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


def reward_func(completions, labels, **kwargs):
    """TRL 1.14 reward entry: completions list[[{role,content}]], labels list[list[str]] — both flat, one entry per completion."""
    rewards = []
    for comp, gt in zip(completions, labels):
        g = set(gt)
        p = parse_labels(_completion_text(comp))
        inter = p & g
        union = p | g
        jac = len(inter) / len(union) if union else 1.0
        rewards.append(jac)
    return rewards


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--beta", type=float, default=0.04)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--num_generations", type=int, default=4)
    ap.add_argument("--max_completion_length", type=int, default=64)
    ap.add_argument("--max_train_samples", type=int, default=0)
    args = ap.parse_args()

    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    print("loading tokenizer + model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)

    rows = load_jsonl(os.path.join(data_dir, "train_rl.jsonl"))
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    records = [
        {"prompt": [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["messages"][-1]["content"]}],
         "labels": r["ground_truth"]["labels"]}
        for r in rows
    ]
    ds = Dataset.from_list(records)
    print(f"rl prompts: {len(ds)}", flush=True)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = GRPOConfig(
        output_dir="/opt/ml/checkpoints",
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=2,
        learning_rate=args.lr,
        beta=args.beta,
        temperature=args.temperature,
        num_train_epochs=1,
        logging_steps=1,
        save_steps=10_000_000,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("/opt/ml/model")
    print("RL_DONE", flush=True)


if __name__ == "__main__":
    main()

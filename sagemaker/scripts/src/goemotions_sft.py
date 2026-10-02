#!/usr/bin/env python
"""Experiment 3 -- GoEmotions SFT (LoRA) via TRL SFTTrainer.

Reads the CHAT-format JSONL from the `train` SageMaker channel and fine-tunes
Qwen/Qwen3-0.6B with a LoRA adapter. Single epoch by default (short demo run).
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3-0.6B"
SYSTEM = ("Classify the text into one or more of the 28 GoEmotions labels. "
          "Reply with a comma-separated list of label names only.")


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_seq_length", type=int, default=512)
    ap.add_argument("--micro_batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--max_train_samples", type=int, default=0, help="0 = all rows")
    args = ap.parse_args()

    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")

    print("loading tokenizer + model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)

    rows = load_jsonl(os.path.join(data_dir, "train_sft.jsonl"))
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    ds = Dataset.from_list(rows)
    print(f"train examples: {len(ds)}", flush=True)

    peft_config = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM", target_modules="all-linear")

    def formatting_func(ex):
        return tokenizer.apply_chat_template(ex["messages"], tokenize=False)

    cfg = SFTConfig(
        output_dir="/opt/ml/checkpoints",
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.micro_batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=10,
        save_steps=10_000_000,
        bf16=True,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=tokenizer,
        formatting_func=formatting_func,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("/opt/ml/model")
    print("SFT_DONE", flush=True)


if __name__ == "__main__":
    main()

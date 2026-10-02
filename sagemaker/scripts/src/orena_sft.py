#!/usr/bin/env python
"""Experiment 1 -- ORena VLM SFT (LoRA) via TRL SFTTrainer.

Reads the swift-style CHAT JSONL (messages + `<image>` string + `images` array),
remaps the local image paths to the SageMaker `frames` channel, and fine-tunes
Qwen/Qwen3.5-9B (vision + language) with a LoRA adapter.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from datasets import Dataset
from peft import LoraConfig
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

MODEL_ID = "Qwen/Qwen3.5-9B"


def load_jsonl(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def convert_row(row: dict, frames_root: str):
    img_paths = row.get("images", [])
    if not img_paths:
        return None
    pil = []
    for p in img_paths:
        local = p.replace("/data/curated_frames/", "").lstrip("/")
        fp = os.path.join(frames_root, local)
        if not os.path.exists(fp):
            return None
        pil.append(Image.open(fp).convert("RGB"))

    # Keep message content as plain strings; trl's
    # DataCollatorForVisionLanguageModeling -> prepare_multimodal_messages converts
    # them to structured blocks and injects the images before the first user turn.
    new_msgs = []
    for m in row["messages"]:
        if m["role"] == "user":
            text = m["content"].replace("<image>", "").strip()
            new_msgs.append({"role": "user", "content": text})
        else:
            new_msgs.append(m)
    return {"messages": new_msgs, "images": pil}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_seq_length", type=int, default=2048)
    ap.add_argument("--micro_batch", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=64)
    ap.add_argument("--lora_alpha", type=int, default=128)
    ap.add_argument("--max_train_samples", type=int, default=0)
    args = ap.parse_args()

    data_dir = os.environ.get("SM_CHANNEL_TRAIN", "/opt/ml/input/data/train")
    frames_root = os.environ.get("SM_CHANNEL_FRAMES", "/opt/ml/input/data/frames")

    print("loading processor + model", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)

    rows = load_jsonl(os.path.join(data_dir, "train.jsonl"))
    if args.max_train_samples:
        rows = rows[: args.max_train_samples]
    records = [r for r in (convert_row(x, frames_root) for x in rows) if r]
    ds = Dataset.from_list(records)
    print(f"train examples (frames resolved): {len(ds)}", flush=True)

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

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
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        dataloader_num_workers=0,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=ds,
        processing_class=processor,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model("/opt/ml/model")
    print("SFT_DONE", flush=True)


if __name__ == "__main__":
    main()

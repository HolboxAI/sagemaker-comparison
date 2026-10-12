#!/usr/bin/env python3
"""CholecT50 SFT: Qwen3-VL-4B-Instruct -> surgical action triplet recognition.

LoRA fine-tune with trl SFTTrainer, then merge the adapter into a full model and
save it to $BT_CHECKPOINT_DIR so the RL job can load it as its policy/reference.
Also runs a before/after eval (base model vs. SFT model) on a fixed test subset.
"""
from __future__ import annotations

import argparse
import json
import os

import torch
from peft import LoraConfig
from transformers import AutoModelForImageTextToText, AutoProcessor
from trl import SFTConfig, SFTTrainer

from cholec_data import build_sft_dataset
from eval import evaluate, mean_metrics, summarize

MODEL_ID = "Qwen/Qwen3-VL-4B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--max_seq_length", type=int, default=8192)
    ap.add_argument("--micro_batch", type=int, default=2)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--lora_rank", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--max_train_samples", type=int, default=0)
    ap.add_argument("--eval_samples", type=int, default=300)
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    cache_dir = os.environ.get("FRAME_CACHE", "/tmp/cholec_frames")
    out_dir = os.environ.get("BT_CHECKPOINT_DIR", "./checkpoints")
    os.makedirs(out_dir, exist_ok=True)

    print("loading processor + model", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_ID, dtype=torch.bfloat16, trust_remote_code=True)
    model = model.to("cuda")

    results = {}

    if not args.skip_eval:
        print("=== BEFORE (base model) ===", flush=True)
        before = evaluate(model, processor, split="test",
                          cache_dir=cache_dir, max_samples=args.eval_samples)
        summarize("base", before)
        results["base"] = mean_metrics(before)

    ds = build_sft_dataset(
        "train", cache_dir, max_samples=args.max_train_samples or None)
    print(f"SFT train examples: {len(ds)}", flush=True)

    peft_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        bias="none", task_type="CAUSAL_LM", target_modules="all-linear")

    cfg = SFTConfig(
        output_dir=out_dir,
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

    merged = trainer.model.merge_and_unload()
    merged.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"SFT_SAVED {out_dir}", flush=True)

    if not args.skip_eval:
        print("=== AFTER (SFT model) ===", flush=True)
        after = evaluate(merged, processor, split="test",
                         cache_dir=cache_dir, max_samples=args.eval_samples)
        summarize("sft", after)
        results["sft"] = mean_metrics(after)

    with open(os.path.join(out_dir, "eval_metrics.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("SFT_DONE", flush=True)


if __name__ == "__main__":
    main()

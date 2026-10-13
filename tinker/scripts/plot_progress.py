#!/usr/bin/env python3
"""Render local PNG graphs from the training logs.

  python plot_progress.py --sft      # loss curve from /tmp/cholec_tinker_sft.jsonl
  python plot_progress.py --rl DIR   # reward curve from a tinker RL log dir

Writes PNGs under cholec-rl/runs/. Re-run anytime to refresh the plots as a
run progresses.
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")


def plot_sft(log_path: str, out: str) -> None:
    rows = [json.loads(l) for l in open(log_path) if l.strip()]
    # Use a monotonic index: the JSONL is appended across smoke tests + the full
    # run, so the per-run "step" field restarts and is not monotonic.
    steps = list(range(1, len(rows) + 1))
    loss = [r["loss"] for r in rows]
    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, loss, marker="o", ms=3, lw=1.5)
    plt.xlabel("training step (cumulative)")
    plt.ylabel("SFT loss (mean NLL)")
    plt.title("CholecT50 SFT loss (Qwen3.5-4B LoRA, Tinker)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}  ({len(rows)} steps, last loss={loss[-1]:.4f})")


def plot_rl(log_dir: str, out: str) -> None:
    # tinker_cookbook ml_log stores per-step metrics as JSONL in the log dir.
    metrics_path = None
    for cand in ("metrics.jsonl", "train_metrics.jsonl", "metrics.json"):
        p = os.path.join(log_dir, cand)
        if os.path.exists(p):
            metrics_path = p
            break
    if metrics_path is None:
        # search one level deep
        for root, _dirs, files in os.walk(log_dir):
            for f in files:
                if f.endswith(".jsonl") and "metric" in f.lower():
                    metrics_path = os.path.join(root, f)
                    break
    if metrics_path is None:
        raise SystemExit(f"no metrics file found under {log_dir}")

    steps, rewards = [], []
    for line in open(metrics_path):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        step = d.get("step", d.get("progress/batch", len(steps)))
        r = None
        for k in ("env/all/reward/total", "env/all/triplet_reward", "reward/mean",
                  "rewards/mean", "reward", "trajectory/reward/mean",
                  "train/reward/mean", "mean_reward"):
            if k in d and isinstance(d[k], (int, float)):
                r = d[k]
                break
        if r is None:
            continue
        steps.append(step)
        rewards.append(r)

    if not rewards:
        raise SystemExit(f"no reward metric found in {metrics_path}")

    plt.figure(figsize=(8, 4.5))
    plt.plot(steps, rewards, marker="o", ms=3, lw=1.5)
    plt.xlabel("iteration")
    plt.ylabel("mean reward (triplet-F1 blend)")
    plt.title("CholecT50 RL reward (GRPO/RLVR, Tinker)")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"wrote {out}  ({len(steps)} iters, last reward={rewards[-1]:.4f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sft", action="store_true")
    ap.add_argument("--rl", metavar="LOG_DIR")
    args = ap.parse_args()
    os.makedirs(RUNS, exist_ok=True)
    if args.sft:
        plot_sft("/tmp/cholec_tinker_sft.jsonl", os.path.join(RUNS, "sft_loss.png"))
    if args.rl:
        plot_rl(args.rl, os.path.join(RUNS, "rl_reward.png"))


if __name__ == "__main__":
    main()

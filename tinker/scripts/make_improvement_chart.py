#!/usr/bin/env python3
"""Before/after bar chart: base vs SFT vs RL for every CholecT50 eval metric.

Reads the three eval JSONs produced by tinker_eval.py (eval_base.json,
eval_sft.json, eval_rl.json) and writes a grouped bar chart showing the SFT ->
RL improvement. Run from anywhere; it locates the sibling metrics/ dir.

    python make_improvement_chart.py            # uses ../metrics relative to cwd
    python make_improvement_chart.py --dir ../metrics
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# (key in eval JSON, human label)
METRICS = [
    ("triplet_f1", "triplet F1\n(exact triplet match)"),
    ("instrument_f1", "instrument F1"),
    ("verb_f1", "verb F1"),
    ("target_f1", "target F1"),
    ("blended_reward", "blended reward\n(0.4*triplet + 0.6*mean F1)"),
]
STAGES = [("base", "Base (no FT)"), ("sft", "SFT"), ("rl", "RL / GRPO")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "metrics"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    data = {name: json.load(open(os.path.join(args.dir, f"eval_{name}.json")))
            for name in ("base", "sft", "rl")}

    x = np.arange(len(METRICS))
    width = 0.26
    colors = {"base": "#bdbdbd", "sft": "#4c8bf5", "rl": "#f5a623"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for i, (key, label) in enumerate(STAGES):
        vals = [data[key][m] for m, _ in METRICS]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=label, color=colors[key])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([lab for _, lab in METRICS], fontsize=9)
    ax.set_ylabel("F1 / reward (val, n=300)")
    ax.set_ylim(0, 1.02)
    ax.set_title("CholecT50 triplet recognition — SFT then RL/GRPO (Thinking Machines Tinker)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    out = args.out or os.path.join(args.dir, "plots", "improvement.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

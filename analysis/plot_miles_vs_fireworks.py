#!/usr/bin/env python3
"""Overlay Miles vs Fireworks GoEmotions RL reward curves.

Fireworks reports ~25 chunks; Miles reports one point per rollout (256). The two
x-axes are not directly comparable, so both are rescaled to "% of training run".
Writes analysis/plots/miles_vs_fireworks_goemotions.png.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

FIREWORKS = os.path.join(ROOT, "fireworks", "metrics", "final_results.json")
MILES = os.path.join(ROOT, "miles", "metrics", "final_results.json")

FW_BLUE = "#2563eb"
MI_PURPLE = "#7c3aed"


def pct(xs):
    """Rescale an index list to 0..100 (% of run)."""
    n = len(xs)
    return [100.0 * i / (n - 1) for i in range(n)]


def rolling(xs, window):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


fw = json.load(open(FIREWORKS))["experiments"]["goemotions_rl"]
mi = json.load(open(MILES))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("GoEmotions RL / GRPO — Miles vs Fireworks (Qwen3-0.6B, fresh, 1 epoch)",
             fontsize=13, fontweight="bold")

# Panel 1: Jaccard (the reward). Fireworks 25 chunks vs Miles 256 rollouts.
ax1.plot(pct(fw["jaccard_curve"]), fw["jaccard_curve"],
         marker="o", ms=3, lw=1.5, color=FW_BLUE,
         label=f"Fireworks (kl=0.04) → {fw['jaccard_curve'][-1]:.3f}")
ax1.plot(pct(mi["jaccard_curve"]), mi["jaccard_curve"],
         lw=0.4, alpha=0.35, color=MI_PURPLE)
ax1.plot(pct(mi["jaccard_curve"]), rolling(mi["jaccard_curve"], 16),
         lw=1.8, color=MI_PURPLE,
         label=f"Miles (kl=0) → {mi['jaccard']['final']:.3f}")
ax1.set_title("Jaccard reward", fontsize=10, fontweight="bold")
ax1.set_xlabel("% of training run")
ax1.set_ylabel("Jaccard")
ax1.set_ylim(0, 0.75)
ax1.grid(alpha=0.3)
ax1.legend(fontsize=8)

# Panel 2: Micro-F1. Fireworks smooth curve vs Miles per-sample (rolling mean).
ax2.plot(pct(fw["micro_f1_curve"]), fw["micro_f1_curve"],
         marker="o", ms=3, lw=1.5, color=FW_BLUE,
         label=f"Fireworks → {fw['micro_f1_curve'][-1]:.3f}")
ax2.plot(pct(mi["micro_f1_curve_finish_sample"]),
         rolling(mi["micro_f1_curve_finish_sample"], 32),
         lw=1.8, color=MI_PURPLE,
         label=f"Miles (sampled, roll-32) → {mi['micro_f1']['mean_sampled']:.3f}")
ax2.set_title("Micro-F1", fontsize=10, fontweight="bold")
ax2.set_xlabel("% of training run")
ax2.set_ylabel("micro-F1")
ax2.set_ylim(0, 1.05)
ax2.grid(alpha=0.3)
ax2.legend(fontsize=8)

fig.text(0.5, 0.005,
         "x-axes rescale different step counts to % of run. Miles is noisier (one point per rollout) "
         "and ran kl_beta=0; Fireworks kl_beta=0.04.",
         ha="center", fontsize=7.5, color="#444")
fig.tight_layout(rect=(0, 0.03, 1, 0.95))
out = os.path.join(OUT, "miles_vs_fireworks_goemotions.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print("wrote", out)

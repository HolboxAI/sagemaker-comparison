#!/usr/bin/env python3
"""Plot Fireworks experiment results from the committed metrics.

Reads fireworks/metrics/{final_results.json, *_sft_metrics.jsonl} and writes
PNG charts to analysis/plots/. Re-run after another platform adds its data to
add side-by-side curves.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MET = os.path.join(ROOT, "fireworks", "metrics")
OUT = os.path.join(HERE, "plots")
os.makedirs(OUT, exist_ok=True)

BLUE = "#2563eb"
VIOLET = "#7c3aed"
GREEN = "#059669"
ORANGE = "#ea580c"


def rolling(xs, window):
    out = []
    for i in range(len(xs)):
        lo = max(0, i - window + 1)
        out.append(sum(xs[lo:i + 1]) / (i - lo + 1))
    return out


def load_sft_loss(path):
    steps, loss = [], []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if "step" in d and ("train/loss" in d or "train/ce_loss" in d):
                steps.append(d["step"])
                loss.append(d.get("train/loss", d.get("train/ce_loss")))
    return steps, loss


def sft_plot(ax, steps, loss, title, color):
    ax.plot(steps, loss, lw=0.7, alpha=0.35, color=color)
    w = max(1, len(loss) // 20)
    ax.plot(steps, rolling(loss, w), lw=1.8, color=color,
            label=f"final {loss[-1]:.3f}")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("step")
    ax.set_ylabel("cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


def rl_plot(ax, a, b, title, label_a, label_b, color_a, color_b):
    x = list(range(len(a)))
    ax.plot(x, a, marker="o", ms=3, lw=1.5, color=color_a, label=f"{label_a} → {a[-1]:.3f}")
    ax.plot(x, b, marker="o", ms=3, lw=1.5, color=color_b, label=f"{label_b} → {b[-1]:.3f}")
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("chunk")
    ax.set_ylabel("score")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)


results = json.load(open(os.path.join(MET, "final_results.json")))
E = results["experiments"]

os_steps, os_loss = load_sft_loss(os.path.join(MET, "orena_sft_metrics.jsonl"))
ge_steps, ge_loss = load_sft_loss(os.path.join(MET, "goemotions_sft_metrics.jsonl"))

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle("Fireworks — 4 experiments (fresh, 1 epoch)", fontsize=14, fontweight="bold")

sft_plot(axes[0, 0], os_steps, os_loss,
         "Exp 1 · ORena SFT (qwen3p5-9b, r64) — train loss", BLUE)
sft_plot(axes[0, 1], ge_steps, ge_loss,
         "Exp 3 · GoEmotions SFT (qwen3-0p6b, r16) — train loss", VIOLET)
rl_plot(axes[1, 0], E["orena_rl"]["score_curve"], E["orena_rl"]["exact_match_curve"],
        "Exp 2 · ORena RL / GRPO (qwen3-vl-8b-instruct)",
        "reward score", "exact_match", ORANGE, GREEN)
rl_plot(axes[1, 1], E["goemotions_rl"]["jaccard_curve"], E["goemotions_rl"]["micro_f1_curve"],
        "Exp 4 · GoEmotions RL / GRPO (qwen3-0p6b)",
        "jaccard", "micro_f1", ORANGE, GREEN)

fig.tight_layout(rect=(0, 0, 1, 0.96))
summary = os.path.join(OUT, "fireworks_summary.png")
fig.savefig(summary, dpi=150)
plt.close(fig)
print("wrote", summary)

# Individual charts
fig, ax = plt.subplots(figsize=(6, 4))
sft_plot(ax, os_steps, os_loss, "ORena SFT — train loss", BLUE)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "orena_sft_loss.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
sft_plot(ax, ge_steps, ge_loss, "GoEmotions SFT — train loss", VIOLET)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "goemotions_sft_loss.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
rl_plot(ax, E["orena_rl"]["score_curve"], E["orena_rl"]["exact_match_curve"],
        "ORena RL (GRPO) — reward", "reward score", "exact_match", ORANGE, GREEN)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "orena_rl_reward.png"), dpi=150); plt.close(fig)

fig, ax = plt.subplots(figsize=(6, 4))
rl_plot(ax, E["goemotions_rl"]["jaccard_curve"], E["goemotions_rl"]["micro_f1_curve"],
        "GoEmotions RL (GRPO) — reward", "jaccard", "micro_f1", ORANGE, GREEN)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "goemotions_rl_reward.png"), dpi=150); plt.close(fig)

print("wrote individual charts into", OUT)

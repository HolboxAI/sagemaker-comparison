#!/usr/bin/env python3
"""Build the Fireworks-vs-SageMaker observation record (.docx).

Run from the repo root. Reads analysis/plots/*.png and writes
docs/fireworks-vs-sagemaker-vs-prime-intellect.docx.
"""
import os

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS = os.path.join(ROOT, "analysis", "plots")
OUT = os.path.join(HERE, "fireworks-vs-sagemaker-vs-prime-intellect.docx")

ACCENT = RGBColor(0x1F, 0x3A, 0x5F)
GRAY = RGBColor(0x55, 0x55, 0x55)

doc = Document()
normal = doc.styles["Normal"]
normal.font.name = "Calibri"
normal.font.size = Pt(10.5)


def title(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.size = Pt(18)
    r.font.color.rgb = ACCENT
    return p


def meta(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.font.size = Pt(9)
    r.font.color.rgb = GRAY
    return p


def h1(text):
    doc.add_heading(text, level=1)


def bullet(text):
    doc.add_paragraph(text, style="List Bullet")


def table(headers, rows):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        r = hdr[i].paragraphs[0].add_run(h)
        r.bold = True
        r.font.size = Pt(9.5)
    for row in rows:
        cells = t.add_row().cells
        for i, val in enumerate(row):
            cells[i].text = str(val)
            for p in cells[i].paragraphs:
                for r in p.runs:
                    r.font.size = Pt(9)
    return t


def picture(path, width_in=6.2, caption=None):
    if os.path.exists(path):
        doc.add_picture(path, width=Inches(width_in))
        doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
        if caption:
            c = doc.add_paragraph()
            r = c.add_run(caption)
            r.italic = True
            r.font.size = Pt(8.5)
            r.font.color.rgb = GRAY
            c.alignment = WD_ALIGN_PARAGRAPH.CENTER


def placeholder(text):
    p = doc.add_paragraph()
    r = p.add_run(text)
    r.bold = True
    r.font.color.rgb = RGBColor(0xB4, 0x54, 0x09)


# ----------------------------------------------------------------------------
title("Fireworks vs SageMaker vs Prime Intellect — User Observation Record")
meta("2026-10-03 · four identical experiments · 2 SFT + 2 RL (GRPO) · three platforms")

h1("1. Steps")
table(
    ["#", "Step"],
    [
        ["1", "Set up Fireworks account + SDK (fireworks-ai 1.2.19). One API key."],
        ["2", "Listed the training API surface."],
        ["3", "Checked SFT-tunable vs RL-tunable models."],
        ["4", "Built datasets: GoEmotions (text), ORena (vision)."],
        ["5", "Uploaded datasets."],
        ["6", "Built a reward evaluator (Python) + uploaded it. Hit a build error; fixed with .tar.gz upload."],
        ["7", "Launched 4 jobs: 2 SFT + 2 RL (GRPO), 1 epoch each."],
        ["8", "Polled jobs. Collected loss + reward curves."],
        ["9", "Sent the same config to the SageMaker team."],
        ["10", "Put all results in one repo."],
    ],
)

h1("2. Fireworks — what came back")
table(
    ["Experiment", "Job", "State", "Final metric", "Cost"],
    [
        ["1. ORena SFT", "etia90sh", "COMPLETED", "train loss 0.354", "$1.47"],
        ["2. ORena RL (GRPO)", "fs-or1-orena-rl", "COMPLETED", "exact_match 0.191 → 0.282", "null*"],
        ["3. GoEmotions SFT", "ktcdqquf", "COMPLETED", "train loss 0.313", "$2.84"],
        ["4. GoEmotions RL (GRPO)", "fs-or1-ge-rl", "COMPLETED", "micro_f1 0.175 → 0.377", "null*"],
    ],
)
meta("* Managed RL returned no per-job cost. Likely free under 16B params — to confirm on the dashboard.")

bullet("Block: qwen3p5-9b is not RL-tunable → switched ORena RL to qwen3-vl-8b-instruct.")
bullet("Block: evaluator build failed (BUILD_FAILED) → upload as one .tar.gz, not loose files.")

h1("3. Results — charts")
picture(os.path.join(PLOTS, "fireworks_summary.png"), 6.5,
        "Figure 1 — all four experiments: SFT loss (top), RL reward (bottom).")
picture(os.path.join(PLOTS, "orena_rl_reward.png"), 4.6,
        "Figure 2 — ORena RL: reward score and exact_match.")
picture(os.path.join(PLOTS, "goemotions_rl_reward.png"), 4.6,
        "Figure 3 — GoEmotions RL: jaccard and micro_f1.")

h1("4. SageMaker — results")
table(
    ["Experiment", "Fireworks", "SageMaker", "Note"],
    [
        ["1. ORena SFT", "loss 0.354", "InProgress", "GPU class differs"],
        ["2. ORena RL", "exact_match 0.282", "InProgress", "base model differs"],
        ["3. GoEmotions SFT", "loss 0.313", "loss 1.54", "5k vs 43k samples, ctx 512 vs 8192"],
        ["4. GoEmotions RL", "micro_f1 0.377", "InProgress", "rollouts 8 vs 4"],
    ],
)
meta("Not like-for-like: SageMaker used smaller GPUs (T4/A10G), fewer samples, shorter context, "
     "different optimizer. 3 of 4 jobs still running.")

h1("5. Prime Intellect — results")
table(
    ["Experiment", "Model", "State", "Note"],
    [
        ["1. ORena SFT", "Qwen3.5-9B", "not started", "data pending"],
        ["2. ORena RL (GRPO)", "Qwen3.5-9B", "not started", "PI can RL the 9B (Fireworks cannot)"],
        ["3. GoEmotions SFT", "Qwen3-0.6B", "RUNNING", "loss 5.79 → 0.25 over 60 steps"],
        ["4. GoEmotions RL", "Qwen3-0.6B", "not started", "—"],
    ],
)
meta("Headless path only: CLI + API, self-run prime-rl on an on-demand A100 40GB ($1.99/hr). "
     "Wallet $48.79, $1.22 spent. No final metrics yet.")
bullet("Volatile GPU inventory — on-demand offers go stale in minutes (5 create attempts).")
bullet("Passphrase-protected SSH key broke automation (2 pod recreations).")
bullet("Config schema drift — pod image ships an older prime-rl than the docs.")
bullet("No managed dashboard for pods — pod work shows nothing in the Jobs view.")

h1("6. Why use Fireworks over SageMaker")
table(
    ["Fireworks gives", "SageMaker requires (observed)"],
    [
        ["Managed RL (GRPO) with a custom reward evaluator as code", "Write TRL GRPO yourself: IAM, instances, reward script"],
        ["Serverless fine-tuning (per-token billing)", "Provision instances (g4dn/g5)"],
        ["Train + serve on the same stack (instant deploy, KL < 0.01)", "Separate training and hosting"],
        ["Multi-LoRA on one base model", "Limited"],
        ["One API key", "IAM role + trust + policy + ECR + S3 + estimator"],
        ["B200/H200 on demand", "g4dn/g5 (T4/A10G) in our account"],
        ["4 jobs at once, no quota", "1 concurrent instance per type"],
    ],
)
bullet("Faster: all four jobs finished in under one hour.")
bullet("Less operations: no GPU, no container, no IAM to manage.")
bullet("For a company like Cursor: focus engineers on the model, not on infrastructure.")
bullet("Prime Intellect sits between: raw GPU pods for control, but self-run — no managed RL.")

h1("7. Screenshots to add")
placeholder("[ADD] Fireworks dashboard — the four jobs and metrics.")
placeholder("[ADD] Fireworks model catalog — SFT-tunable vs RL-tunable.")
placeholder("[ADD] SageMaker console — training jobs + quota error.")
placeholder("[ADD] Prime Intellect — pod create flow / empty GPU list.")
placeholder("[ADD] Fireworks pricing — per-token training / RL.")
placeholder("[ADD] SageMaker pricing — g4dn/g5 vs B200/H200 rates.")

h1("Sources")
for url in [
    "https://fireworks.ai/",
    "https://fireworks.ai/training/rl-rollouts",
    "https://fireworks.ai/blog/train-past-the-frontier-training-api-now-generally-available",
    "https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models",
    "https://fireworks.ai/blog/aws-sagemaker",
    "https://stack.tools/tool/fireworks-ai",
    "https://primeintellect.ai/",
    "https://docs.primeintellect.ai/",
]:
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(url)
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

doc.save(OUT)
print("wrote", OUT)

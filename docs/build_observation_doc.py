#!/usr/bin/env python3
"""Build the fine-tuning and RL observation record (.docx).

Run from the repo root. Reads analysis/plots/*.png and docs/images/*.png,
writes docs/ft-rl-platform-observation-record.docx.
"""
import os

from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
PLOTS = os.path.join(ROOT, "analysis", "plots")
IMAGES = os.path.join(HERE, "images")
OUT = os.path.join(HERE, "ft-rl-platform-observation-record.docx")

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


def h2(text):
    doc.add_heading(text, level=2)


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


def shot(name, caption, width_in=6.2):
    picture(os.path.join(IMAGES, name), width_in, caption)


# ----------------------------------------------------------------------------
title("Fine-tuning and RL observation record")
meta("Fireworks, SageMaker, Prime Intellect  |  2026-10-03  |  "
     "ORena (vision) + GoEmotions (text)  |  2 SFT + 2 RL (GRPO) per platform")

# ---- Fireworks -------------------------------------------------------------
h1("1. Fireworks")

h2("Steps")
table(
    ["#", "Step"],
    [
        ["1", "Set up a Fireworks account and SDK (fireworks-ai 1.2.19). One API key."],
        ["2", "Listed the training API: SFT, DPO, RL jobs, datasets, evaluators, deployments."],
        ["3", "Checked which models are SFT-tunable and RL-tunable."],
        ["4", "Built datasets: GoEmotions (text), ORena (vision)."],
        ["5", "Uploaded datasets."],
        ["6", "Built a reward evaluator in Python. The upload failed. Fixed it with one .tar.gz file."],
        ["7", "Launched 4 jobs: 2 SFT + 2 RL (GRPO), 1 epoch each."],
        ["8", "Polled the jobs. Collected loss and reward curves."],
    ],
)
shot("run_brief.png", "The brief that started the run.", 6.5)

h2("Results")
table(
    ["Experiment", "Model", "Final metric", "Cost", "Wall"],
    [
        ["ORena SFT", "qwen3p5-9b", "train loss 0.354", "$1.47", "28 min"],
        ["ORena RL (GRPO)", "qwen3-vl-8b-instruct", "exact_match 0.191 to 0.282", "null", "37 min"],
        ["GoEmotions SFT", "qwen3-0.6b", "train loss 0.313", "$2.84", "50 min"],
        ["GoEmotions RL (GRPO)", "qwen3-0.6b", "micro_f1 0.175 to 0.377", "null", "not logged"],
    ],
)
meta("RL cost is null. Managed RL (GRPO) returned no per-job cost on the Fireworks side.")

h2("Screenshots")
shot("fireworks_orena_sft_job.png", "ORena SFT job page. Base model qwen3p5-9b. Estimated cost $1.47.")
shot("fireworks_goemotions_sft_job.png", "GoEmotions SFT job page. Base model qwen3-0.6b. Estimated cost $2.84.")
shot("fireworks_orena_rl_job.png", "ORena RL job page. Base model qwen3-vl-8b-instruct. Scores: exact_match, excluded_class, no_hallucinated_class.")
shot("fireworks_goemotions_rl_job.png", "GoEmotions RL job page. Reward evaluator fs-or1-goemotions-reward.")
shot("fireworks_goemotions_rl_scores.png", "GoEmotions RL scores. jaccard and micro_f1. 1,600 successful rollouts.")
picture(os.path.join(PLOTS, "fireworks_summary.png"), 6.5,
        "All four Fireworks jobs. SFT loss on top. RL reward below.")
picture(os.path.join(PLOTS, "orena_rl_reward.png"), 4.6,
        "ORena RL: reward score and exact_match.")
picture(os.path.join(PLOTS, "goemotions_rl_reward.png"), 4.6,
        "GoEmotions RL: jaccard and micro_f1.")

h2("Observations")
bullet("qwen3p5-9b is not RL-tunable. ORena RL switched to qwen3-vl-8b-instruct.")
bullet("The evaluator build failed (BUILD_FAILED). It must be one .tar.gz file, not loose files.")
bullet("All four jobs finished in under one hour.")
bullet("Managed RL returned no per-job cost.")

# ---- SageMaker -------------------------------------------------------------
h1("2. SageMaker")

h2("Steps")
table(
    ["#", "Step"],
    [
        ["1", "Wrote an IAM execution role (trust = sagemaker.amazonaws.com) and a scoped policy."],
        ["2", "Prepared S3 data: ORena frame files + parquet index, GoEmotions JSONL."],
        ["3", "Wrote 4 script-mode entry points (PyTorch 2.5.1, transformers 5.18.0, trl 1.14.1)."],
        ["4", "Launched 4 estimator jobs on g4dn/g5 (T4/A10G)."],
        ["5", "Polled to completion. ORena SFT needed 4 attempts."],
        ["6", "Collected final metrics and raw logs."],
    ],
)

h2("Results")
table(
    ["Experiment", "Model", "Instance", "Final metric", "Wall", "Cost"],
    [
        ["ORena SFT", "Qwen3.5-9B", "g5.12xlarge", "train loss 1.494", "787 s", "$1.24"],
        ["ORena RL (GRPO)", "Qwen3.5-9B", "g5.2xlarge", "score 0.0", "1747 s", "$0.59"],
        ["GoEmotions SFT", "Qwen3-0.6B", "g4dn.xlarge", "train loss 1.54", "1174 s", "$0.17"],
        ["GoEmotions RL (GRPO)", "Qwen3-0.6B", "g5.xlarge", "jaccard 0.0", "1605 s", "$0.45"],
    ],
)
meta("All four jobs completed. Dataset caps were 256 to 5000 samples (demo budget), "
     "not the full set Fireworks used.")

h2("Observations")
bullet("ORena SFT 9B bf16 did not fit one 24 GB A10G. Three failed attempts. "
       "It fit after LoRA rank 64 to 32 and expandable_segments.")
bullet("Both RL jobs completed with reward 0. Qwen3/3.5 thinking mode exhausted "
       "max_completion_length. Fireworks disabled thinking. This launch did not.")
bullet("One concurrent training instance per instance type (account quota).")
bullet("Script mode: no Docker and no ECR. Dependencies install at job start.")

# ---- Prime Intellect -------------------------------------------------------
h1("3. Prime Intellect")

h2("Steps")
table(
    ["#", "Step"],
    [
        ["1", "Created a Prime Intellect account and wallet ($50)."],
        ["2", "Installed the prime CLI (0.9.1) and used the REST API."],
        ["3", "Tried hosted training. LoRA SFT is gated. Full fine-tune showed an empty GPU list."],
        ["4", "Switched to on-demand pods: create an A100 40GB, ssh in, run prime-rl."],
        ["5", "Fetched GoEmotions from Hugging Face (800 train / 200 dev)."],
        ["6", "Ran a GoEmotions SFT smoke job. Loss 5.79 to 0.25 over 60 steps."],
        ["7", "Paused. ORena data and the RL jobs are pending."],
    ],
)

h2("Results")
table(
    ["Experiment", "Model", "State", "Note"],
    [
        ["ORena SFT", "Qwen3.5-9B", "not started", "data pending"],
        ["ORena RL (GRPO)", "Qwen3.5-9B", "not started", "PI can RL the 9B directly"],
        ["GoEmotions SFT", "Qwen3-0.6B", "running", "loss 5.79 to 0.25 over 60 steps"],
        ["GoEmotions RL (GRPO)", "Qwen3-0.6B", "not started", "pending"],
    ],
)
meta("On-demand A100 40GB at $1.99/hr. Wallet $48.79, $1.22 spent. No final metrics yet.")

h2("Observations")
bullet("Volatile GPU inventory. On-demand offers go stale in minutes. Five create attempts.")
bullet("A passphrase-protected SSH key broke automation. Two pod recreations.")
bullet("Config schema drift. The pod image ships an older prime-rl than the docs.")
bullet("No dashboard for pods. Pod work shows nothing in the Jobs view.")

# ---- Repo ------------------------------------------------------------------
h1("4. Repo record")
shot("repo_commit.png", "Final Fireworks results committed to the repo.", 6.5)
bullet("One repo holds all three platforms: fireworks/, sagemaker/, prime-intellect/.")
bullet("Each platform folder holds config, scripts, logs, and metrics in one schema.")

h1("Sources")
for url in [
    "https://fireworks.ai/",
    "https://fireworks.ai/training/rl-rollouts",
    "https://docs.fireworks.ai/fine-tuning/reinforcement-fine-tuning-models",
    "https://aws.amazon.com/sagemaker/",
    "https://primeintellect.ai/",
    "https://docs.primeintellect.ai/",
]:
    p = doc.add_paragraph(style="List Bullet")
    r = p.add_run(url)
    r.font.size = Pt(8)
    r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)

doc.save(OUT)
print("wrote", OUT)

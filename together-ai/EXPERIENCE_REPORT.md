# Together AI — experience report (feasibility check)

**Date:** 2026-10-12
**Status:** not viable. Checked the platform; nothing was trained or billed.

## 1. The task

Same CholecT50 surgical action-triplet task as the Baseten leg: SFT a VLM, then
**RLVR** (GRPO + custom triplet-F1 reward) to show a before/after F1 improvement.

## 2. Problems we hit (in order)

| # | What we wanted | What Together offers | Result |
|---|---|---|---|
| 1 | Run RL/GRPO with a custom reward | self-serve API = SFT + DPO only | ❌ RL unavailable |
| 2 | Fine-tune `Qwen3-VL-4B-Instruct` | model not in the catalog | ❌ would need Qwen3.5-4B/9B |
| 3 | Use the advertised "RL (GRPO)" | gated "Custom Training" beta, "Request access", no pricing, no VLM mention | ❌ not self-serve |

## 3. Root cause

Together's self-serve fine-tuning product is **alignment-only (SFT + DPO)**. It does
not expose reinforcement learning, so there is no surface for a reward function at
all. Their RL offering is a separate, sales-gated beta that (a) is not reachable
with an API key and (b) does not document vision-language support.

## 4. Cost

**Cannot be quoted.** The RL stage — the entire purpose of the demo — cannot run on
this platform. (SFT on a supported VLM would be the only thing billable, and it
doesn't achieve the goal.)

## 5. Recommendation

Skip Together for this project. Two viable paths:

1. **AWS SageMaker** — already proven for RL/GRPO in earlier runs.
2. **Baseten** — pipeline already built; only needs the org's Training Jobs
   entitlement enabled.

`scripts/probe_together.py` reproduces the checks that led to this conclusion.

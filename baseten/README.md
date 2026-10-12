# Baseten — run data (blocked at authorization)

This folder documents the attempt to run a **CholecT50 surgical action-triplet
recognition** pipeline (SFT → RLVR) on **Baseten Training Jobs**. The pipeline is
fully built and push-ready, but the run was **blocked before launch** by an
authorization error (HTTP 403) — not a code problem.

## The task

| | |
|---|---|
| Dataset | CholecT50 — endoscopic surgery frames |
| Task | list every `instrument-verb-target` action triplet visible in a frame |
| Model | `Qwen/Qwen3-VL-4B-Instruct` (vision-language, LoRA) |
| Stage 1 | SFT (supervised fine-tuning) |
| Stage 2 | RLVR — GRPO with a custom triplet-F1 reward |
| Success metric | before/after triplet F1 on a fixed test subset |

## Status

| # | Experiment | State |
|---|---|---|
| 1 | CholecT50 SFT | 🚫 blocked (HTTP 403 — training not authorized) |
| 2 | CholecT50 RL (GRPO / RLVR) | 🚫 blocked (same) |

## Contents

| Path | What it holds |
|---|---|
| `config/config_sft.py`, `config/config_rl.py` | exact `truss_train` launch configs |
| `config/BASETEN_FINDINGS.md` | the config, the reward, and the blocker |
| `scripts/` | training entry points, reward, data pipeline, eval |
| `logs/00_push_403.txt` | the captured 403 error |
| `metrics/` | `final_results.json` + `run_manifest.json` (blocked states) |
| `EXPERIENCE_REPORT.md` | the problems we hit, root cause, and next step |

## Key facts

- Training on Baseten is offered **only** on H100 ($6.50/hr) and H200 ($7.50/hr);
  there is no smaller/cheaper GPU for training.
- The push was rejected with **HTTP 403 "You are not authorized for Baseten
  training"** — an org-level entitlement, not fixable from the CLI.
- Adding a payment method and a Hugging Face token did **not** fix it.

## Reproduce (once the org is authorized)

```bash
baseten train push --config ./config/config_sft.py
baseten train push --config ./config/config_rl.py   # loads the SFT checkpoint
```

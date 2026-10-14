# Together AI — feasibility assessment (not viable for RL)

This folder documents a **feasibility check** of running the CholecT50 surgical
action-triplet pipeline (SFT → RLVR) on **Together AI**. Conclusion: the RL stage —
the whole point of the demo — is **not available** on Together's self-serve API.

## The task

| | |
|---|---|
| Dataset | CholecT50 — endoscopic surgery frames |
| Task | list every `instrument-verb-target` action triplet visible in a frame |
| Model | `Qwen/Qwen3-VL-4B-Instruct` (vision-language) |
| Stage 1 | SFT |
| Stage 2 | RLVR — GRPO with a custom triplet-F1 reward |

## Verdict

| Requirement | Together AI |
|---|---|
| SFT a vision model | ✅ possible (Qwen3.5-4B/9B, not Qwen3-VL-4B) |
| RL with a custom reward (RLVR / GRPO) | ❌ not offered on the self-serve API |
| SFT → RL → before/after F1 demo | ❌ the RL stage is impossible here |

## Status

| # | Experiment | State |
|---|---|---|
| 1 | CholecT50 SFT | ⚠️ possible only on a different model (Qwen3.5-4B/9B) |
| 2 | CholecT50 RL (GRPO / RLVR) | 🚫 not feasible — no RL/GRPO |

## Contents

| Path | What it holds |
|---|---|
| `config/TOGETHER_FINDINGS.md` | what the API supports, and what's missing |
| `scripts/probe_together.py` | the API probe that produced these findings |
| `logs/00_api_probe.txt` | condensed probe output |
| `metrics/` | `final_results.json` + `run_manifest.json` |
| `EXPERIENCE_REPORT.md` | the problems + the recommendation |

## Key facts

- Together's self-serve fine-tuning supports **SFT and DPO only** — no GRPO, no
  RLVR, no way to supply a custom reward function.
- `Qwen/Qwen3-VL-4B-Instruct` is **not** on Together (inference or fine-tuning).
- Their "RL (GRPO)" exists only in a gated **Custom Training** beta
  ("Request access"), with no self-serve access and no published pricing.
- **No cost can be quoted**, because the RL stage cannot run here at all.

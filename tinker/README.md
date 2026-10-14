# Thinking Machines Tinker — successful SFT → RL/GRPO run

This folder documents the **first successful end-to-end run** of the CholecT50
surgical action-triplet pipeline (SFT → RLVR) in this benchmark — on **Thinking
Machines Tinker**. The same task was probed as *not feasible* on Baseten and
Together AI; Tinker ran both stages with a custom reward and delivered a clean
before/after F1 improvement.

## The task

| | |
|---|---|
| Dataset | CholecT50 — endoscopic surgery frames |
| Task | list every `instrument-verb-target` action triplet visible in a frame |
| Model | `Qwen/Qwen3.5-4B` (vision-language, LoRA) |
| Stage 1 | SFT (cross-entropy) |
| Stage 2 | RLVR — GRPO with a custom triplet-F1 reward |

## Verdict

| Requirement | Tinker |
|---|---|
| SFT a vision model (LoRA) | ✅ |
| RL with a custom reward (RLVR / GRPO) | ✅ |
| SFT → RL → before/after F1 demo | ✅ full pipeline ran |

## Before / After (val, n=300 frames, greedy decoding)

| Metric | Base (no FT) | SFT | RL / GRPO | RL vs SFT |
|---|---|---|---|---|
| triplet F1 | 0.000 | 0.136 | **0.159** | +17% |
| instrument F1 | 0.000 | 0.847 | **0.864** | +2.0% |
| verb F1 | 0.000 | 0.396 | **0.528** | +33% |
| target F1 | 0.000 | 0.306 | **0.383** | +25% |
| blended reward | 0.000 | 0.364 | **0.419** | +15% |

`blended reward = 0.4 * triplet_F1 + 0.6 * mean(instrument, verb, target F1)`

The base model cannot emit the triplet format at all (all-zero F1), so this is a
clean "0 → X" demo: SFT teaches the task from scratch, RL lifts every component
further.

## Contents

| Path | What it holds |
|---|---|
| `config/TINKER_CONFIG.md` | every hyperparameter for both stages |
| `scripts/tinker_sft.py` | SFT loop (Tinker low-level `forward_backward` + `optim_step`) |
| `scripts/tinker_rl.py` | GRPO/RLVR run (`tinker_cookbook.rl.train` + custom Env) |
| `scripts/tinker_eval.py` | F1 evaluation of base / SFT / RL checkpoints |
| `scripts/cholec_data.py`, `scripts/reward.py` | data pipeline + triplet-F1 reward (shared) |
| `scripts/plot_progress.py`, `scripts/make_improvement_chart.py` | curve + before/after charts |
| `logs/` | full console logs for SFT, RL, and both evals |
| `metrics/` | `final_results.json` + `run_manifest.json` + raw JSONL/JSON + PNGs |

## Key facts

- Tinker exposes a **low-level training loop** and a **`tinker_cookbook` RL
  harness**, so the custom triplet-F1 reward runs as an ordinary Python `Env` —
  no platform gap for RLVR.
- Training and inference use **two non-interchangeable checkpoint kinds**
  (`weights/` vs `sampler_weights/`) — see the experience report.
- No per-job USD cost is surfaced by the SDK used; `estimated_cost_usd` is
  `null` (see `findings`).

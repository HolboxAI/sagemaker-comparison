# Thinking Machines Tinker — experience report

**Date:** 2026-10-13
**Status:** success. Full SFT → RL/GRPO pipeline ran; see `metrics/final_results.json`.

## 1. The task

CholecT50 surgical action-triplet recognition: SFT `Qwen/Qwen3.5-4B` (VLM, LoRA)
to emit `instrument-verb-target` triplets, then **RLVR** (GRPO + custom triplet-F1
reward) to lift the before/after F1.

## 2. What worked

Tinker exposes exactly the two surfaces this task needs, so the demo ran with no
platform gap:

| Need | Tinker surface |
|---|---|
| SFT a VLM | low-level `create_lora_training_client` + `forward_backward` + `optim_step` |
| RL with a custom reward | `tinker_cookbook.rl.train`; the triplet-F1 reward is a plain Python `Env` |

The training loop is written locally but executes on managed GPUs; the local
process only ever holds the tokenizer, image processor, and one micro-batch.

## 3. Problems we hit (and fixed)

| # | Problem | Cause | Fix |
|---|---|---|---|
| 1 | SFT eval 400: "must point to a sampler weights checkpoint" | `save_state()` → `weights/` (training-only) is not loadable by a sampling client | load the `weights/` checkpoint into a training client, then `save_weights_for_sampler()` → `sampler_weights/` |
| 2 | RL refused to start from the SFT checkpoint | `create_training_client_from_state` only accepts `weights/` (not `sampler_weights/`) | save **both** kinds: `save_state` for resume/RL-init, `save_weights_for_sampler` for eval |
| 3 | Final checkpoint name collision ("already exists") | the in-loop save reused the last step's auto name | give the final save an explicit name (`cholec-sft-final`) |
| 4 | Orchestration host killed under memory pressure | a thin host cannot hold the dataset + tokenizer in RAM | build datums one micro-batch at a time; keep only ~1 batch resident |

The root cause of #1–#2 is worth calling out: **`weights/` and `sampler_weights/`
are non-interchangeable.** A checkpoint is either a *training* state (resumable,
RL-initable) or a *sampling* snapshot (inference-only), and the wrong client
rejects the wrong kind. Saving both after training sidesteps the whole class of
error.

## 4. Cost

`estimated_cost_usd` is `null`: the SDK used does not surface a per-job USD cost.
Only token counts (`SFT` ≈ 1.52M tokens) and wall time (`SFT` ≈ 55 min, `RL` ≈
7 min) were available.

## 5. Recommendation

Tinker is the right platform for this demo — it is the only leg of this benchmark
so far that ran the full CholecT50 SFT → RLVR pipeline with a custom reward.
Reuse `scripts/` as-is; the two things to remember are (a) save both checkpoint
kinds, and (b) keep the orchestrator's memory footprint to one batch.

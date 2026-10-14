# Baseten — CholecT50 SFT → RLVR config + findings

**Date:** 2026-10-12
**Task:** surgical action-triplet recognition on CholecT50 (see `../README.md`).
**Status:** 🚫 blocked at authorization (HTTP 403). No training step ran.

## 1. What we built

A two-job pipeline in Baseten's `truss_train` config format:

1. **SFT job** (`config_sft.py`) — LoRA-finetune `Qwen/Qwen3-VL-4B-Instruct` on the
   2,500-frame train split, merge the adapter, and save the full model to
   `$BT_CHECKPOINT_DIR`.
2. **RL job** (`config_rl.py`) — load the SFT checkpoint from
   `$BT_LOAD_CHECKPOINT_DIR/rank-0`, apply a fresh LoRA, and run **GRPO** with a
   custom triplet-F1 reward (RLVR), then eval before/after.

The checkpoint handoff uses `LoadCheckpointConfig` +
`BasetenCheckpoint.from_latest_checkpoint(...)`.

## 2. Config summary

| | SFT | RL (GRPO) |
|---|---|---|
| Base image | `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-runtime` | same |
| GPU | 1× H100 | 1× H100 |
| HF token | `SecretReference(name="hf_access_token")` | same |
| learning rate | 2e-4 | 1e-6 |
| schedule | 1 epoch | 150 steps |
| LoRA | r16 / a32 | r16 / a32 |
| KL beta / temperature | — | 0.04 / 1.0 |
| rollouts per prompt | — | 8 |
| max_seq / prompt | 8192 | 8192 prompt / 256 completion |

## 3. The reward (RLVR)

`../scripts/reward.py` scores a completion against ground truth as:

```
reward = 0.4 * triplet_F1 + 0.6 * mean(instrument_F1, verb_F1, target_F1)
```

Component F1 gives partial credit when only part of a triplet is right, so the
reward is non-zero on most parseable completions (avoids the "reward always 0"
trap that hit the earlier ORena/GoEmotions thinking-mode runs). Unit tests in the
file confirm exact = 1.0 and malformed = 0.0.

## 4. The blocker

`baseten train push` returns:

```
HTTP 403 — "You are not authorized for Baseten training"
```

on `POST /v1/teams/<team-id>/training_projects`.

**What we tried (none fixed it):**

1. Re-authenticated the CLI and re-pushed — same 403.
2. Added a payment method to the account — same 403.
3. Added a Hugging Face token as a secret — same 403.

**Root cause:** Training Jobs is an **org-level entitlement**. The account's org
does not have it enabled, so every training push is rejected at the API before any
job is created. This is not something the CLI or a config can change.

## 5. Findings

- Baseten's training surface requires an **org training entitlement**; the error is
  a bare 403 with no guidance, which is easy to misdiagnose as a token/config issue.
- **GPU selection is narrow:** only H100 ($6.50/hr) and H200 ($7.50/hr) are offered
  for training — no smaller GPU for a cheap smoke test. Spot is available via
  `--spot`.
- The `truss_train` config API (`Runtime`, `TrainingJob`, `TrainingProject`,
  `LoadCheckpointConfig`, `SecretReference`) was validated against the installed CLI
  and compiles cleanly; the code is push-ready.

## 6. Next step

Contact Baseten to enable Training Jobs for the org, then `baseten train push`
both configs. The pipeline itself needs no changes.

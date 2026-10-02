# SageMaker — 4-Experiment Setup Config

_Compiled 2026-10-02. This is the SageMaker counterpart to
[`fireworks/config/FIREWORKS_4EXP_CONFIG.md`](../../fireworks/config/FIREWORKS_4EXP_CONFIG.md).
It records the exact launch config used for the four runs, the SageMaker-specific
wiring (IAM role, script-mode estimator, S3 channels), and — per the repo rule —
every place SageMaker forced a different value, as a finding rather than silently._

---

## 1. Experiment matrix

| # | Experiment | Dataset | Method | Base model (HF) | SageMaker instance | Entry script |
|---|---|---|---|---|---|---|
| 1 | ORena VLM SFT | ORena train | SFT (LoRA) | `Qwen/Qwen3.5-9B` | `ml.g5.12xlarge` (4× A10G) | `scripts/src/orena_sft.py` |
| 2 | ORena VLM RL | ORena train | GRPO | `Qwen/Qwen3.5-9B` | `ml.g5.2xlarge` (A10G) | `scripts/src/orena_rl.py` |
| 3 | GoEmotions SFT | GoEmotions train | SFT (LoRA) | `Qwen/Qwen3-0.6B` | `ml.g4dn.xlarge` (T4) | `scripts/src/goemotions_sft.py` |
| 4 | GoEmotions RL | GoEmotions train | GRPO | `Qwen/Qwen3-0.6B` | `ml.g5.xlarge` (A10G) | `scripts/src/goemotions_rl.py` |

Rules applied to all four:
- **Fresh runs** — no warm-start / resume.
- **RL method = GRPO** — TRL `GRPOTrainer` / `GRPOConfig`.
- **1 epoch** everywhere (SFT and RL); RL also capped by `max_steps` / `max_samples`
  for a short budget-friendly demo run.
- **Script mode** — PyTorch estimator (framework `2.5.1`, Python 3.11), no Docker/ECR;
  the `scripts/src/` files are uploaded as `source_dir` and run as entry points.

---

## 2. Shared environment

| Item | Value |
|---|---|
| Account / region | `751871643798` · `us-east-1` (AWS profile `stanford_gpu`) |
| Auth | IAM execution role `sagemaker-stanford-exec` (trust = `sagemaker.amazonaws.com`) |
| SDK | `sagemaker` 2.257.6 (PyTorch estimator) + `boto3` |
| Runtime image | SageMaker PyTorch 2.5.1 / py311 |
| Python deps (installed at job start from `requirements.txt`) | `transformers==5.18.0`, `trl==1.14.1`, `peft==0.21.2`, `accelerate>=1.4.0`, `datasets>=4.7.0` |
| Data bucket | `s3://stanford-train-data` |

**Version pins matter:** `transformers==5.18.0` is required for the `qwen3_5`
architecture used by `Qwen/Qwen3.5-9B` (4.x does not recognize it) and it uses
`dtype=` (not the removed `torch_dtype=`). `trl==1.14.1` uses `SFTConfig.max_length`
(not `max_seq_length`) and its GRPO reward contract passes completions as
`list[list[{"role","content"}]]` and expects a flat `list[float]` reward.

---

## 3. Data

Data is read from S3 channels (script-mode `inputs=`), not from the canonical
`../data/` files. This is the largest divergence and is recorded in §8.

### 3.1 ORena (surgical VQA — vision + text)

- **Frames** (not base64 inline): `s3://stanford-train-data/curated_frames/` —
  JPEG frames in `First_/Second_Test/Train_Frames/<video>/frame_HHMMSS.jpg`.
- **SFT**: `s3://stanford-train-data/sft/train.jsonl` — CHAT JSONL with
  `messages` + an `images` array of `<image>`-tagged frame paths.
- **RL**: `s3://stanford-train-data/rlvr/frames_train_rl.parquet` — columns
  `video`, `timestamp_start`, `question`, `answer`, `answer_format`.
- Answer space (5 formats): `fo_class`, `number`, `open_ended`, `binary`,
  `multiple_choice`. Foreign-object class list in `scripts/src/fo_defs.txt`.

### 3.2 GoEmotions (text classification — multi-label)

- **SFT**: `s3://stanford-train-data/goemotions/train_sft.jsonl` — CHAT JSONL
  (system + user text + assistant comma-separated label list).
- **RL**: `s3://stanford-train-data/goemotions/train_rl.jsonl` — RL JSONL with
  top-level `ground_truth.labels`.
- 28 labels (27 emotions + neutral), multi-label.

---

## 4. Experiment 1 — ORena VLM SFT

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-9B` (AutoModelForImageTextToText + AutoProcessor) |
| Instance | `ml.g5.12xlarge`, 200 GB volume |
| Epochs | 1 |
| Learning rate | 1e-4 |
| LoRA rank / alpha / dropout | 64 / 128 / 0.05 (`target_modules="all-linear"`) |
| Batch size | micro 1 / effective 4 |
| Max length | 2048 |
| Max train samples | 256 (launch hyperparameter) |
| Optimizer | TRL default (AdamW), `gradient_checkpointing=True`, `bf16=True` |
| Channels | `train` = `s3://…/sft/train.jsonl`, `frames` = `s3://…/curated_frames/` |

VLM note: `orena_sft.py` keeps message content as plain strings and passes images
via a separate `images` column; trl's `DataCollatorForVisionLanguageModeling` →
`prepare_multimodal_messages` injects the image before the first user turn.

## 5. Experiment 2 — ORena VLM RL (GRPO)

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3.5-9B` ⚠️ (Fireworks cannot RL this model — see §8) |
| Instance | `ml.g5.2xlarge`, 200 GB volume |
| Epochs | 1 (bounded by `max_steps`) |
| Learning rate | 1e-6 |
| LoRA rank / alpha | 16 / 32 |
| KL beta / temperature | 0.04 / 0.7 |
| Max output tokens | 128 |
| Generations per prompt | 4 |
| Batch size | micro 1 / grad-accum 4 (generation batch 4) |
| Max steps / max samples | 50 / 256 |
| Reward | `scripts/src/orena_reward.py` (per-format verifiable scorer) |
| Channels | `data` = `frames_train_rl.parquet`, `frames` = `curated_frames/` |

## 6. Experiment 3 — GoEmotions SFT

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` (AutoModelForCausalLM) |
| Instance | `ml.g4dn.xlarge`, 100 GB volume |
| Epochs | 1 |
| Learning rate | 1e-4 |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| Batch size | micro 8 / effective 64 |
| Max length | 512 |
| Max train samples | 5000 |
| Formatting | `tokenizer.apply_chat_template(messages, tokenize=False)` |
| Channel | `train` = `s3://…/goemotions/train_sft.jsonl` |

## 7. Experiment 4 — GoEmotions RL (GRPO)

| Field | Value |
|---|---|
| Base model | `Qwen/Qwen3-0.6B` |
| Instance | `ml.g5.xlarge`, 100 GB volume |
| Epochs | 1 |
| Learning rate | 1e-5 |
| LoRA rank / alpha | 16 / 32 |
| KL beta / temperature | 0.04 / 0.7 |
| Max output tokens | 64 |
| Generations per prompt | 4 |
| Batch size | micro 2 / grad-accum 2 (generation batch 4) |
| Max train samples | 256 |
| Reward | inline Jaccard of predicted vs ground-truth label set (micro-F1 tracked) |
| Channel | `train` = `s3://…/goemotions/train_rl.jsonl` |

---

## 8. Findings that change the comparison

Recorded rather than silently applied, per the repo rules.

1. **GPU class mismatch.** Fireworks ran on dedicated B200/H200 shapes; SageMaker
   ran on on-demand `g4dn`/`g5` (T4 / A10G, 16–96 GB). This forced smaller
   batches, shorter contexts, and fewer rollouts (below).

2. **Hyperparameter deltas forced by the smaller GPUs:**

   | Experiment | Field | Fireworks | SageMaker |
   |---|---|---|---|
   | 1 ORena SFT | batch (micro/effective) | 4 / 32 | 1 / 4 |
   | 1 ORena SFT | max context | 65536 | 2048 |
   | 2 ORena RL | learning rate | 1e-5 | 1e-6 |
   | 2 ORena RL | max output tokens | 512 | 128 |
   | 2 ORena RL | rollouts/prompt | 8 | 4 |
   | 3 GoEmotions SFT | max context | 8192 | 512 |
   | 4 GoEmotions RL | rollouts/prompt | 8 | 4 |

3. **Rollouts reduced 8 → 4.** trl `GRPOTrainer` requires the generation batch
   size (`per_device_train_batch_size × gradient_accumulation_steps`) to be
   divisible by `num_generations`; with the batch sizes the A10G/T4 could fit,
   8 rollouts did not divide evenly, so 4 was used.

4. **ORena RL base model differs.** SageMaker ran GRPO on `Qwen/Qwen3.5-9B`
   directly; Fireworks can only RL `qwen3-vl-8b-instruct`. This is the known
   feature gap the SageMaker README flags.

5. **Data source & format diverge.** SageMaker read data from S3
   (`s3://stanford-train-data/…`), not the byte-identical canonical `../data/`
   files, and ORena uses **frame-file paths** (`curated_frames/` JPEGs + a
   parquet index) instead of Fireworks' base64-inline JSONL. The comparison of
   loss/score is still meaningful (same task, same answer space) but the input
   bytes are not identical, so strict per-sample diffs are not valid.

6. **Optimizer.** SageMaker used the TRL default AdamW; Fireworks used AdamW
   8-bit with cosine decay + 10% warmup.

7. **ORena reward semantics differ.** Fireworks' reward is exact-match after
   normalization. The SageMaker `orena_reward.py` gives partial credit: `fo_class`
   is micro-F1 over parsed class sets, `open_ended` is token-F1, `number`/`binary`/
   `multiple_choice` are exact. Reward magnitudes are therefore not directly
   comparable on ORena. GoEmotions reward (Jaccard) matches Fireworks §8.2.

8. **RL runs complete but do not learn (reward = 0).** Both GRPO jobs reached
   `RL_DONE`, but Qwen3/Qwen3.5 "thinking" mode made every completion run to
   `max_completion_length` (`clipped_ratio=1`), so reward stayed `0` with `kl≈0`.
   Fireworks disabled thinking (`enable_thinking: False`); the SageMaker launch did
   not, so the two RL runs have no meaningful score as launched.

---

## 9. IAM / execution-role setup

`scripts/iam/` holds the role the jobs assume:

- `execution-role-trust.json` — trust policy for `sagemaker.amazonaws.com`.
- `execution-role-policy.json` — inline policy scoped to `s3://stanford-train-data`,
  `sagemaker:Create/Describe/ListTrainingJob`, ECR image pull, and CloudWatch logs.
- `create_role.sh` — idempotent create + attach (run once, locally).

```bash
AWS_PROFILE=stanford_gpu bash scripts/iam/create_role.sh
export ROLE_ARN=arn:aws:iam::751871643798:role/sagemaker-stanford-exec
```

## 10. Concurrency quota (SageMaker-specific)

The account allows **1 concurrent training instance per instance type** (not per
family). The four jobs each use a distinct type precisely so they can run in
parallel: `ml.g4dn.xlarge`, `ml.g5.xlarge`, `ml.g5.2xlarge`, `ml.g5.12xlarge`.
A second job on an occupied type fails with `ResourceLimitExceeded`; stale jobs
keep the quota until their async `Stopping` phase completes.

---

## 11. Run status (launched 2026-10-02)

| # | SageMaker job name | Instance | State |
|---|---|---|---|
| 1 | `orena-sft-2026-10-02-17-58-46-825` | `ml.g5.12xlarge` | InProgress (3rd attempt; single-GPU env now applied) |
| 2 | `orena-rl-2026-10-02-17-18-25-972` | `ml.g5.2xlarge` | Completed (RL_DONE; reward=0 — §8.8) |
| 3 | `goemotions-sft-2026-10-02-16-07-29-859` | `ml.g4dn.xlarge` | Completed (loss 1.54) |
| 4 | `goemotions-rl-2026-10-02-16-30-25-068` | `ml.g5.xlarge` | Completed (RL_DONE; reward=0 — §8.8) |

Final metrics land in [`metrics/final_results.json`](../metrics/final_results.json)
in the exact schema from [`analysis/schema.md`](../../analysis/schema.md), once all
jobs reach a terminal state.

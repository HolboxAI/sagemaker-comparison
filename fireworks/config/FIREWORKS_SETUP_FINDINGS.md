# Fireworks Phase — Setup Findings

_Verified 2026-10-02. All raw output is in `fireworks_vs_sagemaker/logs/`._

## 1. Credentials and account

- `FIREWORKS_API_KEY` stored in `fireworks_vs_sagemaker/.env` (git-ignored).
- Key is valid for **inference** and **training** (all training resource lists
  return 200).
- Account: **purchase@holbox.ai** · account_id `purchase-j087ceepz9i`.
- SDK: `fireworks-ai 1.2.19` in `fireworks_vs_sagemaker/.venv`.

## 2. Training access — confirmed live

The SDK exposes and this key can call:

| Resource | Notes |
|---|---|
| `supervised_fine_tuning_jobs` | SFT (LoRA) |
| `dpo_jobs` | DPO + `get_metrics_file_endpoint` |
| `reinforcement_fine_tuning_jobs` | RL — `cancel/create/get/list/resume` |
| `reinforcement_fine_tuning_steps` | RL step control |
| `datasets` | upload / validate / download endpoints |
| `evaluators` | reward evaluator code + build logs |
| `evaluation_jobs` | eval runs + log endpoint |
| `deployments` | serve fine-tuned model, scale |
| `lora` | load / unload adapters |
| `batch_inference_jobs` | offline batch |

## 3. Prior ORena work already on this account

The account already ran ORena RL on Fireworks (July–Aug 2026). This is real
prior art — reuse it as a reference, or build fresh with full instrumentation.

- **3 completed RL jobs** using Fireworks **RFT** (`loss_config.method =
  METHOD_UNSPECIFIED`).
- Reward evaluator: `evaluation-orena-focus-reward`.
- Datasets:
  - `orena-focus-train-full` — 10,980 examples, 91.6M tokens
  - `orena-focus-resume-train` — 8,864 examples, 72.5M tokens
  - `orena-focus-unfiltered-5000` — 5,000 examples, 41.3M tokens
  - `orena-focus-eval-200` — 200 examples
  - ~24 `rft-evalv3-iqev22iu-epoch-0-chunk-N` rollout datasets
- Fine-tuned model `orena-focus-rl-v3` deployed on **1× NVIDIA H200 (READY)**.
- Unrelated: `glm-5p2` deployed on **8× NVIDIA B300 (READY)** — cost watch.

## 4. GoEmotions data

Extracted to `fireworks_vs_sagemaker/data/goemotions/`:

- `data/train.tsv` 43,410 rows · `dev.tsv` 5,426 · `test.tsv` 5,427
- 28 labels in `data/emotions.txt` (27 emotions + neutral)
- Row format: `text \t comma-separated-label-ids \t id`

## 5. Watch-items

1. Account `state=UPDATING`, `status=INTERNAL` ("contact Fireworks AI team").
   Lists still work, so it may be stale — but confirm before launching paid jobs.
2. `glm-5p2` on 8× B300 is READY. Confirm it is wanted, or it may be billing.

## 6. Open decisions (blocking the run)

1. Reuse prior ORena artifacts (evaluator, dataset format) or build fresh?
2. RL method: RFT (already used), GRPO (standard), or DPO?
3. Base models — ORena VLM and GoEmotions text.
4. Budget ceiling.

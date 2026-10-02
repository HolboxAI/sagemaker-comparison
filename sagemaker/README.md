# AWS SageMaker — run data

This folder is the SageMaker team's contribution to the three-platform benchmark.
It mirrors the Fireworks layout so the comparison script can join the two.

## Contents

| Folder | What it holds |
|---|---|
| `config/` | the launch config + findings — `SAGEMAKER_4EXP_CONFIG.md` |
| `scripts/` | every script used to set up and launch the jobs |
| `scripts/src/` | the training entry points uploaded to SageMaker as `source_dir` |
| `scripts/iam/` | the execution-role trust policy, scoped policy, and create script |
| `logs/` | raw console logs (populated when jobs reach a terminal state) |
| `metrics/` | `run_manifest.json` now; `final_results.json` when all jobs finish |

## Result summary

Four jobs launched 2026-10-02 (fresh, 1 epoch, GRPO for RL). One finished, three
running — see [`config/SAGEMAKER_4EXP_CONFIG.md`](config/SAGEMAKER_4EXP_CONFIG.md)
and [`metrics/run_manifest.json`](metrics/run_manifest.json).

| # | Experiment | Job name | State |
|---|---|---|---|
| 1 | ORena SFT | `orena-sft-2026-10-02-16-48-24-104` | InProgress |
| 2 | ORena RL (GRPO) | `orena-rl-2026-10-02-16-30-52-271` | InProgress |
| 3 | GoEmotions SFT | `goemotions-sft-2026-10-02-16-07-29-859` | **Completed** (loss 1.54) |
| 4 | GoEmotions RL (GRPO) | `goemotions-rl-2026-10-02-16-30-25-068` | InProgress |

## Reproduce

```bash
# one-time: create the execution role (assumes AWS_PROFILE=stanford_gpu)
bash scripts/iam/create_role.sh

# launch all four (or skip some). Blocking with --wait.
export ROLE_ARN=arn:aws:iam::751871643798:role/sagemaker-stanford-exec
python scripts/launch_jobs.py
python scripts/launch_jobs.py --skip goemotions_rl orena_rl   # e.g. only SFT

# tail the jobs to completion; full logs land in scripts/logs/<job>.log
python scripts/monitor_jobs.py
```

The estimator is **script mode** (PyTorch 2.5.1 / py311, no Docker/ECR): the
`scripts/src/` files are uploaded as `source_dir` and run as entry points, and
`requirements.txt` pins the transformers/trl/peft versions at job start.

## Rules

1. Use the **same input data** as Fireworks — the canonical files in `../data/`
   with hashes in `../data/manifest.json`. Do not re-split or re-shuffle.
   _This run read from S3 instead; recorded as a finding in §8.5 of the config._
2. Use the **same hyperparameters** — the matrix in
   [`../fireworks/config/FIREWORKS_4EXP_CONFIG.md`](../fireworks/config/FIREWORKS_4EXP_CONFIG.md).
   _Deviations forced by the smaller GPUs are recorded in §8.2._
3. Note any place where SageMaker forced a different value as a finding, not silently.
4. Write `final_results.json` in the exact schema from
   [`../analysis/schema.md`](../analysis/schema.md).

> Secrets: AWS credentials live in the local `~/.aws` profile `stanford_gpu`; the
> scripts reference it by name only. The IAM role ARN / account id are identifiers,
> not secrets.

# AWS SageMaker — run data

This folder is for the SageMaker team's results.

## What to add

Mirror the Fireworks layout so the comparison script can join them:

```
sagemaker/
├── config/       # launch config (dataset ids, base models, hyperparameters)
├── scripts/      # every script used to build data and launch jobs
├── logs/         # raw console logs
└── metrics/
    └── final_results.json   # same schema as analysis/schema.md
```

## Rules

1. Use the **same input data** as Fireworks — the canonical files in `../data/`
   with hashes in `../data/manifest.json`. Do not re-split or re-shuffle.
2. Use the **same hyperparameters** — the matrix in
   [`../fireworks/config/FIREWORKS_4EXP_CONFIG.md`](../fireworks/config/FIREWORKS_4EXP_CONFIG.md).
3. Note any place where SageMaker forced a different value (for example, a
   different base model). Record it as a finding, not silently.
4. Write `final_results.json` in the exact schema from
   [`../analysis/schema.md`](../analysis/schema.md).

One known difference to record: SageMaker TRL GRPO can run RL on
`Qwen/Qwen3.5-9B`, which Fireworks cannot. If you run Experiment 2 on the 9B
model, flag that the base model differs from Fireworks' `qwen3-vl-8b-instruct`.

# Prime Intellect — run data

This folder is for the Prime Intellect team's results.

## What to add

Mirror the Fireworks layout:

```
prime-intellect/
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
3. Note any place where Prime Intellect forced a different value, and record it
   as a finding.
4. Write `final_results.json` in the exact schema from
   [`../analysis/schema.md`](../analysis/schema.md).

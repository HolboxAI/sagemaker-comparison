# Fine-tuning + RL Platform Benchmark

This repository is the single aggregation point for one experiment run on three
platforms. The experiment compares fine-tuning and reinforcement-learning
features across:

1. **Fireworks AI** — data is in [`fireworks/`](fireworks/)
2. **AWS SageMaker** — data goes in [`sagemaker/`](sagemaker/)
3. **Prime Intellect** — data goes in [`prime-intellect/`](prime-intellect/)

All analysis, metric comparison, and cost comparison happens here in
[`analysis/`](analysis/). Every platform runs the **same** experiment with the
**same** data and the **same** logging schema, so the results can be diffed
one-to-one.

---

## The experiment

Two datasets, each run through side-by-side SFT and RL (GRPO):

| # | Experiment | Dataset | Model class | Method | Base model (HF) |
|---|---|---|---|---|---|
| 1 | ORena VLM SFT | ORena surgical VQA | vision-language | supervised fine-tuning (LoRA) | `Qwen/Qwen3.5-9B` |
| 2 | ORena VLM RL | ORena surgical VQA | vision-language | GRPO | `Qwen/Qwen3-VL-8B-Instruct`¹ |
| 3 | GoEmotions SFT | GoEmotions text | text classifier | supervised fine-tuning (LoRA) | `Qwen/Qwen3-0.6B` |
| 4 | GoEmotions RL | GoEmotions text | text classifier | GRPO | `Qwen/Qwen3-0.6B` |

¹ Fireworks does not allow RL on `qwen3p5-9b`; SageMaker and Prime Intellect can
use `Qwen/Qwen3.5-9B` for RL. This is itself a measured platform difference.
Full matrix and all hyperparameters: [`fireworks/config/FIREWORKS_4EXP_CONFIG.md`](fireworks/config/FIREWORKS_4EXP_CONFIG.md).

**Shared constraints (all platforms):**
- Fresh runs — no warm-start from a prior artifact.
- RL method = GRPO, KL beta 0.04, temperature 0.7, 8 rollouts per prompt.
- 1 epoch each (short runs to fit a small budget).

---

## Repository layout

```
.
├── data/                  # canonical input data (identical on every platform)
│   ├── manifest.json      # sha256 + row counts + sources for every dataset file
│   ├── goemotions/        # GoEmotions (text) — splits, labels, build scripts
│   └── orena/             # ORena (VLM) — format samples; big files not committed
├── fireworks/             # Fireworks run: config, scripts, logs, metrics, evaluators
├── sagemaker/             # SageMaker team adds their data here
├── prime-intellect/       # Prime Intellect team adds their data here
└── analysis/              # shared comparison code + the canonical metrics schema
```

---

## Status

| Platform | Status | Results |
|---|---|---|
| Fireworks | ✅ complete (4/4 jobs) | [`fireworks/metrics/final_results.json`](fireworks/metrics/final_results.json) |
| SageMaker | ⏳ pending — team to add | — |
| Prime Intellect | ⏳ pending — team to add | — |

### Fireworks quick results (1 epoch, fresh)

| Experiment | Final metric | Start → final |
|---|---|---|
| ORena SFT | train loss | 0.354 |
| ORena RL | exact_match | 0.191 → **0.282** |
| GoEmotions SFT | train loss | 0.313 |
| GoEmotions RL | micro_f1 | 0.175 → **0.377** |

**Results graphs:** [`analysis/plots/fireworks_summary.png`](analysis/plots/fireworks_summary.png)
(SFT loss curves + RL reward curves). Individual charts are in the same folder.

---

## How to add a platform's data

1. Create a folder named after the platform if it does not exist
   (`sagemaker/`, `prime-intellect/`).
2. Put the platform's run artifacts there:
   - `config/` — the exact launch config (dataset ids, base models, hyperparameters).
   - `scripts/` — every script used to build data and launch jobs.
   - `logs/` — raw console logs, verbatim.
   - `metrics/` — a `final_results.json` in the **same schema** as
     [`analysis/schema.md`](analysis/schema.md), plus any raw metric files.
3. Use the **same** input data as Fireworks. The canonical files are in
   [`data/`](data/) with hashes in [`data/manifest.json`](data/manifest.json).
   Do not re-split or re-shuffle the data between platforms.
4. Open a pull request. A single `analysis/` script will join all platforms'
   `final_results.json` files for comparison.

---

## Key findings so far (Fireworks)

- Fireworks managed RL has a **narrower vision-model catalog** than SageMaker:
  `qwen3p5-9b` is SFT-tunable but not RL-tunable. Vision RL has one managed
  shape (`qwen3-vl-8b-instruct`).
- Custom reward evaluators must be uploaded as a **single `.tar.gz` tarball**
  (`evaluators.create` with no `source`/`requirements` field, then one
  signed-URL PUT). Loose-file uploads fail with an opaque `BUILD_FAILED`.

Details: [`fireworks/config/FIREWORKS_SETUP_FINDINGS.md`](fireworks/config/FIREWORKS_SETUP_FINDINGS.md).

---

## Data notes

ORena rows carry base64 JPEG images inline, so the full JSONL files are large
(up to 362 MB). They are **not** committed to git. Their hashes, row counts, and
regeneration steps are in [`data/manifest.json`](data/manifest.json). See
[`data/README.md`](data/README.md).

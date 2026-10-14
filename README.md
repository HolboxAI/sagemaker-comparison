# Fine-tuning + RL Platform Benchmark

This repository is the single aggregation point for one experiment run on four
platforms. The experiment compares fine-tuning and reinforcement-learning
features across:

1. **Fireworks AI** — data is in [`fireworks/`](fireworks/)
2. **AWS SageMaker** — data goes in [`sagemaker/`](sagemaker/)
3. **Prime Intellect** — data goes in [`prime-intellect/`](prime-intellect/)
4. **Miles** — data is in [`miles/`](miles/) (self-hosted RL framework)

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

> ⚠️ Deviation: the Miles GoEmotions RL leg ran with `kl_beta = 0` (see the
> Miles quick-results caveats below). Re-running at `kl_beta = 0.04` is the
> clean follow-up before the reward curves are compared one-to-one.

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
├── miles/                 # Miles (self-hosted RL) — reward ports, data prep, launch scripts
├── baseten/               # Baseten attempt (CholecT50 task) — blocked at authorization
├── together-ai/           # Together AI feasibility probe (CholecT50) — RL not available
├── tinker/                # Thinking Machines Tinker run (CholecT50) — SFT → RL/GRPO complete
└── analysis/              # shared comparison code + the canonical metrics schema
```

---

## Status

| Platform | Status | Results |
|---|---|---|
| Fireworks | ✅ complete (4/4 jobs) | [`fireworks/metrics/final_results.json`](fireworks/metrics/final_results.json) |
| SageMaker | ⏳ pending — team to add | — |
| Prime Intellect | ✅ partial — 2 complete, 1 blocked, 1 not started | [`prime-intellect/metrics/final_results.json`](prime-intellect/metrics/final_results.json) |
| Miles | ✅ complete — GoEmotions RL (256 rollouts, GRPO) on g5.2xlarge | [`miles/metrics/final_results.json`](miles/metrics/final_results.json) |
| Baseten (CholecT50) | 🚫 blocked — training not authorized | [`baseten/EXPERIENCE_REPORT.md`](baseten/EXPERIENCE_REPORT.md) |
| Together AI (CholecT50) | ❌ not feasible — no RL/GRPO | [`together-ai/EXPERIENCE_REPORT.md`](together-ai/EXPERIENCE_REPORT.md) |
| Tinker (CholecT50) | ✅ complete — SFT → RL/GRPO | [`tinker/metrics/final_results.json`](tinker/metrics/final_results.json) |

### Fireworks quick results (1 epoch, fresh)

| Experiment | Final metric | Start → final |
|---|---|---|
| ORena SFT | train loss | 0.354 |
| ORena RL | exact_match | 0.191 → **0.282** |
| GoEmotions SFT | train loss | 0.313 |
| GoEmotions RL | micro_f1 | 0.175 → **0.377** |

**Results graphs:** [`analysis/plots/fireworks_summary.png`](analysis/plots/fireworks_summary.png)
(SFT loss curves + RL reward curves). Individual charts are in the same folder.

### Prime Intellect quick results (on-demand pods, prime-rl 0.7.0)

| Experiment | Final metric | Result |
|---|---|---|
| ORena SFT | train loss | 🚫 blocked (pod image rejects `[model.vlm]` for SFT) |
| ORena RL | exact_match | ⏳ not started (feasible) |
| GoEmotions SFT | micro_f1 | **0.4956** (train loss 0.167) |
| GoEmotions RL | micro_f1 | **0.035** (Jaccard 0.018; did not learn) |

Full matrix, small-run sizes, and the platform-specific findings (VLM SFT version drift,
thinking-mode RL footgun, LoRA checkpoint bug, …):
[`prime-intellect/config/PRIME_INTELLECT_4EXP_CONFIG.md`](prime-intellect/config/PRIME_INTELLECT_4EXP_CONFIG.md).

### Miles quick results (self-hosted GRPO, Qwen3-0.6B, 1 epoch, fresh)

| Experiment | Metric | Start → final | Peak |
|---|---|---|---|
| GoEmotions RL | Jaccard | 0.133 → **0.602** | 0.656 |
| GoEmotions RL | micro_f1 (sampled) | — → **0.439** | — |

**vs Fireworks (same base model, same data, same 64-token cap):**

| Metric | Fireworks | Miles | Δ |
|---|---|---|---|
| GoEmotions RL Jaccard final | 0.344 | **0.602** | +0.26 |
| GoEmotions RL micro_f1 final | 0.377 | **0.439** | +0.06 |

Miles reaches a higher Jaccard on the same text-RL task, but two caveats keep the
comparison from being strictly apples-to-apples:

- **KL divergence**: Miles ran `kl_beta = 0` (fire-and-forget GRPO). Fireworks used
  `kl_beta = 0.04` per the shared spec. A higher KL penalty constrains Fireworks to
  stay closer to the base model, which caps its reward ceiling. The fair follow-up
  is Miles at `kl_beta = 0.04`.
- **Verbosity**: Miles' model ends at ~100% truncated responses (`truncated_mean 0.84`,
  `response_len_mean 55` of 64). It learned to fill the token budget while getting
  labels parseable, a mild reward-hacking signature. Fireworks' truncation rate was
  not reported.

Details: [`miles/metrics/run_manifest.json`](miles/metrics/run_manifest.json) and
[`miles/metrics/final_results.json`](miles/metrics/final_results.json).
Comparison chart: [`analysis/plots/miles_vs_fireworks_goemotions.png`](analysis/plots/miles_vs_fireworks_goemotions.png)
(regenerate with `python3 analysis/plot_miles_vs_fireworks.py`).

---

### Platform feasibility probes (a different task: CholecT50)

`baseten/`, `together-ai/`, and `tinker/` document a **separate, later experiment** —
CholecT50 surgical action-triplet recognition (SFT → RLVR with a custom triplet-F1
reward) — not the ORena/GoEmotions benchmark above. Two platforms could not run it;
Tinker completed the full pipeline:

| Platform | Outcome | One-line reason |
|---|---|---|
| Baseten | 🚫 blocked before launch | HTTP 403 — org lacks the Training Jobs entitlement |
| Together AI | ❌ not feasible | self-serve fine-tuning is SFT + DPO only (no RL/GRPO) |
| Tinker | ✅ complete | SFT → GRPO ran; triplet F1 0.000 → 0.136 (SFT) → 0.159 (RL) |

Details: [`baseten/EXPERIENCE_REPORT.md`](baseten/EXPERIENCE_REPORT.md),
[`together-ai/EXPERIENCE_REPORT.md`](together-ai/EXPERIENCE_REPORT.md), and
[`tinker/README.md`](tinker/README.md).

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

# Prime Intellect run data

This folder holds the Prime Intellect leg of the three-platform benchmark.

The run is **headless** (CLI + REST API, no dashboard). Training is self-run
`prime-rl` on an on-demand pod, not the hosted "training runs" surface.

## Status

| # | Experiment | Model | State |
|---|---|---|---|
| 1 | ORena SFT | Qwen3.5-9B (VLM) | 🚫 blocked (VLM SFT version drift) |
| 2 | ORena RL (GRPO) | Qwen3.5-9B (VLM) | ⏳ not started (feasible) |
| 3 | GoEmotions SFT | Qwen3-0.6B | ✅ complete — micro-F1 **0.4956** / Jaccard 0.3294 |
| 4 | GoEmotions RL (GRPO) | Qwen3-0.6B | ✅ complete — micro-F1 **0.0349** / Jaccard 0.0178 |

Machine-readable state (canonical schema): [`metrics/final_results.json`](metrics/final_results.json).

## Contents

| File | What it holds |
|---|---|
| `config/PRIME_INTELLECT_4EXP_CONFIG.md` | the 4-experiment config + small-run sizes + findings (Fireworks parity) |
| `config/*.toml` | exact launch configs for all four experiments |
| `scripts/` | data-build, reward, and eval scripts |
| `logs/` | raw console logs + the Exp 3 loss CSV |
| `metrics/final_results.json` | results in the canonical `analysis/schema.md` schema |
| `metrics/run_manifest.json` | run state snapshot (legacy) |
| `DECISIONS_LOG.md` | every decision + execution step, append-only |
| `EXPERIENCE_REPORT.md` | the headless experience: failures, root causes, retries |
| `EXPERIMENT_RESULTS.md` | per-experiment results, loss curves, billing |

## Key facts

- **Compute:** on-demand pods, self-run `prime-rl` 0.7.0 (`prime pods create` → `ssh` →
  `uv run sft/rl @ config.toml`). Full fine-tune is gated (empty GPU list); hosted LoRA SFT
  is gated on closed-beta volumes.
- **Models:** `Qwen/Qwen3.5-9B` (ORena), `Qwen/Qwen3-0.6B` (GoEmotions). PI can RL the 9B
  directly, unlike Fireworks.
- **Pod image version drift:** the pods ship prime-rl **0.7.0**, which rejects `[model.vlm]`
  for SFT (ORena SFT blocked) — GitHub main has removed the check. See §8.1 of the config doc.
- **Data:** GoEmotions fetched from canonical HF `google-research-datasets/go_emotions`
  (800 train / 200 dev).
- **Cost:** ≈ $20 total pod spend; an idle H200 ($4.50/hr) was the largest line item and
  produced no run.

## Headless friction (the ease-of-use story)

1. Volatile GPU inventory. On-demand offers go stale in minutes (5 create attempts).
2. SSH key. A passphrase-protected local key silently breaks automation (2 pod recreations).
3. Config schema drift. The pod image ships an older prime-rl than the public docs.
4. No managed dashboard for pods. Pod work shows nothing in the "Jobs" view.

Details in [`EXPERIENCE_REPORT.md`](EXPERIENCE_REPORT.md).

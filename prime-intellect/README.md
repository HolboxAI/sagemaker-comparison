# Prime Intellect — run data

This folder holds the Prime Intellect leg of the three-platform benchmark.

The run is **headless** (CLI + REST API, no dashboard). Training is self-run
`prime-rl` on an on-demand pod, not the hosted "training runs" surface.

## Status

| # | Experiment | Model | State |
|---|---|---|---|
| 1 | ORena SFT | Qwen3.5-9B (VLM) | not started |
| 2 | ORena RL (GRPO) | Qwen3.5-9B (VLM) | not started |
| 3 | GoEmotions SFT | Qwen3-0.6B | running (loss 5.79 → 0.25 over 60 steps) |
| 4 | GoEmotions RL (GRPO) | Qwen3-0.6B | not started |

No final metrics yet. Machine-readable state: [`metrics/run_manifest.json`](metrics/run_manifest.json).

## Contents

| File | What it holds |
|---|---|
| `DECISIONS_LOG.md` | every decision + execution step, append-only |
| `EXPERIENCE_REPORT.md` | the headless experience: failures, root causes, retries |
| `EXPERIMENT_RESULTS.md` | status, live loss curve, smoke test, billing |
| `metrics/run_manifest.json` | run state in the canonical schema |

## Key facts

- **Compute:** on-demand pod — A100 40GB SXM4 at $1.99/hr, self-run `prime-rl`
  (`prime pods create` → `ssh` → `uv run sft @ config.toml`). Full fine-tune is
  gated (empty GPU list); hosted LoRA SFT is gated on closed-beta volumes.
- **Models:** `Qwen/Qwen3.5-9B` (ORena), `Qwen/Qwen3-0.6B` (GoEmotions). PI can
  RL the 9B directly, unlike Fireworks.
- **Data:** GoEmotions fetched from canonical HF `google-research-datasets/go_emotions`
  (800 train / 200 dev). ORena data still pending.
- **Billing:** wallet `$48.79`; `$1.22` spent so far.

## Headless friction (the ease-of-use story)

1. Volatile GPU inventory — on-demand offers go stale in minutes (5 create attempts).
2. SSH key — a passphrase-protected local key silently breaks automation (2 pod recreations).
3. Config schema drift — the pod image ships an older prime-rl than the public docs.
4. No managed dashboard for pods — pod work shows nothing in the "Jobs" view.

Details in [`EXPERIENCE_REPORT.md`](EXPERIENCE_REPORT.md).

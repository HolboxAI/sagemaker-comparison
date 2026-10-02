# Fireworks AI — run data

This folder holds everything from the Fireworks phase of the benchmark. It is
the reference platform: the other two teams reproduce these runs from the same
data and schema.

## Contents

| Folder | What it holds |
|---|---|
| `config/` | the 4-experiment config, the plan, setup findings, and the playbook |
| `scripts/` | every script used, numbered in the order they ran (`00_` → `36_`) |
| `logs/` | raw console logs, verbatim |
| `metrics/` | `final_results.json` + `run_manifest.json` + per-step SFT metrics |
| `evaluators/` | the GoEmotions reward evaluator code |
| `reference/` | the prior team's ORena evaluator (code + README, no data) |

## Result summary

All four jobs ran on 2026-10-02, fresh, 1 epoch, GRPO with 8 rollouts/prompt
for RL. Full numbers: [`metrics/final_results.json`](metrics/final_results.json).

| # | Experiment | Job | State | Final metric |
|---|---|---|---|---|
| 1 | ORena SFT | `etia90sh` | COMPLETED | train loss 0.354 |
| 2 | ORena RL (GRPO) | `fs-or1-orena-rl` | COMPLETED | exact_match 0.191 → 0.282 |
| 3 | GoEmotions SFT | `ktcdqquf` | COMPLETED | train loss 0.313 |
| 4 | GoEmotions RL (GRPO) | `fs-or1-ge-rl` | COMPLETED | micro_f1 0.175 → 0.377 |

### Platform findings

1. **`qwen3p5-9b` is SFT-tunable but not RL-tunable** on Fireworks managed RFT.
   Vision RL has a single managed shape: `qwen3-vl-8b-instruct`. SageMaker TRL
   GRPO has no such limit.
2. **Custom reward evaluators upload as a single `.tar.gz` tarball.**
   `evaluators.create(evaluator_id=…, evaluator={display_name, description,
   entry_point})` with no `source`/`requirements` field, then one signed-URL
   PUT. Loose-file uploads fail with `BUILD_FAILED status=INTERNAL`.

Details and verification steps are in
[`config/FIREWORKS_SETUP_FINDINGS.md`](config/FIREWORKS_SETUP_FINDINGS.md).

## Reproduce

The portable config is
[`config/FIREWORKS_4EXP_CONFIG.md`](config/FIREWORKS_4EXP_CONFIG.md). The
scripts are numbered in run order. Start there; every script is self-contained
and prints its own results.

> Secrets: the Fireworks API key lives in a git-ignored `.env`. Do not commit it.

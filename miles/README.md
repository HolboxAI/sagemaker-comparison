# Miles — RL leg

Fourth platform in the fine-tuning + RL benchmark. Miles is an open-source RL
framework (fork of `slime`): SGLang for rollout, Megatron-LM or PyTorch FSDP for
training. It is self-hosted, like Prime Intellect.

This leg covers the two RL experiments (GRPO). SFT stays on the other three
platforms.

## Files

- [`config/MILES_RL_CONFIG.md`](config/MILES_RL_CONFIG.md) — portable RL config,
  reward contract, data format, and findings.
- [`DECISIONS_LOG.md`](DECISIONS_LOG.md) — decisions and reasoning.
- [`PROMPTS_AND_INTENTS.md`](PROMPTS_AND_INTENTS.md) — the brief and what the
  agent did with it.
- [`scripts/reward_orena.py`](scripts/reward_orena.py) — ORena reward, port of
  the Fireworks evaluator. Offline smoke test included.
- [`scripts/reward_goemotions.py`](scripts/reward_goemotions.py) — GoEmotions
  reward (Jaccard). Offline smoke test included.
- [`scripts/prepare_data.py`](scripts/prepare_data.py) — builds Miles JSONL from
  the canonical `data/` files.
- [`scripts/run_goemotions_rl.sh`](scripts/run_goemotions_rl.sh) — launch,
  FSDP backend, de-risk leg.
- [`scripts/run_orena_rl.sh`](scripts/run_orena_rl.sh) — launch, Megatron
  backend + VLM generate function, follow-on.
- [`scripts/generate_orena.py`](scripts/generate_orena.py) — VLM generate
  function (template).
- [`data/goemotions_rl_miles.jsonl`](data/goemotions_rl_miles.jsonl) — 5,000
  rows, committed, runnable now.
- [`data/orena_rl_miles_sample.jsonl`](data/orena_rl_miles_sample.jsonl) — 2
  rows, smoke only. Full ORena data is regenerated before the run.
- [`metrics/run_manifest.json`](metrics/run_manifest.json) — status.

## Status

Launch-ready. GoEmotions RL reward and data are built and smoke-tested offline.
The GPU run is the next step.

## Run it

1. Start a GPU node on the `riley` AWS profile.
2. Pull the Miles image: `docker pull radixark/miles:latest`.
3. Copy `scripts/` and `data/goemotions_rl_miles.jsonl` to the node.
4. Inside the container, run `BENCH_DIR=/root/miles_bench bash run_goemotions_rl.sh`.

# Miles — Agent Prompt & Intent

Agent: Claude Code (harness) + DeepSeek-V4-Pro (model). Same agent as the
Fireworks, SageMaker, and Prime Intellect legs.

## The brief

> We will also be using Miles as a comparison for RL. We already have loss
> curves for the rest. Test ORena and the other tests using Miles and compare.
> Analyze the difference in performance. Push the comparable results to the
> same repo. Use the riley AWS profile to run the experiments.

## What the agent did with it

1. Identified Miles (open-source RL framework, fork of `slime`, SGLang rollout +
   Megatron/FSDP training).
2. Read the Miles docs and source: install, GRPO launch args, the `Sample`
   dataclass, and the `--custom-rm-path` / `--reward-key` / `--label-key`
   reward contract.
3. Ported the two Fireworks reward evaluators to single-file Miles reward
   modules, byte-for-byte on scoring logic.
4. Built the Miles-format prompt data from the canonical `data/` sources.
5. Wrote the two launch scripts (GoEmotions de-risk on FSDP, ORena on Megatron).
6. Checked the `riley` AWS profile for GPU compute.

## Outcome

The leg is launch-ready. GoEmotions RL data and reward are built and smoke-tested
offline. The GPU run is the next step.

## Observed vs the other platforms

Miles is the second self-run platform (like Prime Intellect), but its RL surface
is a clean single-file reward function, not Prime Intellect's raw pod plus
`prime-rl` config drift. Miles is the only platform whose VLM RL base model
(`Qwen3-VL-8B-Instruct`) matches Fireworks exactly.

# Fireworks — Agent Prompt & Intent

Agent: Claude Code (harness) + DeepSeek-V4-Pro (model). Same agent as the SageMaker and Prime Intellect legs.

## The brief (initial prompt)

Reconstructed from the saved screenshot `docs/images/run_brief.png`.

> Need you to plan the experiment. The whole experiment is comparing SageMaker vs Fireworks, but this run focuses only on the Fireworks side. We have to carry out two experiments, both fine-tuning and RL, on the ORena dataset. Train for 1 to 2 epochs (15 to 20 min). Log the metrics and other things like the loss, so we can map and get the general idea of the scripts, structure, logs, steps taken, and the actual analytic metrics. Finally some graphs to help compare Fireworks and SageMaker. Carry out the same experiment on the GoEmotions dataset (kaggle.com/datasets/debarshichanda/goemotions) with a small LLM base model, and train a sentiment classifier with fine-tuning and RL, same as the last use case. This gives us ideas for VLM (ORena) and text LLMs (sentiment). We use only Fireworks. Let me know what you need. Finally, this is a deep research project: map out why Fireworks is more feature-rich than SageMaker.

## What the agent did with it

1. Planned the Fireworks phase (`fireworks/config/FIREWORKS_VS_SAGEMAKER_PLAN.md`).
2. Verified the account and training access (`fireworks/config/FIREWORKS_SETUP_FINDINGS.md`).
3. Built the two datasets (ORena vision, GoEmotions text).
4. Launched 4 jobs: 2 SFT + 2 RL (GRPO).
5. Collected loss and reward curves.

## Outcome

All four jobs completed. Final metrics: `fireworks/metrics/final_results.json`.

## Observed vs SageMaker

The Fireworks agent found the managed training and RL paths through the SDK on its own. It did not need a recipe step or a serverless discovery step. The SageMaker agent defaulted to script mode and never surfaced the managed recipes or serverless model customization.

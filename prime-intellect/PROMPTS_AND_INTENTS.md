# Prime Intellect — Agent Prompts & Intents

Agent: Claude Code (harness) + DeepSeek-V4-Pro (model). Same agent as the SageMaker and Fireworks legs.

Prompts in order (from `prime-intellect/EXPERIENCE_REPORT.md`, section 1).

| # | Prompt | Intent |
|---|---|---|
| 1 | Reproduce the 4 experiments in `FIREWORKS_4EXP_CONFIG.md` on Prime Intellect (not Fireworks); small runs (20 to 30 min each); everything in `prime_intellect/`; one doc per step; capture screenshot + loss/reward plot + final result; tensorboard/CSV, no W&B | Overall brief + evidence + logging constraints |
| 2 | "run it" | Drive pod creation + training to completion |
| 3 | "please explain the error" | Explain SSH "Permission denied" |
| 4 | "i have only prime intellect api key… what should i do?" | Auth provisioning (API key only, no OAuth) |
| 5 | "register the key" | Explicit authorization to register the SSH public key via API |
| 6 | "update?" / "check again" (several times) | Progress checks during pod provisioning |
| 7 | "make a report from this chat…" | The experience report itself |

Compute and data-source decisions were made interactively before the run phase.
See `prime-intellect/DECISIONS_LOG.md` for those decisions.

## Observed vs SageMaker

Prime Intellect exposed a raw pod surface, so the agent ran headless `prime-rl`
on an on-demand A100. It never reached a hosted or managed training surface,
because those were gated or empty for this account. SageMaker is the opposite:
the managed surface exists but the agent defaulted to script mode and never
surfaced it.

# Tinker run configuration (CholecT50 SFT → RL/GRPO)

Everything below is reproducible from the `scripts/` directory. The training
loop runs on Tinker's managed GPUs; the local process only holds the tokenizer,
image processor, and one micro-batch at a time.

## Shared

| Setting | Value |
|---|---|
| Model | `Qwen/Qwen3.5-4B` (vision-language) |
| Renderer | `qwen3_5_disable_thinking` (no `<think>` blocks → plain triplet output) |
| Max sequence length | 8192 |
| Dataset | CholecT50 — train 2500 / val 300 / test 800 |
| Prompt / vocab | `reward.py` (instruments, verbs, targets; triplet format `instrument-verb-target`) |
| Reward | `0.4 * triplet_F1 + 0.6 * mean(instrument, verb, target F1)` |
| SDK | `tinker` 0.33.1, `tinker-cookbook` 0.5.7 |

## Stage 1 — SFT (`tinker_sft.py`)

| Setting | Value |
|---|---|
| Samples | 2500 train frames |
| Micro-batch | 4 |
| Steps | 625 (= one epoch) |
| LoRA rank | 16 |
| Learning rate | 2e-4 (Adam) |
| Loss | cross-entropy (`forward_backward(loss_fn="cross_entropy")`) |
| Checkpoint | `save_state()` every 10 steps → `weights/` path |
| Final loss | 0.1254 (from ~1.6) |
| Wall time | ~3277 s |

CLI: `python tinker_sft.py --micro_batch 4 --lr 2e-4 --rank 16 --max_steps 625`

## Stage 2 — RL / GRPO (`tinker_rl.py`)

| Setting | Value |
|---|---|
| Samples | 256 train frames |
| Batch size | 16 (groups per step) |
| Group size | 8 rollouts per prompt |
| Learning rate | 1e-5 |
| LoRA rank | 16 |
| Loss | importance sampling (`loss_fn="importance_sampling"`) |
| Init | loads the SFT `weights/` checkpoint (`load_checkpoint_path`) |
| Steps | 16 GRPO batches |
| Reward | 0.31 → peak 0.487 → final 0.38 |
| Wall time | ~393 s |

CLI: `python tinker_rl.py --max_samples 256 --batch_size 16 --group_size 8 --lr 1e-5`

## Eval (`tinker_eval.py`)

| Setting | Value |
|---|---|
| Split | val (n = 300 frames) |
| Decoding | greedy (temp 0.0), max 128 tokens |
| Metrics | triplet / instrument / verb / target F1 + blended reward |

`--base` evaluates the unfine-tuned model; `--ckpt <tinker://.../sampler_weights/...>`
evaluates a converted checkpoint (see `EXPERIENCE_REPORT.md` for the conversion).

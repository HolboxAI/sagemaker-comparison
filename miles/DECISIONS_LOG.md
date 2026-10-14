# Miles — Decisions Log

_Compiled 2026-10-14. Decisions and reasoning for the Miles RL leg._

## 1. What Miles is, and why it fits

Miles is an open-source RL framework, a fork of `slime`. Rollout runs on SGLang.
Training runs on Megatron-LM or PyTorch FSDP. It is self-hosted on a GPU node.

It fits the benchmark because:
- It supports GRPO, the method the other three platforms use.
- It supports VLM RL on `Qwen3-VL-8B-Instruct`, the exact Fireworks RL base model.
- It uses the same RLVR shape (prompt + verifiable ground truth) as Fireworks
  and SageMaker.

## 2. Compute

- AWS profile `riley` (user `stanford_gpu`, account 751871643798, us-east-1).
- The standard Miles recipe wants one node with 8 H100/H200/B200 GPUs and Docker.
- The `riley` profile has smaller classes: g6e (L40S), g7 (L4), g7e (Blackwell).
- The FSDP backend runs on fewer GPUs, so GoEmotions de-risks on a single L40S.
- ORena VLM RL needs the 8-GPU node. It is the follow-on.

## 3. Models

- GoEmotions RL: `Qwen3-4B`. `Qwen3-0.6B` (the Fireworks/SageMaker base) is not
  in Miles' supported list. This is a catalog gap, not a choice.
- ORena RL: `Qwen3-VL-8B-Instruct`. Exact match to Fireworks.

## 4. Data

- Reuse the canonical `data/` files. Do not reshuffle.
- Miles wants one `--input-key` (messages) and one `--label-key` (JSON string).
- Ground truth is JSON-encoded into a single key because `Sample.label` is `str`.
- The reward function parses that string back.

## 5. Reward

- Port the Fireworks scoring byte-for-byte into two single-file reward modules.
- ORena: dense correctness trains, exact-match reports (same split as Fireworks).
- GoEmotions: Jaccard trains, micro-F1 reports.
- Note the Miles fractional-reward caveat (spurious gradient when a group shares
  one reward value).

## 6. Run order

1. GoEmotions RL first (FSDP, one GPU, no VLM plumbing). De-risk.
2. ORena RL second (Megatron, VLM generate function, full data, 8 GPUs).

## 7. Open items

- [ ] Start the GPU instance on `riley` and pull `radixark/miles:latest`.
- [ ] Validate `reward_goemotions.reward_fn` against the installed Miles (Sample
      field names can drift between releases; the port targets `miles/utils/types.py`).
- [ ] Validate the VLM generate function against `examples/geo3k_vlm/rollout/generate.py`.
- [ ] Regenerate `data/orena/orena_rl.jsonl` (66 MB, not committed) before ORena RL.

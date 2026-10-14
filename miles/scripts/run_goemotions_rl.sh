#!/bin/bash
# GoEmotions RL (GRPO) on Miles. De-risk leg: small text model, no VLM plumbing.
#
# Runs INSIDE the miles container (radixark/miles:latest). Assumes:
#   - /root/models/Qwen3-0.6B/     HF weights (downloaded)
#   - /root/miles_bench/            reward_goemotions.py + goemotions_rl_miles.jsonl
#
# Backend is FSDP: it reads HF weights directly, so there is no Megatron
# conversion step. KL is 0 for the de-risk (no reference model needed).
set -ex

MODEL_NAME=${MODEL_NAME:-Qwen3-0.6B}
NUM_GPUS=${NUM_GPUS:-2}
BENCH_DIR=${BENCH_DIR:-/root/miles_bench}
NUM_ROLLOUT=${NUM_ROLLOUT:-256}

export PYTHONUNBUFFERED=1
export MASTER_ADDR=127.0.0.1
export no_proxy="127.0.0.1,localhost"

# Clean up any stale workers from a previous attempt.
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
pkill -9 miles 2>/dev/null || true
sleep 3

ray start --head \
  --node-ip-address 127.0.0.1 \
  --num-gpus "${NUM_GPUS}" \
  --disable-usage-stats \
  --dashboard-host=0.0.0.0 \
  --dashboard-port=8265

CKPT_ARGS=(
  --hf-checkpoint /root/models/${MODEL_NAME}
)

ROLLOUT_ARGS=(
  --prompt-data "${BENCH_DIR}/goemotions_rl_miles.jsonl"
  --input-key messages
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --custom-rm-path reward_goemotions.reward_fn
  --reward-key reward_value
  --log-reward-category reward_cat
  --num-rollout "${NUM_ROLLOUT}"
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --rollout-max-response-len 64
  --rollout-temperature 0.7
  --global-batch-size 128
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.00
  --kl-loss-type low_var_kl
  --entropy-coef 0.00
  --eps-clip 0.2
  --eps-clip-high 0.28
)

OPTIMIZER_ARGS=(
  --optimizer adam
  --lr 1e-5
  --lr-decay-style constant
  --weight-decay 0.1
  --adam-beta1 0.9
  --adam-beta2 0.98
)

BACKEND_ARGS=(
  --train-backend fsdp
  --gradient-checkpointing
  --sglang-attention-backend triton
  --update-weight-buffer-size 536870912
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "${NUM_GPUS}"
  --colocate
)

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${BENCH_DIR}:/root/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\"
  }
}"

cd /root/miles
ray job submit --address="http://127.0.0.1:8265" \
  --runtime-env-json="${RUNTIME_ENV_JSON}" \
  -- python3 train.py \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${MISC_ARGS[@]}"

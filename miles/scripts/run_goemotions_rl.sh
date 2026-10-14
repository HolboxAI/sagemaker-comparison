#!/bin/bash
# GoEmotions RL (GRPO) on Miles. De-risk leg: small text model, no VLM plumbing.
#
# Run inside the Miles container (radixark/miles:latest). The reward module and
# data must already be at $BENCH_DIR on the pod. See miles/README.md.
set -ex

BENCH_DIR=${BENCH_DIR:-/root/miles_bench}
MODEL_NAME=${MODEL_NAME:-Qwen3-4B}          # smallest text model Miles supports
NUM_GPUS=${NUM_GPUS:-1}                     # FSDP can run on 1 GPU for a smoke

export PYTHONUNBUFFERED=1
export PYTHONPATH="$BENCH_DIR:$PYTHONPATH"

# Download model (idempotent).
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
  hf download "Qwen/${MODEL_NAME}" --local-dir "/root/models/${MODEL_NAME}"
fi

CKPT_ARGS=(
  --hf-checkpoint "/root/models/${MODEL_NAME}"
)

ROLLOUT_ARGS=(
  --prompt-data "$BENCH_DIR/goemotions_rl_miles.jsonl"
  --input-key messages
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --custom-rm-path reward_goemotions.reward_fn
  --reward-key reward_value
  --log-reward-category reward_cat
  --num-rollout 256
  --rollout-batch-size 16
  --n-samples-per-prompt 8
  --rollout-max-response-len 64
  --rollout-temperature 0.7
  --global-batch-size 128
)

GRPO_ARGS=(
  --advantage-estimator grpo
  --kl-loss-coef 0.04
  --kl-loss-type low_var_kl
  --entropy-coef 0.0
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

# FSDP backend: reads HF weights directly, no Megatron conversion.
BACKEND_ARGS=(
  --train-backend fsdp
  --gradient-checkpointing
  --attn-implementation flash_attention_3
)

TOPOLOGY_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "$NUM_GPUS"
  --colocate
)

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.6
)

python3 train.py \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${TOPOLOGY_ARGS[@]}" \
  "${SGLANG_ARGS[@]}"

#!/bin/bash
# ORena VLM RL (GRPO) on Miles. Follow-on leg: needs the VLM generate function
# and the full ORena RL data (regenerate data/orena/orena_rl.jsonl first).
#
# Modeled on the canonical examples/geo3k_vlm/run_geo3k_vlm.sh. Model
# Qwen3-VL-8B-Instruct matches the Fireworks RL base model exactly.
set -ex

BENCH_DIR=${BENCH_DIR:-/root/miles_bench}
MODEL_NAME=${MODEL_NAME:-Qwen3-VL-8B-Instruct}
TRAIN_BACKEND=${TRAIN_BACKEND:-megatron}
NUM_GPUS=${NUM_GPUS:-8}

export PYTHONUNBUFFERED=1
export PYTHONPATH="$BENCH_DIR:$PYTHONPATH"
export MULTIMODAL_KEYS='{"image": "images"}'

# Download model (idempotent).
if [ ! -d "/root/models/${MODEL_NAME}" ]; then
  hf download "Qwen/${MODEL_NAME}" --local-dir "/root/models/${MODEL_NAME}"
fi

CKPT_ARGS=(
  --hf-checkpoint "/root/models/${MODEL_NAME}"
)

ROLLOUT_ARGS=(
  --prompt-data "$BENCH_DIR/orena_rl_miles.jsonl"
  --input-key messages
  --label-key label
  --apply-chat-template
  --rollout-shuffle
  --custom-rm-path reward_orena.reward_fn
  --reward-key reward_value
  --log-reward-category reward_cat
  --custom-generate-function-path generate_orena.generate
  --num-rollout 2000
  --rollout-batch-size 64
  --n-samples-per-prompt 8
  --rollout-max-response-len 512
  --rollout-temperature 0.7
  --global-batch-size 512
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

SGLANG_ARGS=(
  --rollout-num-gpus-per-engine 1
  --sglang-mem-fraction-static 0.6
)

if [ "$TRAIN_BACKEND" = "fsdp" ]; then
  BACKEND_ARGS=(
    --train-backend fsdp
    --gradient-checkpointing
    --attn-implementation flash_attention_3
  )
else
  BACKEND_ARGS=(
    --train-backend megatron
    --tensor-model-parallel-size 4
    --sequence-parallel
    --pipeline-model-parallel-size 1
    --context-parallel-size 1
    --expert-model-parallel-size 1
    --expert-tensor-parallel-size 1
    --recompute-granularity full
    --recompute-method uniform
  )
fi

TOPOLOGY_ARGS=(
  --actor-num-nodes 1
  --actor-num-gpus-per-node "$NUM_GPUS"
  --colocate
)

python3 train.py \
  "${CKPT_ARGS[@]}" \
  "${ROLLOUT_ARGS[@]}" \
  "${GRPO_ARGS[@]}" \
  "${OPTIMIZER_ARGS[@]}" \
  "${BACKEND_ARGS[@]}" \
  "${SGLANG_ARGS[@]}" \
  "${TOPOLOGY_ARGS[@]}"

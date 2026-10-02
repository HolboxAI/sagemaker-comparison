#!/usr/bin/env python3
"""Stage 3 — launch the 2 RL (GRPO) jobs.

Exp 2 (ORena VLM RL): reuses the prior ACTIVE evaluator
    `evaluation-orena-focus-reward` (same ground_truth schema).
Exp 4 (GoEmotions RL): blocked on `fs-or1-goemotions-reward` reaching ACTIVE;
    this script checks state and launches only if ACTIVE, else prints a note.

Logs everything to logs/32_rl_launch.log and records job ids in the run manifest.
"""
import os, json, datetime
from fireworks import Fireworks
from fireworks.types import (
    ReinforcementLearningLossConfig,
    TrainingConfig,
)
from fireworks.types.reinforcement_fine_tuning_job_create_params import InferenceParameters

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

log = open(f"{BASE}/logs/32_rl_launch.log", "w")
def emit(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True)
    log.write(line + "\n"); log.flush()

def state_of(ev_id):
    e = fw.evaluators.get(ev_id)
    return getattr(e, "state", None)

# ---- Exp 2: ORena VLM RL (GRPO) — evaluator already ACTIVE ----
# NOTE: qwen3p5-9b is NOT RL-tunable on managed RFT ("model is not tunable for
# reinforcement fine-tuning"). Vision RL has one managed shape: qwen3-vl-8b-instruct.
emit("===== Exp 2 — ORena VLM RL (GRPO) =====")
try:
    resp = fw.reinforcement_fine_tuning_jobs.create(
        account_id=ACCOUNT,
        dataset=f"accounts/{ACCOUNT}/datasets/fs-or1-orena-rl",
        evaluator=f"accounts/{ACCOUNT}/evaluators/evaluation-orena-focus-reward",
        reinforcement_fine_tuning_job_id="fs-or1-orena-rl",
        display_name="fs-or1-orena-rl",
        training_config=TrainingConfig(
            base_model="accounts/fireworks/models/qwen3-vl-8b-instruct",
            output_model=f"accounts/{ACCOUNT}/models/fs-or1-orena-rl",
            epochs=1,
            learning_rate=1e-5,
            lora_rank=16,
        ),
        loss_config=ReinforcementLearningLossConfig(method="GRPO", kl_beta=0.04),
        inference_parameters=InferenceParameters(
            response_candidates_count=8,
            max_output_tokens=512,
            temperature=0.7,
            extra_body='{"chat_template_kwargs":{"enable_thinking":false}}',
        ),
        chunk_size=200,
    )
    d = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
    emit("  CREATE OK")
    emit(json.dumps(d, default=str, indent=1))
except Exception as e:
    emit(f"  CREATE ERR {type(e).__name__}: {str(e)[:800]}")

# ---- Exp 4: GoEmotions RL (GRPO) — needs ACTIVE evaluator ----
emit("\n===== Exp 4 — GoEmotions RL (GRPO) =====")
ev_state = state_of("fs-or1-goemotions-reward")
emit(f"  evaluator fs-or1-goemotions-reward state={ev_state}")
if ev_state == "ACTIVE":
    try:
        resp = fw.reinforcement_fine_tuning_jobs.create(
            account_id=ACCOUNT,
            dataset=f"accounts/{ACCOUNT}/datasets/fs-or1-ge-rl",
            evaluator=f"accounts/{ACCOUNT}/evaluators/fs-or1-goemotions-reward",
            reinforcement_fine_tuning_job_id="fs-or1-ge-rl",
            display_name="fs-or1-ge-rl",
            training_config=TrainingConfig(
                base_model="accounts/fireworks/models/qwen3-0p6b",
                output_model=f"accounts/{ACCOUNT}/models/fs-or1-ge-rl",
                epochs=1,
                learning_rate=1e-5,
                lora_rank=16,
                max_context_length=8192,
            ),
            loss_config=ReinforcementLearningLossConfig(method="GRPO", kl_beta=0.04),
            inference_parameters=InferenceParameters(
                response_candidates_count=8,
                max_output_tokens=64,
                temperature=0.7,
                extra_body='{"chat_template_kwargs":{"enable_thinking":false}}',
            ),
            chunk_size=200,
        )
        d = resp.model_dump() if hasattr(resp, "model_dump") else str(resp)
        emit("  CREATE OK")
        emit(json.dumps(d, default=str, indent=1))
    except Exception as e:
        emit(f"  CREATE ERR {type(e).__name__}: {str(e)[:800]}")
else:
    emit("  SKIP — evaluator not ACTIVE; rebuild it first (see 33_rebuild_evaluator.py)")

log.close()
emit("\nDONE (log -> logs/32_rl_launch.log)")

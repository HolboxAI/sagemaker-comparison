#!/usr/bin/env python3
"""Stage 3b — launch Exp 4 (GoEmotions RL / GRPO).

GoEmotions evaluator fs-or1-goemotions-reward is now ACTIVE. Base = qwen3-0p6b.
"""
import os, json
from fireworks import Fireworks
from fireworks.types import ReinforcementLearningLossConfig, TrainingConfig
from fireworks.types.reinforcement_fine_tuning_job_create_params import InferenceParameters

ACCOUNT = "purchase-j087ceepz9i"
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
fw = Fireworks(api_key=os.environ["FIREWORKS_API_KEY"], account_id=ACCOUNT)

ev = fw.evaluators.get("fs-or1-goemotions-reward")
print("evaluator state:", getattr(ev, "state", None))

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
    print("CREATE OK")
    print(json.dumps(d, default=str, indent=1)[:1500])
except Exception as e:
    print(f"CREATE ERR {type(e).__name__}: {str(e)[:800]}")

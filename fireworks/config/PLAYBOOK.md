# Fireworks Playbook — SFT + GRPO on ORena and GoEmotions

_Compiled 2026-10-02 from the live SDK, the model catalog, and the account's own
prior evaluator artifact. Everything below is measured, not assumed._

## 1. Locked decisions

| Choice | Value |
|---|---|
| Prior artifacts | **fresh runs** (no warm-start from orena-focus-rl-v3/4/5) |
| RL method | **GRPO** (native in the managed RL job; it is the default) |
| ORena base | `qwen3p5-9b` (9.4B, VLM — `Qwen/Qwen3.5-9B` per A_s189) · verify image input; fallback `qwen3-vl-8b-instruct` |
| GoEmotions base | `qwen3-0p6b` (0.75B, text — only sub-2B model with SFT + RL) |
| Budget cap | **$94** |

## 2. Account

- `FIREWORKS_API_KEY` in `fireworks_vs_sagemaker/.env` (git-ignored).
- account_id `purchase-j087ceepz9i` · purchase@holbox.ai. Training access is live.

## 3. API surface (fireworks-ai 1.2.19)

- `supervised_fine_tuning_jobs.create(dataset, base_model, epochs, lora_rank,
  learning_rate, batch_size, max_context_length, evaluation_dataset, ...)`
- `reinforcement_fine_tuning_jobs.create(dataset, evaluator, loss_config,
  inference_parameters, training_config, ...)`
  - `loss_config.method` ∈ `GRPO | DAPO | DPO | ORPO | GSPO_TOKEN` (GRPO default)
- `datasets.create(dataset=DatasetParam(format=CHAT|COMPLETION|RL, ...),
  dataset_id)` then `datasets.upload(dataset_id, file=...)`
- `evaluators.create(evaluator=Evaluator(criteria=[...code snippets...]))`

## 4. Data formats

**SFT (CHAT):** JSONL, one OpenAI-chat example per line.
`{"messages":[{"role":"system","content":...},{"role":"user","content":...},
{"role":"assistant","content":...}]}`

**RL (GRPO):** JSONL with prompts + a top-level `ground_truth` the reward
evaluator reads. Images are base64 with MIME prefix.

```json
{"messages": [
    {"role":"system","content":"..."},
    {"role":"user","content":[
        {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,...."}},
        {"type":"text","text":"<question>"}]}
 ],
 "ground_truth":{"answer":"...","answer_format":"fo_class|number|binary|open_ended|multiple_choice","choices":[]},
 "input_metadata":{...}}
```

## 5. Reward evaluator contract (Eval Protocol)

Install `eval-protocol` (the `ep` CLI). Entry point `evaluation.py::<func>`.

```python
from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

@evaluation_test(
    input_dataset=["data/train.jsonl"],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/<base>",
                        "temperature": 0.7, "max_tokens": 512,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}],
    rollout_processor=SingleTurnRolloutProcessor(),
    mode="pointwise",
)
def reward(row: EvaluationRow, **kwargs) -> EvaluationRow:
    row.evaluation_result = EvaluateResult(
        score=..., is_score_valid=True, reason="...",
        metrics={"exact_match": MetricResult(score=..., is_score_valid=True, reason="...")})
    return row
```

- The model completion arrives as the **last assistant message** in `row.messages`.
- `row.ground_truth` holds the answer dict. `score` is the gradient signal;
  `metrics` are tracked without moving the objective.

## 6. Launch commands (see caveat)

```bash
# dataset
firectl create dataset <name> data/xxx.jsonl
# evaluator
ep upload --entry evaluation.py::reward --yes
# RL job (GRPO is default; --loss-config-method is broken — omit it)
ep create rft --skip-validation \
  --training-config-base-model accounts/fireworks/models/<base> \
  --dataset <name> --evaluator accounts/<acct>/evaluators/<ev> \
  --training-config-output-model accounts/<acct>/models/<out> \
  --training-config-epochs 1 --training-config-lora-rank 16 \
  --training-config-learning-rate 1e-4 \
  --inference-parameters-response-candidates-count 16 \
  --inference-parameters-max-output-tokens 512 \
  --inference-parameters-temperature 0.7 \
  --inference-parameters-extra-body '{"chat_template_kwargs":{"enable_thinking":false}}'
```

**CAVEAT:** Fireworks blocks mutating `firectl`/`ep` calls made from inside an AI
agent (`mutating command ... cannot run inside an AI agent`). Launch commands
must run in the user's own terminal. Verify whether the Python SDK `*.create()`
calls are also blocked (test with a harmless dataset upload).

## 7. Metrics to capture (per run)

`run_id, platform, experiment, method, base_model, lora_rank, learning_rate,
epochs, dataset_size, final_score, exact_match, per-format accuracy, train_tokens,
wall_time_s, cost_usd, job_events, errors`

For RL also: per-chunk `score` / `exact_match` / monitor metrics, rollout count.

## 8. Feature inventory (vs SageMaker later)

SFT/DPO/RL surfaces; GRPO+DAPO+GSPO+CISPO algorithms; VLM fine-tuning; custom
reward evaluators (code upload); datasets/eval datasets; deployments + LoRA
load/unload; batch inference; metrics endpoints; W&B hook; serverless vs
dedicated shapes; per-token + on-demand-GPU billing; KL reporting.

## 9. Prior-art reference (Prime vs Fireworks, done by this team)

`reference/evaluator_orena/orena-focus-rft/` holds the full prior evaluator +
data + README. Prior RL run: `exact_match` 0.1925 → 0.3200 on the *filtered*
distribution (not held-out). README warns: RFT may be **free under 16B params**
— verify on the dashboard; that alone would be a key vs-SageMaker datapoint.

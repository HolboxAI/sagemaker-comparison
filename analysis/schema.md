# Canonical metrics schema

Every platform writes one `metrics/final_results.json` in this schema. The
top-level `platform` field is the only field that differs by platform. All
nested field names must match exactly so the comparison script can join them.

```json
{
  "platform": "fireworks | sagemaker | prime-intellect",
  "generated_utc": "2026-10-02",
  "experiments": {
    "orena_sft": {
      "experiment": "orena_sft",
      "method": "sft | grpo",
      "job_id": "<platform job id>",
      "state": "JOB_STATE_COMPLETED | ...",
      "base_model": "<platform model id>",
      "hf_model_id": "Qwen/Qwen3.5-9B",
      "output_model": "<platform output model id>",
      "lora_rank": 64,
      "learning_rate": 1e-4,
      "max_context_length": 32768,
      "epochs": 1,
      "dataset_size": 10980,
      "steps": 344,
      "final_train_loss": 0.3542,
      "estimated_cost_usd": 1.47,
      "wall_time_s": 1680
    },
    "orena_rl": {
      "experiment": "orena_rl",
      "method": "grpo",
      "job_id": "...",
      "state": "JOB_STATE_COMPLETED",
      "base_model": "...",
      "hf_model_id": "Qwen/Qwen3-VL-8B-Instruct",
      "output_model": "...",
      "lora_rank": 16,
      "learning_rate": 1e-5,
      "dataset_size": 2000,
      "start_score": 0.2231,
      "final_score": 0.3108,
      "start_exact_match": 0.1906,
      "final_exact_match": 0.2819,
      "estimated_cost_usd": null,
      "wall_time_s": 2220,
      "score_curve": [0.223, 0.228, 0.231],
      "exact_match_curve": [0.191, 0.196, 0.195]
    },
    "goemotions_sft": {
      "experiment": "goemotions_sft",
      "method": "sft",
      "job_id": "...",
      "state": "JOB_STATE_COMPLETED",
      "base_model": "...",
      "hf_model_id": "Qwen/Qwen3-0.6B",
      "output_model": "...",
      "lora_rank": 16,
      "learning_rate": 1e-4,
      "max_context_length": 8192,
      "epochs": 1,
      "dataset_size": 43410,
      "steps": 1357,
      "final_train_loss": 0.3127,
      "estimated_cost_usd": 2.84,
      "wall_time_s": 3000
    },
    "goemotions_rl": {
      "experiment": "goemotions_rl",
      "method": "grpo",
      "job_id": "...",
      "state": "JOB_STATE_COMPLETED",
      "base_model": "...",
      "hf_model_id": "Qwen/Qwen3-0.6B",
      "output_model": "...",
      "lora_rank": 16,
      "learning_rate": 1e-5,
      "dataset_size": 5000,
      "start_jaccard": 0.1225,
      "final_jaccard": 0.3442,
      "start_micro_f1": 0.1746,
      "final_micro_f1": 0.3774,
      "estimated_cost_usd": null,
      "jaccard_curve": [0.123, 0.148, 0.157],
      "micro_f1_curve": [0.175, 0.208, 0.214]
    }
  },
  "findings": ["<platform-specific gaps, free-form strings>"]
}
```

## Field notes

- `estimated_cost_usd` is `null` when the platform does not surface a per-job
  cost (for example, Fireworks managed RL). Record the reason in `findings`.
- `score_curve` / `exact_match_curve` / `jaccard_curve` / `micro_f1_curve` are
  per-chunk averages in order. Use the same chunking where possible so the
  curves line up across platforms.
- `start_*` and `final_*` are the first and last chunk values.
- `hf_model_id` must be the shared Hugging Face id so different platforms map to
  the same model. `base_model` is the platform's own id string.

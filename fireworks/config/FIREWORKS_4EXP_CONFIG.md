# Fireworks → SageMaker — 4-Experiment Setup Config

_Compiled 2026-10-02. This is the portable configuration for the four runs below,
so the SageMaker team can reproduce them in **isolated** fashion and the results can
be compared one-to-one. Every value is the same on both platforms; only the
"platform model id" column changes._

---

## 1. Experiment matrix

| # | Experiment | Dataset | Method | Base model (Fireworks id) | Base model (HF / SageMaker id) | Params |
|---|---|---|---|---|---|---|
| 1 | ORena VLM SFT | ORena train | supervised fine-tuning (LoRA) | `accounts/fireworks/models/qwen3p5-9b` | `Qwen/Qwen3.5-9B` | 9.4B (VLM) |
| 2 | ORena VLM RL | ORena train | GRPO | `accounts/fireworks/models/qwen3-vl-8b-instruct` ⚠️ | `Qwen/Qwen3.5-9B` (SageMaker can do 9B; Fireworks can't) | 8.7B (VLM) |
| 3 | GoEmotions SFT | GoEmotions train | supervised fine-tuning (LoRA) | `accounts/fireworks/models/qwen3-0p6b` | `Qwen/Qwen3-0.6B` | 0.75B (text) |
| 4 | GoEmotions RL | GoEmotions train | GRPO | `accounts/fireworks/models/qwen3-0p6b` | `Qwen/Qwen3-0.6B` | 0.75B (text) |

Rules that apply to all four:
- **Fresh runs** — no warm-start / resume from any prior artifact on either platform.
- **RL method = GRPO** everywhere (native default in both Fireworks `reinforcement_fine_tuning_jobs`
  and SageMaker TRL `GRPOConfig`).
- **Budget cap = $94** across all four Fireworks runs.
- **Short runs for the $94 budget** — every epoch capped at **1** (SFT and RL); RL rollouts reduced to **8** per prompt; ORena RL uses a 2,000-prompt subset unless the bill allows more.
- Side-by-side SFT (experiments 1 and 3) runs **before** RL (2 and 4) on the same data.

---

## 2. Shared environment

| Item | Fireworks | SageMaker |
|---|---|---|
| Account / region | `purchase-j087ceepz9i` · purchase@holbox.ai | org account (isolated) |
| Auth | `FIREWORKS_API_KEY` env var | IAM role / `AWS_*` env |
| SDK | `fireworks-ai` 1.2.19 | `sagemaker` + `trl` + `peft` + `transformers` |
| Python | 3.11 | 3.11 |
| GPU (RL shape) | dedicated (see §4/§5 shapes) | matching A100/H100/B200 — pin to equivalent class |

**Data are identical on both platforms.** ORena and GoEmotions datasets are the same
files, same splits, same order (do not reshuffle between platforms).

---

## 3. Data

### 3.1 ORena (surgical VQA — vision + text)

- Source: `reference/evaluator_orena/orena-focus-rft/data/orena_train_full.jsonl`
  (10,980 rows, base64 JPEG inline). For a faster smoke run, `data/orena_train.jsonl`.
- Answer space (5 formats): `fo_class`, `number`, `open_ended`, `binary`, `multiple_choice`.
- Metric: **4-bucket unweighted mean accuracy** (`meanAccuracy`) over `{fo_class, number,
  open_ended, multiple_choice}` — see §8.

**SFT (CHAT) JSONL** — one OpenAI-chat example per line:

```json
{"messages": [
  {"role": "system", "content": "You are a surgical VQA assistant. Answer the question using the image. Keep the answer short and in the requested format."},
  {"role": "user", "content": [
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}},
     {"type": "text", "text": "<question>"}]},
  {"role": "assistant", "content": "<reference answer>"}
]}
```

**RL (GRPO) JSONL** — prompt + top-level `ground_truth` (the reward evaluator reads it):

```json
{"messages": [
  {"role": "system", "content": "You are a surgical VQA assistant. Answer the question using the image. Keep the answer short and in the requested format."},
  {"role": "user", "content": [
     {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,...."}},
     {"type": "text", "text": "<question>"}]}
 ],
 "ground_truth": {"answer": "<reference answer>", "answer_format": "fo_class|number|open_ended|binary|multiple_choice", "choices": []},
 "input_metadata": {}}
```

### 3.2 GoEmotions (text classification — multi-label)

- Source: `fireworks_vs_sagemaker/data/goemotions/` — `train.tsv` (43,410) / `dev.tsv`
  (5,426) / `test.tsv` (5,427). Format per line: `text \t <comma-separated label ids> \t <id>`.
- 28 labels (27 emotions + neutral). **Multi-label** (0..N labels per example).
- Mapping file `emotions.txt` provides id → label name.

**SFT (CHAT) JSONL** — the model outputs a comma-separated label list:

```json
{"messages": [
  {"role": "system", "content": "Classify the text into one or more of the 28 GoEmotions labels. Reply with a comma-separated list of label names only."},
  {"role": "user", "content": "<text>"},
  {"role": "assistant", "content": "joy, gratitude"}
]}
```

**RL (GRPO) JSONL**:

```json
{"messages": [
  {"role": "system", "content": "Classify the text into one or more of the 28 GoEmotions labels. Reply with a comma-separated list of label names only."},
  {"role": "user", "content": "<text>"}
 ],
 "ground_truth": {"labels": ["joy", "gratitude"]},
 "input_metadata": {}}
```

---

## 4. Experiment 1 — ORena VLM SFT

| Field | Value |
|---|---|
| Base model | `accounts/fireworks/models/qwen3p5-9b` / `Qwen/Qwen3.5-9B` |
| Dataset | ORena CHAT JSONL (§3.1) |
| Format | `CHAT` |
| Epochs | 1 |
| Learning rate | 1e-4 |
| LoRA rank | 64 |
| LoRA alpha | 128 |
| LoRA dropout | 0.05 |
| LoRA target modules | language model + `model.visual.*` + `model.visual.merger.*` (train the vision tower + projector, matching `A_s189/adapter_config.json`) |
| Batch size | micro 4 / effective 32 |
| Max context length | 65536 (image + text fits the 65k shape) |
| Optimizer | AdamW 8-bit, cosine decay, 10% warmup |
| Eval dataset | ORena held-out split |
| Early stop | patience 1 (no improvement on `meanAccuracy`) |
| Fireworks training shape | `qwen3p5-9b-65k-lora` (LoRA SFT) |
| SageMaker equivalent | PEFT LoRA on `Qwen/Qwen3.5-9B`, TRL `SFTTrainer` |

> Note for SageMaker team: A_s189 (the reference adapter) was trained as LoRA
> `r=64, alpha=128, dropout=0.05` on `Qwen/Qwen3.5-9B`, with `task_type=CAUSAL_LM`
> and a target regex covering both `model.language_model.*` and `model.visual.*`.
> Reproduce that exact adapter shape for a like-for-like comparison.

---

## 5. Experiment 2 — ORena VLM RL (GRPO)

| Field | Value |
|---|---|
| Base model | `accounts/fireworks/models/qwen3-vl-8b-instruct` / `Qwen/Qwen3.5-9B` |
| ⚠️ Platform constraint | **`qwen3p5-9b` is NOT RL-tunable** on Fireworks managed RFT (`model is not tunable for reinforcement fine-tuning`). Vision RL has one managed shape: `qwen3-vl-8b-instruct`. SageMaker TRL GRPO is not constrained this way — it can RL `Qwen/Qwen3.5-9B` directly. Record this as a Fireworks feature gap. |
| Dataset | ORena RL JSONL (§3.1) |
| Format | `RL` |
| Loss method | `GRPO` |
| Epochs | 1 |
| Learning rate | 1e-5 |
| LoRA rank | 16 |
| KL beta | 0.04 |
| Temperature | 0.7 |
| Max output tokens | 512 |
| Generations (rollouts) per prompt | 8 (Fireworks `response_candidates_count`; SageMaker `num_generations`) — reduced to 8 for the $94 budget |
| Reward evaluator | ORena exact-match + 4-bucket accuracy (§8.1) |
| Fireworks training shape | `qwen3p5-9b-65k-lora` (RL, 3× B200) |
| SageMaker equivalent | TRL `GRPOTrainer` with the same reward fn |

---

## 6. Experiment 3 — GoEmotions SFT

| Field | Value |
|---|---|
| Base model | `accounts/fireworks/models/qwen3-0p6b` / `Qwen/Qwen3-0.6B` |
| Dataset | GoEmotions CHAT JSONL (§3.2) |
| Format | `CHAT` |
| Epochs | 1 |
| Learning rate | 1e-4 |
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Batch size | micro 8 / effective 64 |
| Max context length | 8192 (short text) |
| Optimizer | AdamW 8-bit, cosine decay, 10% warmup |
| Eval dataset | GoEmotions dev split |
| Early stop | patience 1 (no improvement on micro-F1) |
| Fireworks training shape | `qwen3-0p6b-65k-lora` (LoRA SFT) |
| SageMaker equivalent | PEFT LoRA on `Qwen/Qwen3-0.6B`, TRL `SFTTrainer` |

---

## 7. Experiment 4 — GoEmotions RL (GRPO)

| Field | Value |
|---|---|
| Base model | `accounts/fireworks/models/qwen3-0p6b` / `Qwen/Qwen3-0.6B` |
| Dataset | GoEmotions RL JSONL (§3.2) |
| Format | `RL` |
| Loss method | `GRPO` |
| Epochs | 1 |
| Learning rate | 1e-5 |
| LoRA rank | 16 |
| KL beta | 0.04 |
| Temperature | 0.7 |
| Max output tokens | 64 |
| Generations (rollouts) per prompt | 8 (reduced for budget) |
| Reward evaluator | GoEmotions multi-label F1 / Jaccard (§8.2) |
| Fireworks training shape | `qwen3-0p6b-65k-lora` (RL, 2× B200) |
| SageMaker equivalent | TRL `GRPOTrainer` with the same reward fn |

---

## 8. Reward evaluators (RL only)

### 8.1 ORena evaluator — `evaluation.py::reward`

Contract (Eval Protocol on Fireworks; a plain function on SageMaker):

```python
from eval_protocol.models import EvaluateResult, EvaluationRow, MetricResult
from eval_protocol.pytest import SingleTurnRolloutProcessor, evaluation_test

@evaluation_test(
    input_dataset=["data/train.jsonl"],
    completion_params=[{"model": "fireworks_ai/accounts/fireworks/models/qwen3p5-9b",
                        "temperature": 0.7, "max_tokens": 512,
                        "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}],
    rollout_processor=SingleTurnRolloutProcessor(), mode="pointwise")
def reward(row: EvaluationRow, **kwargs) -> EvaluationRow:
    pred = normalize(row.messages[-1]["content"])      # last assistant message
    gt   = normalize(row.ground_truth["answer"])
    fmt  = row.ground_truth["answer_format"]
    exact = 1.0 if pred == gt else 0.0
    row.evaluation_result = EvaluateResult(
        score=exact, is_score_valid=True,
        reason=f"exact_match={exact}", 
        metrics={"exact_match": MetricResult(score=exact, is_score_valid=True, reason=""),
                 f"acc_{fmt}": MetricResult(score=exact, is_score_valid=True, reason="")})
    return row
```

- `normalize()` = lowercase, strip whitespace/punctuation, collapse spaces.
  For `number`: parse to float, tolerate `±0.01`. For `fo_class`/`multiple_choice`:
  exact string match after normalization. For `open_ended`: exact match after normalization.
- **Reward = exact match.** The 4-bucket `meanAccuracy` is computed off the per-format
  `acc_*` metrics post-run (unweighted mean of the 4 format buckets).

### 8.2 GoEmotions evaluator — `evaluation.py::reward`

```python
def reward(row, **kwargs):
    pred = set(parse_labels(row.messages[-1]["content"]))   # last assistant message
    gt   = set(row.ground_truth["labels"])
    inter = pred & gt
    union = pred | gt
    f1 = 2 * len(inter) / (len(pred) + len(gt)) if (pred or gt) else 1.0      # micro-F1
    jac = len(inter) / len(union) if union else 1.0                           # Jaccard
    row.evaluation_result = EvaluateResult(
        score=jac, is_score_valid=True, reason=f"f1={f1:.4f} jaccard={jac:.4f}",
        metrics={"micro_f1": MetricResult(score=f1, is_score_valid=True, reason=""),
                 "jaccard": MetricResult(score=jac, is_score_valid=True, reason="")})
    return row
```

- `parse_labels()` = split on commas, lowercase, strip, map synonyms to the canonical
  28 labels, drop unknowns. Empty prediction → empty set (reward 0).
- **Reward = Jaccard** of predicted vs ground-truth label set; `micro_f1` is tracked.

---

## 9. Metrics / logging schema (parity)

Every run emits one JSON object (append to a per-platform JSONL log) with **identical
keys on both platforms**, so results can be diffed directly:

```json
{
  "run_id": "<uuid>",
  "platform": "fireworks | sagemaker",
  "experiment": "orena_sft | orena_rl | goemotions_sft | goemotions_rl",
  "method": "sft | grpo",
  "base_model": "qwen3p5-9b | qwen3-0p6b",
  "hf_model_id": "Qwen/Qwen3.5-9B | Qwen/Qwen3-0.6B",
  "lora_rank": 64, "lora_alpha": 128, "lora_dropout": 0.05,
  "learning_rate": 1e-4, "epochs": 2,
  "batch_size_micro": 4, "batch_size_effective": 32,
  "max_context_length": 65536,
  "dataset": "orena_train_full", "dataset_size": 10980,
  "final_score": 0.0, "exact_match": 0.0, "mean_accuracy_4bucket": 0.0,
  "micro_f1": null, "per_format_accuracy": {}, "per_label_f1": {},
  "train_tokens": 0, "wall_time_s": 0.0, "cost_usd": 0.0,
  "job_events": [], "errors": []
}
```

For RL also append per-chunk: `score`, `exact_match`, `micro_f1`, rollout count, and any
KL / advantage monitor metrics the platform exposes.

---

## 10. Budget and limits

- Total Fireworks cap: **$94** across all four runs.
- Expected cost drivers: ORena VLM RL is the expensive one (3× B200, 10,980 prompts).
  Run experiment 1 (SFT) first, check the bill, then decide whether to run the full
  ORena RL set or a subset. GoEmotions (0.6B) is cheap.
- Do **not** leave dedicated GPU deployments running after jobs complete.

---

## 11. Feature inventory to compare (Fireworks vs SageMaker)

Check each item on both platforms and record yes/no + notes:
1. Managed SFT (LoRA) — dataset upload, launch, metrics endpoint.
2. Managed RL/GRPO with custom reward evaluator (code upload).
3. RL loss algorithms available: GRPO, DAPO, GSPO, DPO, ORPO, CISPO.
4. VLM (image-input) fine-tuning — SFT and RL.
5. LoRA hyperparameter surface exposed (rank, alpha, dropout, target modules).
6. Warm-start / resume from a prior adapter.
7. Deployment of the resulting adapter + LoRA load/unload at inference.
8. Batch inference.
9. W&B integration / external metric hook.
10. Billing model: per-token vs on-demand GPU, dedicated vs serverless shapes.
11. RL step-level metrics (KL, advantage, per-chunk score).

---

## 12. Open items / verification checklist

- [ ] **Verify Fireworks `qwen3p5-9b` accepts image input** (it is multimodal on HF —
      `Qwen/Qwen3.5-9B` has `model.visual` per A_s189 — but the Fireworks catalog carries
      no vision flag for it). If image input is rejected, fall back to
      `accounts/fireworks/models/qwen3-vl-8b-instruct` for experiments 1–2 and record the
      discrepancy as a finding.
- [ ] Confirm whether SDK `*.create()` calls are blocked from this AI agent (like `firectl`/
      `ep`); if so, all launches run in the user's terminal.
- [ ] Confirm RFT/RFT-under-16B pricing on the dashboard (prior README claims it may be free).
- [ ] Build GoEmotions CHAT + RL JSONL from `train.tsv` + `emotions.txt` (local, unblocked).
- [ ] Confirm the SageMaker team pins the **same GPU class** as the Fireworks shape for a
      fair wall-time / cost comparison.

---

## 13. Fireworks run status (launched 2026-10-02)

All four jobs are live. Fresh runs, 1 epoch each, GRPO with 8 rollouts/prompt for RL.

| # | Job id | State | Base model | LoRA rank | LR | Output model |
|---|---|---|---|---|---|---|
| 1 | `supervisedFineTuningJobs/etia90sh` | RUNNING (60%) | `qwen3p5-9b` | 64 | 1e-4 | `models/fs-or1-orena-sft` |
| 2 | `reinforcementFineTuningJobs/fs-or1-orena-rl` | RUNNING | `qwen3-vl-8b-instruct` | 16 | 1e-5 | `models/fs-or1-orena-rl` |
| 3 | `supervisedFineTuningJobs/ktcdqquf` | RUNNING (36%) | `qwen3-0p6b` | 16 | 1e-4 | `models/fs-or1-ge-sft` |
| 4 | `reinforcementFineTuningJobs/fs-or1-ge-rl` | RUNNING | `qwen3-0p6b` | 16 | 1e-5 | `models/fs-or1-ge-rl` |

Evaluators: `evaluation-orena-focus-reward` (prior, ACTIVE) and `fs-or1-goemotions-reward` (ACTIVE).

### Findings that change the comparison

1. **`qwen3p5-9b` is SFT-tunable but NOT RL-tunable** on Fireworks managed RFT.
   Experiment 2 therefore runs on `qwen3-vl-8b-instruct` — the *only* managed
   vision-RL shape. On SageMaker this constraint does not exist. → Fireworks
   managed RL has a narrower vision-model catalog than SageMaker TRL GRPO.

2. **Custom reward evaluators must be uploaded as a single `.tar.gz` tarball**
   via `evaluators.create(evaluator_id=…, evaluator={display_name, description,
   entry_point})` — *without* a `source`/`requirements` field — then a single
   signed-URL PUT. Uploading loose `evaluation.py` + `requirements.txt` files
   (or setting `source=TYPE_UPLOAD`) makes the remote build fail with an opaque
   `BUILD_FAILED status=INTERNAL` and no build log. The `mcp<2` pin belongs in
   the *local* `ep` CLI env only, not in the remote `requirements.txt`.

```

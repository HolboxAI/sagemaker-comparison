# Fireworks vs SageMaker — Fireworks Phase Plan

_Planned 2026-10-02. This phase covers Fireworks only. SageMaker comes later,
using the same scripts, data, and metric schema so the two are comparable._

## 1. Goal

Map why Fireworks is more feature-rich than SageMaker for fine-tuning + RL.
We do this by measuring, not by reading marketing pages.

Two experiments:

1. **ORena** — a vision-language model (VLM). Surgical visual question
   answering. Images + question → answer.
2. **GoEmotions** — a text model. Sentiment / emotion classification.
   Comment → one or more of 28 emotion labels.

Both run SFT (supervised fine-tuning) and RL. Both run for 1-2 epochs or
15-20 minutes, whichever comes first.

## 2. What we deliver

| Artifact | Purpose |
|---|---|
| Scripts + configs | Reproduce the run exactly on SageMaker later |
| Steps log | Record every command and decision |
| Training telemetry | Loss, eval metric, token counts, wall time, cost |
| Platform feature inventory | What Fireworks exposes vs SageMaker |
| Graphs | Side-by-side comparison |

## 3. Experiment 1 — ORena VLM (surgical VQA)

### Data
- `orena-toolkit/data/runB/frames_train_true.parquet` — 17,748 rows.
- Columns: `question`, `answer`, `answer_format`, `procedure_type`, and more.
- Answer formats: `fo_class` (45.6%), `number` (27.7%), `open_ended`,
  `binary`, `multiple_choice`.
- Input = one JPEG frame + question. Output = answer.
- Frames come from two public HuggingFace datasets:
  `orena-dkfz/heico-focus-vqa` and `orena-dkfz/lapchole-focus-vqa`.

### Method
- **SFT (LoRA)** on a Fireworks VLM. Candidate base:
  `accounts/fireworks/models/qwen2p5-vl-32b-instruct` (docs example) or a
  smaller `qwen2.5-vl-7b`. Confirm against the live model matrix.
- **RL (GRPO)** with a correctness reward (answer matches ground truth).

### Metrics
- SFT loss (per step), eval loss.
- Accuracy per `answer_format`, plus the 4-bucket mean (matches ORena metric).
- Training tokens, wall time, cost.

### Caveat
RL on a VLM may not be in Fireworks' supported matrix. Check first. If not
supported, run RL on the text classifier only and record the gap as a finding.

## 4. Experiment 2 — GoEmotions text classification

### Data
- Kaggle `debarshichanda/goemotions`.
- ~58k Reddit comments. 28 labels (27 emotions + neutral). Multi-label.

### Method
- Base: small LLM — Llama 3.1 8B Instruct or Qwen2.5 7B Instruct.
- **SFT (LoRA)**: comment → comma-separated emotion list.
- **RL (GRPO)**: reward = F1 / exact match on predicted label set.

### Metrics
- SFT loss, eval loss, accuracy, macro-F1.
- Training tokens, wall time, cost.

## 5. Instrumentation (same schema for both platforms later)

Capture every run in one JSON/CSV per experiment:

```
run_id, platform, experiment, method, base_model, lora_r, lora_alpha,
learning_rate, epochs, steps, dataset_subset_size, train_tokens,
final_loss, eval_metric, wall_time_s, cost_usd, job_events, errors
```

Also capture a **feature inventory** per platform:

- Training surface (managed vs training API; serverless vs dedicated)
- LoRA / full-parameter support
- RL algorithms offered (GRPO, DAPO, GSPO, CISPO, DPO, ORPO)
- Model catalog size, incl. VLM support
- Loss / metric export (does it stream? is it a graph? can we download?)
- Job events and logs
- Serving the fine-tuned model (serverless / dedicated / multi-LoRA)
- Cost transparency (per-token, per-GPU-hour)
- KL-divergence reporting (Fireworks publishes train/inference KL)

## 6. Fireworks facts (measured from docs, 2026-10)

- Managed Training = SFT, DPO, ORPO, RFT. **LoRA only.**
- Training API = full control, incl. RL. Serverless Training is GA.
  Dedicated (full-param) is private preview.
- **RL (GRPO/DAPO/GSPO/CISPO) runs on the Training API only.**
- VLM SFT is supported. Images must be base64 in JSONL (no URLs).
- Pricing: SFT $0.50/1M tokens (≤16B); DPO $1.00; RL = on-demand GPU rate.
- SDK: `pip install fireworks-ai` (use `--pre` for Training API). CLI: `firectl`.
- Training needs a **training-scoped** API key (inference-only keys get 401).

## 7. Steps

1. Install SDK + CLI: `pip install fireworks-ai` and `firectl`, `kaggle`.
2. Set `FIREWORKS_API_KEY` (training scope). Verify training access.
3. Build JSONL:
   - ORena: subset rows, encode frame JPEG → base64, chat format.
   - GoEmotions: subset rows, chat format.
4. Upload dataset (`firectl dataset create` or SDK).
5. Launch SFT job. Poll. Capture loss + metrics + events.
6. Launch RL (GRPO) job. Poll. Capture reward + metrics + events.
7. Deploy fine-tuned model. Measure latency + throughput.
8. Export all telemetry to JSON/CSV.
9. Plot graphs (loss, eval, cost, time).

## 8. Open decisions (need confirmation)

1. Base model for each experiment.
2. RL method: GRPO (reward) vs DPO (preference).
3. Data subset size to hit the 15-20 min target.
4. Budget ceiling (RL bills GPU time).
5. Whether ORena train frame JPEGs are already on disk or need re-extraction.

## 9. What we need (see response checklist)

- Fireworks API key (training scope) + account with Training API access.
- Kaggle API key (or manual GoEmotions download).
- HF token (only if the ORena datasets are gated; they are public).
- ORena train frames: path on disk, or permission to re-extract from video.
- AWS credentials (for the later SageMaker phase, not now).

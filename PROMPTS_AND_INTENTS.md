# Prompts & Intents — SageMaker 4-Experiment Session

Covers two things: (1) the prompts given during this session, in order, that drove the
SageMaker training from environment setup to the final result — each with its instance,
short intent, and short error; and (2) the per-experiment training prompt and intent.

**Session model:** this session was run with **DeepSeek-V4-Pro**.

---

## 1. Session prompts → instances (chronological, with intent + error)

| # | Prompt / step | Instance(s) | Error (short) | Intent (short) |
|---|---|---|---|---|
| 1 | **Run the 4 experiments** (ORena SFT/RL, GoEmotions SFT/RL) on SageMaker script-mode — account `751871643798`, profile `stanford_gpu`, `us-east-1` | GE-SFT `ml.g4dn.xlarge` · GE-RL `ml.g5.xlarge` · ORena-RL `ml.g5.2xlarge` · ORena-SFT `ml.g5.12xlarge` | — | Set up IAM role + S3 channels and launch all 4 demo jobs |
| 2 | ORena SFT re-attempt | `ml.g5.12xlarge` | `pyarrow` "cannot mix list and non-list" | Plain-string content + separate `images` column |
| 3 | ORena SFT re-attempt | `ml.g5.12xlarge` | `DataParallel` device mismatch `cuda:0` vs `cuda:1` | Pin to single GPU (`CUDA_VISIBLE_DEVICES=0`) |
| 4 | ORena SFT re-attempt | `ml.g5.12xlarge` | Same mismatch — env var silently dropped | Wire `environment` pass-through into the estimator |
| 5 | ORena RL re-attempt | `ml.g5.2xlarge` | `UnicodeDecodeError` (ascii, model-card save) | Force UTF-8 (`PYTHONUTF8=1` + locale) → ✅ completed |
| 6 | **"…ignore the non similarities"** (hyperparameter / GPU differences) | all four | — | Record GPU-forced deltas as findings; don't change them |
| 7 | ORena SFT re-attempt | `ml.g5.12xlarge` | `torch.OutOfMemoryError` (~200 MB over 24 GB) | Resolve 9B bf16 OOM |
| 8 | **"Cheap bf16 fix"** | `ml.g5.12xlarge` | (the OOM above) | LoRA rank 64→32/64 + `expandable_segments` → ✅ loss 1.494 |
| 9 | **"…push relevant files … through vishnu-holbox … don't add claude as co-author"** | all four (artifacts) | — | Commit logs + metrics + config to GitHub |

Retries were concentrated on **ORena SFT (5 attempts)** and **ORena RL (2 attempts)**;
GoEmotions SFT and RL completed on their first documented run. Every retry stayed on the
same instance type — the fixes were code/config changes, not instance changes.

---

## 2. Experiment prompts → intent (training prompts)

| # | Experiment | Training prompt | Intent |
|---|---|---|---|
| 1 | ORena VLM SFT | **System:** "You are a surgical assistant that analyzes surgical video frames. Be precise and concise." **User:** `<image>` + surgical question (e.g. "Which combination of foreign object classes is visible in this frame? …"). **Assistant:** answer (e.g. "Clip, Silicone loop") | Supervised fine-tune Qwen3.5-9B (vision + language) to answer surgical VQA questions from endoscopic frames |
| 2 | ORena VLM RL (GRPO) | **User:** `<image>` + "You are a surgical assistant. You are given an endoscopic frame from a minimally invasive procedure. … [8 foreign-object class defs] Question: {question} Expected Format: {fmt} Instruction: {instruction} Return ONLY the answer. Do NOT explain." | GRPO on the same surgical VQA task, optimizing a verifiable per-format reward (fo_class micro-F1, number/binary/multiple-choice exact, open-ended token-F1) |
| 3 | GoEmotions SFT | **System:** "Classify the text into one or more of the 28 GoEmotions labels. Reply with a comma-separated list of label names only." **User:** text (e.g. "WHAT ARE THOOOOOSE"). **Assistant:** "anger, surprise" | Supervised fine-tune Qwen3-0.6B for 28-label multi-label emotion classification |
| 4 | GoEmotions RL (GRPO) | **System:** "Classify the text into one or more of the 28 GoEmotions labels. Reply with a comma-separated list of label names only." **User:** text → generate comma-separated labels | GRPO to maximize Jaccard overlap between predicted and ground-truth label sets |

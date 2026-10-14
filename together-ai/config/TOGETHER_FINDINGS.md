# Together AI — CholecT50 SFT → RLVR feasibility findings

**Date:** 2026-10-12
**Status:** ❌ not viable for the RL stage. Feasibility check only — nothing was run.

## 1. What we checked

Using the Together API we verified:

- The API key is valid (fine-tuning endpoint reachable, 265 models listed, no
  existing fine-tunes).
- The model catalog for our target (`Qwen3-VL`) — **absent**.
- `Qwen2.5-VL` / `Qwen2-VL` — present **for inference only** (not fine-tunable).
- The fine-tuning methods offered — **SFT and DPO only**.
- The documented "Custom Training" (RL) product — gated behind "Request access".

## 2. Fine-tuning methods (self-serve)

| Method | Available? |
|---|---|
| SFT (supervised fine-tuning) | ✅ |
| DPO (direct preference optimization) | ✅ |
| RLHF / RLVR / GRPO / custom reward | ❌ not offered |

There is **no** GRPO/RLVR path and **no** way to attach a custom reward function
like our triplet-F1 scorer. That removes the core of the project.

## 3. Vision-language fine-tuning models

Together can fine-tune these VLMs: Qwen3.8-27B, Qwen3.5-9B, Qwen3.5-4B, Qwen3.6-27B,
Qwen3.5-27B, Gemma 4 31B VLM, Llama 4 Scout/Maverick VLM.

**Not available for fine-tuning:** Qwen3-VL (absent entirely), Qwen2.5-VL and
Qwen2-VL 72B (inference-only). So even the SFT stage would need a model swap
(Qwen3.5-4B or Qwen3.5-9B).

## 4. The "Custom Training" RL beta

Together markets an RL product ("GRPO, custom losses, reward and KL tracking")
under **Custom Training**, but it is:

- gated behind **"Request access"** (sales, not self-serve via API key),
- described around text/conversation tuning — **no vision-language mention**,
- **unpriced**.

Even if access were granted, vision-language RL is not documented.

## 5. Bottom line

| Goal | Possible on Together? |
|---|---|
| SFT a VLM on CholecT50 | ✅ (Qwen3.5-4B/9B, model swap required) |
| RL with verifiable reward (RLVR) | ❌ |
| SFT → RL → before/after F1 demo | ❌ |

**No cost can be quoted** for the RL stage because it cannot run here. Together AI
is not a fit for this project.

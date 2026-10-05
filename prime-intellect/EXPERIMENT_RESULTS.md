# Experiment Results — Prime Intellect (Fireworks-parity, 4 experiments)

**Last updated:** 2026-10-05
**Status:** 2 of 4 experiments complete (GoEmotions SFT + RL); ORena SFT blocked by a pod-image version gap; ORena RL feasible but not started.

---

## 0. How to read this

The 4 reference experiments are reproduced on Prime Intellect via **on-demand pods** (`prime pods` → SSH → self-run `prime-rl`), *not* the hosted "training runs/jobs" surface. So results here come from pod logs, not the dashboard. Parity rule (from `DECISIONS_LOG.md` D6): every reference hyperparameter is kept identical; we shrink **only** data volume and step count to fit the 20–30 min wall.

---

## 1. Status at a glance

| Exp | Task | Model | Method | Status | Final metric |
|---|---|---|---|---|---|
| 1 | ORena SFT | Qwen3.5-9B (VLM) | LoRA r=64 | 🚫 **BLOCKED** (VLM SFT version drift) | — |
| 2 | ORena GRPO | Qwen3.5-9B (VLM) | LoRA r=16 | ⏳ not started (feasible) | meanAccuracy |
| 3 | GoEmotions SFT | Qwen3-0.6B | LoRA r=16 | ✅ **COMPLETE** | micro-F1 **0.4956** / Jaccard 0.3294 |
| 4 | GoEmotions GRPO | Qwen3-0.6B | LoRA r=16 | ✅ **COMPLETE** | micro-F1 **0.0349** / Jaccard 0.0178 |

---

## 2. Exp 3 — GoEmotions SFT (✅ COMPLETE)

**Config** (`configs/sft_goemotions.toml`):
- model `Qwen/Qwen3-0.6B`; LoRA r=16 α=32 dropout=0.05 (10,092,544 adapter params over 440.4M base)
- data: 800 train rows (`/workspace/data/goemotions/chat`), seq_len 256, batch 64 / micro 8, loss_mask = assistant-only
- optim adamw lr 1e-4; scheduler cosine warmup 10; max_steps 100
- renderer auto → **Qwen3Renderer** (detected automatically)
- `[ckpt.weights] save_adapter_separately = true` (see bug note below)

**Final loss curve (re-run):**

| Step | Loss |
|---|---|
| 1 | 5.7935 |
| 6 | 3.2676 |
| 25 | 0.4760 |
| 50 | 0.2648 |
| 75 | 0.2084 |
| 100 | **0.1672** |

- Loss fell **5.79 → 0.167** over 100 steps (real learning). Throughput ~2,800 tok/s; peak VRAM **3.3 / 39.5 GiB**.

**Final held-out metrics (200-row dev set, greedy decode):**

| Metric | Value |
|---|---|
| micro-F1 | **0.4956** |
| micro-Jaccard | 0.3294 |
| mean instance-F1 | 0.4948 |
| mean instance-Jaccard | 0.4708 |

*Greedy decode with `enable_thinking=False`; labels parsed from comma-separated output; TP/FP/FN = 113/104/126.*

**⚠️ Platform bug found & worked around:** prime-rl's default `save_adapter_separately = false` **silently drops the LoRA adapter** at checkpoint time — `clean_lora_state_dict` deletes `lora_A`/`lora_B` and keeps only the frozen base weights (despite the docstring claiming "before merging into full model weights"). Result: the saved `model.safetensors` is the *un-fine-tuned base model*, and a naive eval scores micro-F1 ≈ **0.02** (base-model behavior). Fix: re-run with `save_adapter_separately = true`, then merge `lora_adapters/adapter_model.safetensors` into the base (`W' = W + (B@A)·α/r`, α/r = 2.0). After merge, micro-F1 = **0.4956**.

---

## 3. Exp 4 — GoEmotions GRPO (✅ COMPLETE)

**Config** (`configs/rl_goemotions.toml`):
- model `Qwen/Qwen3-0.6B`; LoRA r=16 α=32 dropout=0.05; optim lr 1e-5
- 800 prompts, batch 64 / group 8 (8 rollouts/prompt), `max_steps = 64`
- sampling temperature 0.7, `max_completion_tokens = 64`
- reward = multi-label Jaccard (taskset `goemotions_v1`)
- `[orchestrator.renderer] name="qwen3" enable_thinking=false`
- `[inference] gpu_memory_utilization=0.6` + `[inference.model] max_model_len=512`
- `save_adapter_separately=true`

**Final held-out metrics (200-row dev set, greedy decode, thinking disabled):**

| Metric | Value |
|---|---|
| micro-F1 | **0.0349** |
| micro-Jaccard | 0.0178 |

**Training reward curve (`jaccard_curve`, per-step mean reward):** flat/noisy, range 0–0.073,
no upward trend. A fresh `Qwen3-0.6B` base model + Jaccard reward over 64 steps is not enough
signal to move the label-set prediction — versus Fireworks' 0.175 → 0.377 (micro-F1) on 5000
prompts.

**Three launch blockers fixed before the run completed:**
1. **Thinking mode** — Qwen3's renderer opens a `<think>` block by default, so a fresh model
   spends the whole 64-token budget "reasoning" and never emits the answer (Truncation 100%,
   reward 0). `enable_thinking=false` dropped truncation to ~0-3%.
2. **zero-advantage filter** — GRPO's default `post_batch_filters` enforce `zero_advantage`,
   which drops every group whose rollouts share a reward (the common case for a fresh model)
   and aborts with "10 consecutive empty train batches". Overriding to `enforce=false` lets
   training proceed.
3. **Single-GPU vLLM memory** — the trainer and inference server share physical GPU 0
   (`CUDA_VISIBLE_DEVICES="0,0"` hack to satisfy the disjoint-GPU check); vLLM's default
   `gpu_memory_utilization=0.9` fails because the trainer already holds ~3.8 GiB. Lowered to
   0.6 and capped `max_model_len=512`.

---

## 4. Exp 1 — ORena SFT (🚫 BLOCKED — VLM SFT version drift)

The pod image ships **prime-rl 0.7.0**, whose SFT data path is renderer-only and rejects
`[model.vlm]`:

> `validate_renderer_vs_vlm: "renderer-only SFT does not support VLMs yet. The renderer
> tokenizes text-only message dicts client-side and cannot handle image inputs."`

GitHub `main` has **removed** this validator (its only remaining VLM/SFT constraint is
`freeze_vision_encoder=false` being LoRA-incompatible), so ORena SFT is blocked by the **pod
image version**, not by the platform itself. This is the mirror of Fireworks' gap: Fireworks
cannot RL `qwen3p5-9b`; Prime Intellect 0.7.0 cannot SFT a VLM but **can** RL one (the RL
config carries `vlm` and rolls out through vLLM, which handles images natively). A prime-rl
upgrade on the pod was judged too risky; recorded as a version-drift finding instead.

---

## 5. Final results (parity table)

| Exp | Metric | Fireworks (ref) | Prime Intellect | Δ |
|---|---|---|---|---|
| 1 ORena SFT | meanAccuracy (4-bucket) | — | 🚫 blocked (VLM SFT version drift) | — |
| 2 ORena RL | meanAccuracy (4-bucket) | — | ⏳ not started | — |
| 3 GoEmotions SFT | micro-F1 | 0.313 (train loss; no F1 reported) | **0.4956** (train loss 0.167) | not like-for-like (800 vs 43k samples) |
| 4 GoEmotions RL | micro-F1 / Jaccard | 0.175 → 0.377 (F1) | **0.0349** / 0.0178 | fresh 0.6B + Jaccard over 64 steps ≈ no signal |

---

## 6. Billing / resource visibility (headless)

- **Balance:** $48.79 USD (wallet `cmuqrcy9q003v5jp12qivuuu9`)
- **Pod spend ≈ $20 total:**
  | Pod | GPU | Rate | Used for | Spend |
  |---|---|---|---|---|
  | facc8311… / 0c22b573… | A100 | — | key/setup smoke (2 recreations) | $0.34 + $0.38 |
  | 234fc42f… | A100 40GB | $1.99/hr | Exp 3 SFT | ~$1.22 |
  | 484b6053… | A10 24GB | ~$1.24/hr | Exp 4 RL (+ 3 failed launches) | $4.16 |
  | 19b97318… | H200 141GB | $4.50/hr | provisioned for ORena, **idle** (blocked) | $15.51 |

  The H200 was the single largest cost and produced no run — a direct consequence of the
  VLM SFT version-drift block being discovered *after* provisioning. Both live pods are
  terminated; no further billing accrues.
- **Why the web UI shows nothing:** these are on-demand **pods**, not hosted **training runs**
  (`prime train list` = empty). The pods/credits are visible via `prime pods list` and
  `prime wallet`; they may not be surfaced in the dashboard's "Jobs" view.

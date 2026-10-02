# Experiment Results — Prime Intellect (Fireworks-parity, 4 experiments)

**Last updated:** 2026-10-03 (during Exp 3 run)
**Status:** 1 of 4 experiments in progress; no final metrics yet.

---

## 0. How to read this

The 4 reference experiments are reproduced on Prime Intellect via **on-demand pods** (`prime pods` → SSH → self-run `prime-rl`), *not* the hosted "training runs/jobs" surface. So results here come from pod logs, not the dashboard. Parity rule (from `DECISIONS_LOG.md` D6): every reference hyperparameter is kept identical; we shrink **only** data volume and step count to fit the 20–30 min wall.

---

## 1. Status at a glance

| Exp | Task | Model | Method | Status | Final metric |
|---|---|---|---|---|---|
| 1 | ORena SFT | Qwen3.5-9B (VLM) | LoRA r=64 | ⏳ not started | meanAccuracy |
| 2 | ORena GRPO | Qwen3.5-9B (VLM) | LoRA r=16 | ⏳ not started | meanAccuracy |
| 3 | GoEmotions SFT | Qwen3-0.6B | LoRA r=16 | 🟢 **running** (step ~60/100) | micro-F1 / Jaccard |
| 4 | GoEmotions GRPO | Qwen3-0.6B | LoRA r=16 | ⏳ not started | micro-F1 / Jaccard |

---

## 2. Exp 3 — GoEmotions SFT (de-risk run, in progress)

**Config** (`configs/sft_goemotions.toml`):
- model `Qwen/Qwen3-0.6B`; LoRA r=16 α=32 dropout=0.05 (10,092,544 adapter params over 440.4M base)
- data: 800 train rows (`/workspace/data/goemotions/chat`), seq_len 256, batch 64 / micro 8, loss_mask = assistant-only
- optim adamw lr 1e-4; scheduler cosine warmup 10; max_steps 100
- renderer auto → **Qwen3Renderer** (detected automatically)

**Live training curve (loss by step):**

| Step | Loss | LR |
|---|---|---|
| 1 | 5.7935 | 1e-12 |
| 6 | 3.3650 | 5e-05 |
| 10 | 1.1570 | 9e-05 |
| 16 | 0.6150 | 9.9e-05 |
| 22 | 0.5017 | 9.6e-05 |
| 33 | 0.4067 | 8.6e-05 |
| 44 | 0.3052 | 7.0e-05 |
| 49 | 0.3063 | 6.2e-05 |
| 60 | 0.2457 | — |

- Loss fell **5.79 → 0.25** over 60 steps (real learning, not memorization noise).
- Throughput ~2,800 tok/s; peak VRAM **3.3 / 39.5 GiB** (LoRA is cheap on A100 40GB).

**To do when it finishes:** greedy-decode the 200-row dev set with the LoRA adapter → compute **micro-F1 + Jaccard** via `rewards.py` → record in §4.

---

## 3. Smoke test (platform validation, completed)

| Item | Result |
|---|---|
| Config | valid (after removing `[run]`, which the pod's prime-rl version rejects) |
| Model load | `Qwen/Qwen3-0.6B` downloaded + loaded |
| Training loop | 10 fake-data steps, "SFT trainer finished" |
| Peak mem | 11.1 GiB |
| Throughput | ~936 tok/s |

This de-risked the platform before the real runs.

---

## 4. Final results (parity table)

*Filled in as scores land. Fireworks reference values to be pulled from `FIREWORKS_4EXP_CONFIG.md` §9.*

| Exp | Metric | Fireworks (ref) | Prime Intellect | Δ |
|---|---|---|---|---|
| 1 ORena SFT | meanAccuracy (4-bucket) | — | — | — |
| 2 ORena RL | meanAccuracy (4-bucket) | — | — | — |
| 3 GoEmotions SFT | micro-F1 | — | — | — |
| 4 GoEmotions RL | micro-F1 / Jaccard | — | — | — |

---

## 5. Billing / resource visibility (headless)

- **Balance:** $48.79 USD (wallet `cmuqrcy9q003v5jp12qivuuu9`)
- **Billings (3 total):**
  | When | Pod | Amount |
  |---|---|---|
  | 2026-10-02 17:29 | facc8311… | $0.34 |
  | 2026-10-02 18:05 | 0c22b573… | $0.38 |
  | 2026-10-02 18:30 | 234fc42f… (current) | $0.50 |
  | **Total** | | **$1.22** |
- **Running pod:** `234fc42f…` A100 40GB, $1.99/hr.
- **Why the web UI shows nothing:** these are on-demand **pods**, not hosted **training runs** (`prime train list` = empty). The pods/credits are visible via `prime pods list` and `prime wallet`; they may not be surfaced in the dashboard's "Jobs" view.
